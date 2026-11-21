"""Blast Radius selector v2: coverage + AST-diff test selection.

Build time (`python selector.py --prep`, run by the Dockerfile, not charged):
  reads maps/maps.json.gz (coverage/open/class maps made by tools/ from the pinned
  commit) and the pristine sqlglot source, builds a reference index (which
  statements mention which names), and pickles everything to /app/index.pkl.

Run time: for each /data/commits/*.patch, apply it in memory to the pristine
source, diff old/new ASTs, turn every change into a set of tests through the maps,
then pack the tests onto workers (LPT, best worker count). Anything unexpected
falls back to running everything for that commit. stdlib only; logs to stderr.
"""

import ast
import difflib
import gzip
import json
import os
import pickle
import re
import sys
from pathlib import Path

DATA = Path(os.environ.get("BR_DATA", "/data"))
APP = Path(__file__).resolve().parent
SRC = Path(os.environ.get("BR_SRC", APP / "src"))
MAPS = APP / "maps" / "maps.json.gz"
INDEX = Path(os.environ.get("BR_INDEX", APP / "index.pkl"))
MAX_WORKERS = 8
WORKER_COST_MS = 2000
MAX_DEPTH = 4


def log(*a):
    print(*a, file=sys.stderr)


# ----------------------------------------------------------------------------- packing


def lpt(items, k):
    workers = [[] for _ in range(k)]
    loads = [0] * k
    for ms, number in items:
        i = loads.index(min(loads))
        workers[i].append(number)
        loads[i] += ms
    workers = [w for w in workers if w]
    return max(loads) + WORKER_COST_MS * len(workers), workers


def pack(numbers, durations):
    items = sorted(((durations[n], n) for n in set(numbers)), reverse=True)
    if not items:
        return []
    best = None
    for k in range(1, min(MAX_WORKERS, len(items)) + 1):
        cost, workers = lpt(items, k)
        if best is None or cost < best[0]:
            best = (cost, workers)
    return best[1]


# ----------------------------------------------------------------------------- AST helpers


def module_name(rel):
    p = rel[:-3].replace("/", ".")
    return p[: -len(".__init__")] if p.endswith(".__init__") else p


def strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            b = node.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) and isinstance(b[0].value.value, str):
                node.body = b[1:] or [ast.Pass()]
    return tree


def dump(node):
    return ast.dump(node, annotate_fields=False)


def first_line(node):
    decos = getattr(node, "decorator_list", None)
    return min([node.lineno] + [d.lineno for d in decos]) if decos else node.lineno


DEF_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def defined_names(stmt):
    """Names a module/class-level statement binds (not looking into function bodies)."""
    out = set()
    if stmt is None:
        return out

    def target(t):
        if isinstance(t, ast.Name):
            out.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                target(e)
        elif isinstance(t, ast.Starred):
            target(t.value)
        elif isinstance(t, ast.Attribute):
            out.add(t.attr)
        elif isinstance(t, ast.Subscript):
            target(t.value)

    def visit(s):
        if isinstance(s, DEF_TYPES):
            out.add(s.name)
        elif isinstance(s, ast.Assign):
            for t in s.targets:
                target(t)
        elif isinstance(s, (ast.AnnAssign, ast.AugAssign)):
            target(s.target)
        elif isinstance(s, ast.Import):
            for a in s.names:
                out.add(a.asname or a.name.split(".")[0])
        elif isinstance(s, ast.ImportFrom):
            for a in s.names:
                out.add(a.asname or a.name)
        elif isinstance(s, ast.Delete):
            for t in s.targets:
                target(t)
        elif isinstance(s, (ast.For, ast.AsyncFor)):
            target(s.target)
        elif isinstance(s, (ast.With, ast.AsyncWith)):
            for it in s.items:
                if it.optional_vars is not None:
                    target(it.optional_vars)
        if not isinstance(s, DEF_TYPES):
            for field in ("body", "orelse", "finalbody", "handlers"):
                for c in getattr(s, field, []) or []:
                    visit(c)
            if isinstance(s, ast.Match):
                for case in s.cases:
                    for c in case.body:
                        visit(c)

    visit(stmt)
    return out


def chain(node):
    """a.b.c -> ['a','b','c'] when the expression is a pure Name/Attribute chain."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return parts[::-1]
    return None


# ----------------------------------------------------------------------------- reference index


class RefIndexer:
    """Collects, for one source file, every Name/Attribute reference that resolves to a
    sqlglot/tests module-level or class-level object, with the statement unit it is in."""

    def __init__(self, rel, tree, units, refs, classinfo):
        self.rel = rel
        self.mod = module_name(rel)
        self.pkg = self.mod if rel.endswith("__init__.py") else self.mod.rpartition(".")[0]
        self.units, self.refs, self.classinfo = units, refs, classinfo
        self.aliases = {}
        self.toplevel = set()
        for s in tree.body:
            self.toplevel |= defined_names(s)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.asname:
                        self.aliases[a.asname] = a.name
                    else:
                        top = a.name.split(".")[0]
                        self.aliases[top] = top
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    pk = self.pkg.split(".")
                    pk = pk[: len(pk) - (node.level - 1)] if node.level > 1 else pk
                    base = ".".join(pk + ([base] if base else []))
                for a in node.names:
                    if a.name != "*":
                        self.aliases[a.asname or a.name] = f"{base}.{a.name}"
        self.tree = tree

    def resolve_name(self, name, class_stack):
        if class_stack:
            q, names = class_stack[-1]
            if name in names:
                return f"{self.mod}.{q}.{name}"
        if name in self.aliases:
            return self.aliases[name]
        if name in self.toplevel:
            return f"{self.mod}.{name}"
        return None

    def run(self):
        self._body(self.tree.body, [], False)

    def _unit(self, start, end, kind, owner, defnames, classkey=None):
        self.units.append((self.rel, start, end, kind, owner, tuple(sorted(defnames)), classkey))
        return len(self.units) - 1

    def _body(self, stmts, class_stack, in_func):
        for s in stmts:
            self._stmt(s, class_stack, in_func)

    def _stmt(self, s, class_stack, in_func):
        if in_func:
            # every expression of a statement inside a function runs at test time
            unit = None
            for field, value in ast.iter_fields(s):
                if field in ("body", "orelse", "finalbody", "handlers", "cases"):
                    continue
                for n in self._iter_nodes(value):
                    if unit is None:
                        unit = self._unit(s.lineno, getattr(s, "end_lineno", s.lineno), "f", None, ())
                    self._refs_in(n, class_stack, unit, True)
            self._children(s, class_stack, True)
            return
        # import-time statement (module or class body)
        if class_stack:
            kind, owner = "c", f"{self.mod}:{class_stack[-1][0]}"
        else:
            kind, owner = "m", self.mod
        names = defined_names(s)
        classkey = None
        if isinstance(s, ast.ClassDef):
            q = f"{class_stack[-1][0]}.{s.name}" if class_stack else s.name
            classkey = f"{self.mod}:{q}"
            self.classinfo[classkey] = (self.rel, first_line(s), s.end_lineno)
        unit = self._unit(first_line(s), s.end_lineno, kind, owner, names, classkey)
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for n in s.decorator_list + [s.args] + ([s.returns] if s.returns else []):
                self._refs_in(n, class_stack, unit, False)
            self._body(s.body, class_stack, True)
        elif isinstance(s, ast.ClassDef):
            for n in s.decorator_list + s.bases + s.keywords:
                self._refs_in(n, class_stack, unit, False)
            q = f"{class_stack[-1][0]}.{s.name}" if class_stack else s.name
            cnames = set()
            for c in s.body:
                cnames |= defined_names(c)
            self._body(s.body, class_stack + [(q, cnames)], False)
        else:
            for field, value in ast.iter_fields(s):
                if field in ("body", "orelse", "finalbody", "handlers", "cases"):
                    continue
                for n in self._iter_nodes(value):
                    self._refs_in(n, class_stack, unit, False)
            self._children(s, class_stack, False)

    def _children(self, s, class_stack, in_func):
        for field in ("body", "orelse", "finalbody"):
            self._body(getattr(s, field, []) or [], class_stack, in_func)
        for h in getattr(s, "handlers", []) or []:
            if h.type is not None:
                pass
            self._body(h.body, class_stack, in_func)
        for case in getattr(s, "cases", []) or []:
            self._body(case.body, class_stack, in_func)

    @staticmethod
    def _iter_nodes(value):
        if isinstance(value, ast.AST):
            yield value
        elif isinstance(value, list):
            for v in value:
                if isinstance(v, ast.AST):
                    yield v

    def _refs_in(self, node, class_stack, unit, runtime):
        stack = [(node, runtime, unit)]
        while stack:
            n, rt, u = stack.pop()
            if isinstance(n, ast.Lambda) and not rt:
                # lambda body runs at test time; give it its own runtime unit
                lu = self._unit(n.lineno, n.end_lineno, "f", None, ())
                stack.append((n.body, True, lu))
                continue
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(n, ast.Name):
                r = self.resolve_name(n.id, class_stack)
                if r and r.startswith(("sqlglot", "tests")):
                    self.refs.setdefault(n.id, []).append((u, r, 0))
                continue
            if isinstance(n, ast.Attribute):
                c = chain(n.value)
                if c:
                    base = self.resolve_name(c[0], class_stack)
                    if base and base.startswith(("sqlglot", "tests")):
                        full = ".".join([base] + c[1:])
                        self.refs.setdefault(n.attr, []).append((u, full, 1))
                stack.append((n.value, rt, u))
                continue
            for child in ast.iter_child_nodes(n):
                stack.append((child, rt, u))


# ----------------------------------------------------------------------------- index build


def build_index():
    m = json.load(gzip.open(MAPS, "rt"))
    sets = [int(x, 16) for x in m["sets"]]
    lines = {f: {int(k): sets[v] for k, v in d.items()} for f, d in m["lines"].items()}
    idx = {
        "tests": m["tests"],
        "lines": lines,
        "importlines": {f: set(v) for f, v in m["importlines"].items()},
        "opens": {k: sets[v] for k, v in m["opens"].items()},
        "code_opens": {k: sets[v] for k, v in m.get("code_opens", {}).items()},
        "classes": {k: sets[v] for k, v in m["classes"].items()},
        "testfiles": {k: sets[v] for k, v in m["testfiles"].items()},
        "missing": m.get("missing", []),
    }
    units, refs, classinfo = [], {}, {}
    modules = set()
    walkers = 0
    for root in ("sqlglot", "tests"):
        for p in sorted((SRC / root).rglob("*.py")):
            rel = p.relative_to(SRC).as_posix()
            if "/fixtures/" in rel:
                continue
            text = p.read_text(encoding="utf-8")
            modules.add(module_name(rel))
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            RefIndexer(rel, tree, units, refs, classinfo).run()
            if rel.startswith("tests/") and re.search(r"walk_packages|iter_modules|listdir|scandir|os\.walk|\.glob\(|rglob", text):
                walkers |= idx["testfiles"].get(rel, 0)
    idx.update(units=units, refs=refs, classinfo=classinfo, modules=modules, walkers=walkers)
    return idx


# ----------------------------------------------------------------------------- patch parsing


class PatchError(Exception):
    pass


def parse_patch(text):
    """-> list of dicts {old, new, hunks:[(ostart, olen, nstart, nlen, lines)], binary}"""
    files = []
    cur = None
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("diff --git "):
            cur = {"old": None, "new": None, "hunks": [], "binary": False, "header": line}
            files.append(cur)
            mm = re.match(r"diff --git a/(.*) b/(.*)$", line)
            if mm:
                cur["old"], cur["new"] = mm.group(1), mm.group(2)
            i += 1
            continue
        if line.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            if cur is None or cur["hunks"]:
                cur = {"old": None, "new": None, "hunks": [], "binary": False, "header": ""}
                files.append(cur)
            o = line[4:].split("\t")[0].strip()
            n = lines[i + 1][4:].split("\t")[0].strip()
            cur["old"] = None if o == "/dev/null" else (o[2:] if o.startswith("a/") else o)
            cur["new"] = None if n == "/dev/null" else (n[2:] if n.startswith("b/") else n)
            i += 2
            continue
        if cur is not None and line.startswith("new file mode"):
            cur["old"] = None
        elif cur is not None and line.startswith("deleted file mode"):
            cur["new"] = None
        elif cur is not None and line.startswith("rename from "):
            cur["old"] = line[len("rename from "):]
        elif cur is not None and line.startswith("rename to "):
            cur["new"] = line[len("rename to "):]
        elif cur is not None and (line.startswith("GIT binary patch") or line.startswith("Binary files")):
            cur["binary"] = True
        elif cur is not None and line.startswith("@@"):
            mm = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if not mm:
                raise PatchError("bad hunk header")
            os_, ol, ns, nl = int(mm.group(1)), int(mm.group(2) or 1), int(mm.group(3)), int(mm.group(4) or 1)
            body = []
            i += 1
            oc = nc = 0
            while i < len(lines) and (oc < ol or nc < nl):
                l = lines[i]
                if l.startswith("\\"):
                    i += 1
                    continue
                tag = l[:1] if l else " "
                if tag not in " +-":
                    break
                body.append((tag, l[1:]))
                if tag in " -":
                    oc += 1
                if tag in " +":
                    nc += 1
                i += 1
            # trailing "\ No newline" markers
            noeol_new = noeol_old = False
            while i < len(lines) and lines[i].startswith("\\"):
                if body and body[-1][0] == "-":
                    noeol_old = True
                else:
                    noeol_new = True
                i += 1
            cur["hunks"].append((os_, ol, ns, nl, body, noeol_new))
            continue
        i += 1
    return files


def apply_hunks(old_text, hunks):
    old = old_text.split("\n") if old_text else []
    trailing_nl = bool(old_text) and old_text.endswith("\n")
    if trailing_nl:
        old = old[:-1]
    out = []
    pos = 0
    noeol = False
    for os_, ol, ns, nl, body, noeol_new in hunks:
        want = [t for tag, t in body if tag in " -"]
        start = os_ - 1 if ol else os_
        found = None
        for delta in range(0, 200):
            for cand in ((start + delta,) if delta == 0 else (start + delta, start - delta)):
                if cand >= pos and old[cand:cand + len(want)] == want:
                    found = cand
                    break
            if found is not None:
                break
        if found is None:
            raise PatchError("hunk does not apply")
        out.extend(old[pos:found])
        out.extend(t for tag, t in body if tag in " +")
        pos = found + len(want)
        noeol = noeol_new
    out.extend(old[pos:])
    text = "\n".join(out)
    if out and (trailing_nl or not old) and not (noeol and pos >= len(old)):
        text += "\n"
    return text


# ----------------------------------------------------------------------------- selection


class Everything(Exception):
    pass


class Selector:
    def __init__(self, idx):
        self.idx = idx
        self.lines = idx["lines"]
        self.memo = {}
        n = len(idx["tests"])
        self.all = (1 << n) - 1

    # --- primitive lookups
    def cover(self, rel, start, end):
        d = self.lines.get(rel)
        if not d:
            return 0
        m = 0
        if end - start > len(d):
            for ln, s in d.items():
                if start <= ln <= end:
                    m |= s
        else:
            for ln in range(start, end + 1):
                s = d.get(ln)
                if s:
                    m |= s
        return m

    def class_users(self, ck):
        key = ("cu", ck)
        if key in self.memo:
            return self.memo[key]
        m = self.idx["classes"].get(ck, 0)
        info = self.idx["classinfo"].get(ck)
        if info:
            m |= self.cover(*info)
        self.memo[key] = m
        return m

    def unit_tests(self, u, depth):
        rel, start, end, kind, owner, defnames, classkey = self.idx["units"][u]
        m = self.cover(rel, start, end)
        if kind == "f":
            return m
        if classkey:
            m |= self.class_users(classkey)
        if depth >= MAX_DEPTH:
            return m | self.module_users(owner if kind == "m" else owner.split(":")[0])
        for dn in defnames:
            if kind == "m":
                m |= self.module_name(owner, dn, depth + 1)
            else:
                m |= self.class_attr(owner, dn, depth + 1)
        if kind == "c":
            m |= self.class_users(owner)
        return m

    def module_users(self, mod):
        """Every test that runs code from, or instantiates a class of, this module."""
        key = ("mu", mod)
        if key in self.memo:
            return self.memo[key]
        rel_py = mod.replace(".", "/") + ".py"
        rel_init = mod.replace(".", "/") + "/__init__.py"
        m = 0
        for rel in (rel_py, rel_init):
            d = self.lines.get(rel)
            if d:
                for s in d.values():
                    m |= s
        pref = mod + ":"
        for ck, s in self.idx["classes"].items():
            if ck.startswith(pref):
                m |= s
        self.memo[key] = m
        return m

    def module_name(self, mod, name, depth=0):
        """Tests that can observe a change of module-level `name` in `mod`."""
        key = ("mn", mod, name)
        if key in self.memo:
            return self.memo[key]
        self.memo[key] = 0  # cycle guard
        m = 0
        full = f"{mod}.{name}"
        for u, resolved, is_attr in self.idx["refs"].get(name, ()):
            if is_attr:
                ok = resolved == mod or resolved in self.idx["modules"]
            else:
                ok = resolved == full or (resolved.endswith("." + name) and resolved.startswith("sqlglot"))
            if ok:
                m |= self.unit_tests(u, depth)
        self.memo[key] = m
        return m

    def class_attr(self, ck, name, depth=0):
        """Tests that can observe a change of attribute `name` of class `ck`."""
        key = ("ca", ck, name)
        if key in self.memo:
            return self.memo[key]
        self.memo[key] = 0
        mod, qual = ck.split(":", 1)
        dotted = f"{mod}.{qual}"
        m = self.class_users(ck)
        for u, resolved, is_attr in self.idx["refs"].get(name, ()):
            if is_attr:
                ok = resolved == dotted or resolved.endswith("." + qual)
            else:
                ok = resolved == f"{dotted}.{name}"
            if ok:
                m |= self.unit_tests(u, depth)
        self.memo[key] = m
        return m

    # --- python diff
    def py_change(self, rel, old, new, why):
        mod = module_name(rel)
        try:
            told = ast.parse(old)
        except SyntaxError:
            raise Everything("old unparsable " + rel)
        try:
            tnew = ast.parse(new)
        except SyntaxError:
            why.append(f"{rel}: syntax error")
            if rel.startswith("tests/"):
                return self.idx["testfiles"].get(rel, 0) | self.importers_of_test(rel)
            raise Everything("syntax error")
        spans = {}
        for node in ast.walk(told):
            if isinstance(node, DEF_TYPES):
                spans[id(node)] = (first_line(node), node.end_lineno)
        strip_docstrings(told)
        strip_docstrings(tnew)
        if dump(told) == dump(tnew):
            m = self.idx["opens"].get(rel, 0)
            why.append(f"{rel}: comments/docstrings only")
            return m
        self.rel, self.mod, self.spans = rel, mod, spans
        return self.body(told.body, tnew.body, None, why)

    def importers_of_test(self, rel):
        # a test module imported by other test modules (e.g. tests/dialects/test_dialect.py)
        mod = module_name(rel)
        m = 0
        for ck, s in self.idx["classes"].items():
            if ck.startswith(mod + ":"):
                m |= s
        return m | self.module_users(mod)

    @staticmethod
    def key(s):
        if isinstance(s, DEF_TYPES):
            return ("def", s.name)
        if isinstance(s, ast.Assign):
            return ("as", tuple(dump(t) for t in s.targets))
        if isinstance(s, ast.AnnAssign):
            return ("as", (dump(s.target),))
        return None

    def body(self, olds, news, ck, why):
        od = [dump(s) for s in olds]
        nd = [dump(s) for s in news]
        m = 0
        sm = difflib.SequenceMatcher(None, od, nd, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            o_block, n_block = olds[i1:i2], news[j1:j2]
            used = set()
            for o in o_block:
                k = self.key(o)
                partner = None
                if k is not None:
                    for jx, n in enumerate(n_block):
                        if jx not in used and self.key(n) == k:
                            partner = jx
                            break
                if partner is not None:
                    used.add(partner)
                    m |= self.pair(o, n_block[partner], ck, why)
                else:
                    m |= self.stmt(o, None, ck, why)
            for jx, n in enumerate(n_block):
                if jx not in used:
                    m |= self.stmt(None, n, ck, why)
        return m

    def name_change(self, name, ck):
        if ck:
            return self.class_attr(ck, name)
        return self.module_name(self.mod, name)

    def pair(self, o, n, ck, why):
        if isinstance(o, ast.ClassDef) and isinstance(n, ast.ClassDef):
            q = f"{ck.split(':', 1)[1]}.{o.name}" if ck else o.name
            new_ck = f"{self.mod}:{q}"
            m = 0
            if [dump(x) for x in o.bases + o.keywords + o.decorator_list] != [dump(x) for x in n.bases + n.keywords + n.decorator_list]:
                why.append(f"class header {new_ck}")
                m |= self.class_users(new_ck) | self.name_change(o.name, ck)
                m |= self.subclass_registry(o, n)
            return m | self.body(o.body, n.body, new_ck, why)
        if isinstance(o, (ast.FunctionDef, ast.AsyncFunctionDef)) and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start, end = self.spans[id(o)]
            m = self.cover(self.rel, start, end)
            where = f"{ck or self.mod}.{o.name}"
            sig_o = [dump(x) for x in o.decorator_list + [o.args] + ([o.returns] if o.returns else [])]
            sig_n = [dump(x) for x in n.decorator_list + [n.args] + ([n.returns] if n.returns else [])]
            body_lines = set(range(o.body[0].lineno, end + 1)) if o.body else set()
            import_time = bool(body_lines & self.idx["importlines"].get(self.rel, set()))
            if sig_o != sig_n or type(o) is not type(n):
                why.append(f"signature {where}")
                m |= self.name_change(o.name, ck)
            if import_time:
                why.append(f"import-time function {where}")
                m |= self.name_change(o.name, ck)
                if ck:
                    m |= self.class_users(ck) | self.module_name(self.mod, ck.split(":", 1)[1].split(".")[0])
            else:
                why.append(f"function {where}")
            return m
        return self.stmt(o, n, ck, why)

    def subclass_registry(self, o, n):
        """Bases changed/added/removed: users of every base class (registries, MRO)."""
        m = 0
        for node in (o, n):
            if node is None:
                continue
            for b in node.bases:
                c = chain(b)
                if c:
                    simple = c[-1]
                    for ck, s in self.idx["classes"].items():
                        if ck.endswith(":" + simple) or ck.endswith("." + simple):
                            m |= s
        return m

    def stmt(self, o, n, ck, why):
        names = defined_names(o) | defined_names(n)
        m = 0
        if o is not None:
            m |= self.cover(self.rel, first_line(o), o.end_lineno)
            if isinstance(o, (ast.FunctionDef, ast.AsyncFunctionDef)):
                m |= self.cover(self.rel, *self.spans[id(o)])
        for node in (o, n):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self.check_import(node)
            if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
                raise Everything("star import changed")
        if isinstance(o or n, ast.ClassDef) and (o is None or n is None):
            m |= self.subclass_registry(o, n)
        if ck and isinstance(o or n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # added/removed method: dynamic dispatch (getattr(self, f"{key}_sql")) and overrides
            m |= self.class_users(ck)
        if not names:
            why.append(f"stmt without names in {ck or self.mod}")
            return m | (self.class_users(ck) if ck else self.module_users(self.mod))
        why.append(f"names {sorted(names)[:5]} in {ck or self.mod}")
        for name in names:
            m |= self.name_change(name, ck)
        return m

    def check_import(self, node):
        """A new import of a sqlglot module that does not exist breaks the module."""
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif node.level == 0 and node.module:
            mods = [node.module]
        for mname in mods:
            if mname.startswith("sqlglot") and mname not in self.idx["modules"] and mname not in self.new_modules:
                raise Everything("import of unknown module " + mname)

    # --- per commit
    def select(self, patch_text):
        why = []
        files = parse_patch(patch_text)
        if not files:
            return 0, ["empty patch"]
        self.new_modules = {module_name(f["new"]) for f in files if f["new"] and f["new"].endswith(".py")}
        mask = 0
        for f in files:
            mask |= self.file_change(f, why)
        return mask, why

    def read_src(self, rel):
        p = SRC / rel
        try:
            return p.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

    def file_change(self, f, why):
        old_rel, new_rel = f["old"], f["new"]
        rel = new_rel or old_rel
        if rel is None:
            raise Everything("no path")
        if f["binary"]:
            if rel.startswith(("sqlglot/", "tests/")):
                raise Everything("binary " + rel)
            return 0
        if old_rel and new_rel and old_rel != new_rel:
            # rename: removal of old + addition of new
            return self.file_change({**f, "new": None, "hunks": []}, why) | self.file_change({**f, "old": None}, why)
        old_text = self.read_src(old_rel) if old_rel else None
        if old_rel and old_text is None:
            raise Everything("unknown file " + old_rel)
        if new_rel:
            new_text = apply_hunks(old_text or "", f["hunks"])
        else:
            new_text = None
        is_py = rel.endswith(".py")
        top = rel.split("/")[0]
        reads = self.idx["opens"].get(rel, 0) | self.idx["code_opens"].get(rel, 0)
        if top in ("sqlglot", "tests") and is_py:
            if rel.startswith("tests/fixtures/"):
                return reads | self.new_or_deleted_fixture(rel, old_text, new_text)
            name = rel.rsplit("/", 1)[-1]
            if name in ("conftest.py",) or (rel.startswith("tests/") and name == "__init__.py"):
                raise Everything("test infra " + rel)
            if old_text is None:  # new module
                why.append(f"new module {rel}")
                return self.idx["walkers"]
            if new_text is None:  # deleted module
                raise Everything("deleted module " + rel)
            return self.py_change(rel, old_text, new_text, why) | self.idx["opens"].get(rel, 0)
        if rel.startswith("tests/"):
            # fixture / data file
            return reads | self.new_or_deleted_fixture(rel, old_text, new_text)
        if top == "sqlglot":
            if rel.endswith((".pyi", "py.typed", ".md", ".txt")):
                return reads
            raise Everything("non-py under sqlglot " + rel)
        # outside the package and tests: only what a test reads
        if rel in ("setup.cfg", "pyproject.toml", "setup.py", "pytest.ini", "tox.ini", "conftest.py"):
            raise Everything("config " + rel)
        if is_py and (old_text is not None and new_text is not None):
            try:
                if dump(strip_docstrings(ast.parse(old_text))) == dump(strip_docstrings(ast.parse(new_text))):
                    return self.idx["opens"].get(rel, 0)
            except SyntaxError:
                pass
        why.append(f"other file {rel}")
        return reads

    def new_or_deleted_fixture(self, rel, old_text, new_text):
        if old_text is not None and new_text is not None:
            return 0
        # a file appeared or vanished: tests that read its siblings or walk directories
        d = rel.rsplit("/", 1)[0] + "/"
        m = self.idx["walkers"]
        for k, s in self.idx["opens"].items():
            if k.startswith(d):
                m |= s
        return m


def load_index():
    try:
        with open(INDEX, "rb") as fh:
            return pickle.load(fh)
    except Exception as e:
        log("index load failed, rebuilding:", e)
        return build_index()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--prep":
        idx = build_index()
        with open(INDEX, "wb") as fh:
            pickle.dump(idx, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.chmod(INDEX, 0o644)
        log("index:", len(idx["units"]), "units", len(idx["refs"]), "names")
        return
    tests = json.loads((DATA / "tests.json").read_text())["tests"]
    durations = [0] * len(tests)
    number_of = {}
    for t in tests:
        durations[t["number"]] = round(t["seconds"] * 1000)
        number_of[t["id"]] = t["number"]
    everything = list(range(len(tests)))
    patches = sorted((DATA / "commits").glob("*.patch"))
    sel = None
    try:
        idx = load_index()
        sel = Selector(idx)
        map_tests = idx["tests"]
        # tests we have no map for are always run
        known = set(map_tests) - set(idx["missing"])
        always = [t["number"] for t in tests if t["id"] not in known]
        to_number = [number_of.get(t) for t in map_tests]
    except Exception as e:
        log("selector unavailable, running everything:", repr(e))
    out = []
    for patch in patches:
        chosen = everything
        if sel is not None:
            try:
                text = patch.read_text(encoding="utf-8", errors="replace")
                mask, why = sel.select(text)
                nums = set(always)
                i = 0
                while mask:
                    if mask & 1 and to_number[i] is not None:
                        nums.add(to_number[i])
                    mask >>= 1
                    i += 1
                chosen = sorted(nums)
                log(patch.stem, len(chosen), "tests;", "; ".join(why)[:300])
            except Everything as e:
                log(patch.stem, "everything:", e)
            except Exception as e:
                log(patch.stem, "error, everything:", repr(e))
        out.append(json.dumps({"commit": patch.stem, "workers": pack(chosen, durations)}, separators=(",", ":")))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
