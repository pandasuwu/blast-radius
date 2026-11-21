"""Blast Radius selector v3: coverage + region-level AST diff test selection.

Build time (`python selector.py --prep`, run by the Dockerfile, not charged):
  reads maps/maps.json.gz (per-test line coverage, opened files and instantiated classes,
  made by tools/ from the pinned commit) and the pristine sqlglot source, builds a reference
  index (which statements mention which names), an import graph, and pickles it all.

Run time: for each /data/commits/*.patch, apply it in memory to the pristine source. Each
changed region is located in the old and new syntax trees; the innermost statements that
differ decide the tests:
  * inside a function:      tests that executed the changed (or neighbouring) lines
  * class/module level:     tests that can read the changed names (coverage of the reading
                            lines, users of the class, recursively through import-time code)
  * could raise at import:  every test that imports the module (transitively)
  * comments / docstrings:  nothing, unless a test reads the file
  * fixtures / data files:  tests that opened the file
Then the tests are packed onto workers (LPT, best worker count). Anything unexpected falls
back to running everything for that commit. stdlib only; logs go to stderr.
"""

import ast
import difflib
import gzip
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

DATA = Path(os.environ.get("BR_DATA", "/data"))
APP = Path(__file__).resolve().parent
SRC = Path(os.environ.get("BR_SRC", APP / "src"))
MAPS = APP / "maps" / "maps.json.gz"
INDEX = Path(os.environ.get("BR_INDEX", APP / "index.pkl"))
MAX_WORKERS = 8
WORKER_COST_MS = 2000
MAX_DEPTH = 4
CHECK_FULL = os.environ.get("BR_FULLPARSE") == "1"
IMPORT_CHECK = r"""
import contextlib, importlib, importlib.util, importlib._bootstrap_external as be, io, json, os, sys
d = json.load(sys.stdin)
ov = {os.path.realpath(k): v.encode("utf-8") for k, v in d["overlay"].items()}
_orig = be.FileLoader.get_data
def _get_data(self, path):
    rp = os.path.realpath(path)
    if rp.endswith(".pyc"):
        # pristine bytecode is fine, except for a file the overlay replaces: compile that from source
        try:
            src_rp = os.path.realpath(importlib.util.source_from_cache(path))
        except Exception:
            raise OSError("unknown bytecode: compile from source")
        if src_rp in ov:
            raise OSError("bytecode of a patched file: compile from source")
    return ov[rp] if rp in ov else _orig(self, path)
be.FileLoader.get_data = _get_data
sys.path.insert(0, d["src"])
failed = []
for m in d["mods"]:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            importlib.import_module(m)
    except BaseException:
        failed.append(m)
print(json.dumps({"failed": failed}))
"""  # evaluation: force the full-file parse path


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


def pack_free(numbers, tiers, durations):
    """pack(), then fill idle worker time with free riders: tests from `tiers` (lists of test
    numbers, most plausible first, then the rest of the suite) placed only where they keep
    every worker at or under the busiest load. Cost and worker count are unchanged."""
    workers = pack(numbers, durations)
    if len(workers) < 2:
        return workers
    loads = [sum(durations[n] for n in w) for w in workers]
    top = max(loads)
    taken = set(numbers)
    for tier in tiers:
        for n in sorted((n for n in tier if n not in taken), key=lambda n: durations[n]):
            if n in taken:
                continue
            i = loads.index(min(loads))
            if loads[i] + durations[n] > top:
                break  # tier sorted by duration: nothing later in it fits either
            workers[i].append(n)
            loads[i] += durations[n]
            taken.add(n)
    return workers


# ----------------------------------------------------------------------------- AST helpers


def module_name(rel):
    p = rel[:-3].replace("/", ".")
    return p[: -len(".__init__")] if p.endswith(".__init__") else p


def dump(node):
    return ast.dump(node, annotate_fields=False)


def first_line(node):
    decos = getattr(node, "decorator_list", None)
    return min([node.lineno] + [d.lineno for d in decos]) if decos else node.lineno


DEF_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)
BLOCK_FIELDS = ("body", "orelse", "handlers", "finalbody", "cases")


def is_docstring(s):
    return isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str)


def strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            b = node.body
            if b and is_docstring(b[0]):
                node.body = b[1:] or [ast.Pass()]
    return tree


def is_type_checking(s):
    if not isinstance(s, ast.If):
        return False
    t = s.test
    return (isinstance(t, ast.Name) and t.id == "TYPE_CHECKING") or (
        isinstance(t, ast.Attribute) and t.attr == "TYPE_CHECKING")


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


IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ----------------------------------------------------------------------------- reference index


class RefIndexer:
    """For one source file: every Name/Attribute/str-constant reference, with the statement
    unit it is in. Resolved refs point at sqlglot/tests objects; unresolved attribute refs are
    kept only for names some class body assigns (class attributes), as (name, None, 1)."""

    def __init__(self, rel, tree, units, refs, classinfo, attrnames, imports):
        self.rel = rel
        self.mod = module_name(rel)
        self.pkg = self.mod if rel.endswith("__init__.py") else self.mod.rpartition(".")[0]
        self.units, self.refs, self.classinfo, self.attrnames = units, refs, classinfo, attrnames
        self.aliases = {}
        self.toplevel = set()
        for s in tree.body:
            self.toplevel |= defined_names(s)
        deps = imports.setdefault(self.mod, set())
        tc_nodes = set()
        for s in tree.body:
            if is_type_checking(s):
                tc_nodes |= {id(n) for n in ast.walk(s)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.asname:
                        self.aliases[a.asname] = a.name
                    else:
                        top = a.name.split(".")[0]
                        self.aliases[top] = top
                    if id(node) not in tc_nodes:
                        deps.add(a.name)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    pk = self.pkg.split(".")
                    pk = pk[: len(pk) - (node.level - 1)] if node.level > 1 else pk
                    base = ".".join(pk + ([base] if base else []))
                for a in node.names:
                    if a.name != "*":
                        self.aliases[a.asname or a.name] = f"{base}.{a.name}"
                    if id(node) not in tc_nodes:
                        deps.add(base)
                        if a.name != "*":
                            deps.add(f"{base}.{a.name}")
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
            unit = None
            for field, value in ast.iter_fields(s):
                if field in BLOCK_FIELDS:
                    continue
                for n in self._iter_nodes(value):
                    if unit is None:
                        unit = self._unit(s.lineno, getattr(s, "end_lineno", s.lineno), "f", None, ())
                    self._refs_in(n, class_stack, unit, True)
            self._children(s, class_stack, True)
            return
        if class_stack:
            kind, owner = "c", f"{self.mod}:{class_stack[-1][0]}"
        else:
            kind, owner = "m", self.mod
        names = defined_names(s)
        classkey = None
        if isinstance(s, ast.ClassDef):
            q = f"{class_stack[-1][0]}.{s.name}" if class_stack else s.name
            classkey = f"{self.mod}:{q}"
            bases = []
            for b in s.bases:
                c = chain(b)
                if c:
                    r = self.resolve_name(c[0], class_stack)
                    bases.append(".".join([r] + c[1:]) if r else ".".join(c))
            meta = None
            for kw in s.keywords:
                if kw.arg == "metaclass":
                    c = chain(kw.value)
                    if c:
                        r = self.resolve_name(c[0], class_stack)
                        meta = ".".join([r] + c[1:]) if r else ".".join(c)
            self.classinfo[classkey] = (self.rel, first_line(s), s.end_lineno, tuple(bases), meta)
        unit = self._unit(first_line(s), s.end_lineno, kind, owner, names, classkey)
        if isinstance(s, FUNC_TYPES):
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
                if field in BLOCK_FIELDS:
                    continue
                for n in self._iter_nodes(value):
                    self._refs_in(n, class_stack, unit, False)
            self._children(s, class_stack, False)

    def _children(self, s, class_stack, in_func):
        for field in ("body", "orelse", "finalbody"):
            self._body(getattr(s, field, []) or [], class_stack, in_func)
        for h in getattr(s, "handlers", []) or []:
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

    def _add(self, name, u, resolved, kind, pos=(0, 0)):
        self.refs.setdefault(name, []).append((u, resolved, kind) + pos)

    def _refs_in(self, node, class_stack, unit, runtime):
        stack = [(node, runtime, unit)]
        while stack:
            n, rt, u = stack.pop()
            if isinstance(n, ast.Lambda) and not rt:
                lu = self._unit(n.lineno, n.end_lineno, "f", None, ())
                stack.append((n.body, True, lu))
                continue
            if isinstance(n, FUNC_TYPES):
                continue
            if isinstance(n, ast.Name):
                if not isinstance(n.ctx, ast.Load):
                    continue
                r = self.resolve_name(n.id, class_stack)
                if r and r.startswith(("sqlglot", "tests")):
                    self._add(n.id, u, r, 0)
                continue
            if isinstance(n, ast.Constant):
                v = n.value
                if isinstance(v, str) and v in self.attrnames:
                    self._add(v, u, None, 2)
                continue
            if isinstance(n, ast.Attribute):
                if not isinstance(n.ctx, ast.Load):
                    stack.append((n.value, rt, u))
                    continue
                c = chain(n.value)
                full = None
                if c:
                    base = self.resolve_name(c[0], class_stack)
                    if base and base.startswith(("sqlglot", "tests")):
                        full = ".".join([base] + c[1:])
                if full is not None:
                    self._add(n.attr, u, full, 1, (n.lineno, n.col_offset))
                elif n.attr in self.attrnames:
                    self._add(n.attr, u, None, 1, (n.lineno, n.col_offset))
                stack.append((n.value, rt, u))
                continue
            for child in ast.iter_child_nodes(n):
                stack.append((child, rt, u))


def class_attr_names(tree):
    """Names assigned directly in class bodies (class attributes, not methods)."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for s in node.body:
                if isinstance(s, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    out |= defined_names(s)
    return out


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
    prof, untrusted = {}, 0
    pf = APP / "maps" / "prof.json.gz"
    if pf.exists():
        p = json.load(gzip.open(pf, "rt"))
        assert p["tests"] == m["tests"]
        psets = [int(x, 16) for x in p["sets"]]
        for rel, nodes in p["nodes"].items():
            prof[rel] = {tuple(map(int, k.split(","))): (kind, op, imp, [(sig, psets[i]) for sig, i in sigs])
                         for k, (kind, op, imp, sigs) in nodes.items()}
        untrusted = int(p["untrusted"], 16)
        attr_instr = set(p.get("attrnames", ()))
        attrprof, attrpos = {}, set()
        for rel, nodes in prof.items():
            for (l, c, el, ec), (kind, op, imp, sigs) in nodes.items():
                if kind == "attr":
                    attrprof.setdefault(op, []).append((rel, l, imp, sigs))
                    attrpos.add((rel, l, c))
        idx["attr_instr"], idx["attrprof"], idx["attrpos"] = attr_instr, attrprof, attrpos
        idx["identity_decorators"] = {tuple(x) for x in p.get("identity_decorators", ())}
    idx["prof"], idx["untrusted"] = prof, untrusted
    trees = {}
    attrnames = set()
    for root in ("sqlglot", "tests"):
        for p in sorted((SRC / root).rglob("*.py")):
            rel = p.relative_to(SRC).as_posix()
            if "/fixtures/" in rel:
                continue
            text = p.read_text(encoding="utf-8")
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            trees[rel] = (tree, text)
            attrnames |= class_attr_names(tree)
    units, refs, classinfo, imports = [], {}, {}, {}
    walkers = 0
    sens_units = []
    modules = set()
    toplevel = {}
    for rel, (tree, text) in trees.items():
        mod = module_name(rel)
        modules.add(mod)
        toplevel[mod] = {n for s in tree.body for n in defined_names(s)}
        RefIndexer(rel, tree, units, refs, classinfo, attrnames, imports).run()
        if rel.startswith("tests/"):
            if re.search(r"walk_packages|iter_modules|listdir|scandir|os\.walk|\.glob\(|rglob", text):
                walkers |= idx["testfiles"].get(rel, 0)
            # import-sensitive test functions: local imports, subprocesses, importlib, sys.modules
            for node in ast.walk(tree):
                if isinstance(node, FUNC_TYPES):
                    for sub in ast.walk(node):
                        hit = isinstance(sub, (ast.Import, ast.ImportFrom)) and any(
                            (a.name if isinstance(sub, ast.Import) else (sub.module or "")).startswith("sqlglot")
                            for a in sub.names)
                        if not hit and isinstance(sub, ast.Attribute) and sub.attr in ("import_module", "modules", "run", "Popen", "reload"):
                            c = chain(sub)
                            hit = bool(c) and c[0] in ("importlib", "sys", "subprocess")
                        if hit:
                            sens_units.append((rel, node.lineno, node.end_lineno))
                            break
    # module re-exports (`from sqlglot import expressions as exp` -> sqlglot.exp is that module)
    mod_alias = {}
    for rel, (tree, text) in trees.items():
        mod = module_name(rel)
        pkg = mod if rel.endswith("__init__.py") else mod.rpartition(".")[0]
        for st in tree.body:
            if isinstance(st, ast.ImportFrom):
                base = st.module or ""
                if st.level:
                    pk = pkg.split(".")
                    pk = pk[: len(pk) - (st.level - 1)] if st.level > 1 else pk
                    base = ".".join(pk + ([base] if base else []))
                for a in st.names:
                    full = f"{base}.{a.name}"
                    if full in modules and a.asname:
                        mod_alias[f"{mod}.{a.asname}"] = full
            elif isinstance(st, ast.Import):
                for a in st.names:
                    if a.asname and a.name in modules:
                        mod_alias[f"{mod}.{a.asname}"] = a.name

    def norm(r):
        for _ in range(4):
            head, rest = r, ""
            hit = False
            while head:
                if head in mod_alias:
                    r = mod_alias[head] + rest
                    hit = True
                    break
                head, _, tail = head.rpartition(".")
                rest = "." + tail + rest
            if not hit:
                break
        return r

    extra = []
    for name, lst in refs.items():
        for i, (u, resolved, kind, *pos) in enumerate(lst):
            if resolved:
                nr = norm(resolved)
                if nr != resolved:
                    lst[i] = (u, nr, kind, *pos)
                    resolved = nr
            if kind == 0 and resolved and resolved.rsplit(".", 1)[-1] != name:
                extra.append((resolved.rsplit(".", 1)[-1], (u, resolved, kind, *pos)))
    for name, entry in extra:
        refs.setdefault(name, []).append(entry)
    # map refs to package re-exports ("sqlglot.dialects.Athena") onto the defining module
    pkg_reexport = {}
    for mod, names in toplevel.items():
        parts = mod.split(".")
        for i in range(1, len(parts)):
            pkg = ".".join(parts[:i])
            for nm in names:
                pkg_reexport.setdefault(f"{pkg}.{nm}", set()).add(mod)
    mod_refunits = {}
    for name, lst in refs.items():
        for u, resolved, kind, *_pos in lst:
            if resolved is None or units[u][3] != "f":
                continue
            target = resolved
            for _ in range(3):
                if target in modules:
                    mod_refunits.setdefault(target, set()).add(u)
                    break
                for dm in pkg_reexport.get(target, ()):
                    mod_refunits.setdefault(dm, set()).add(u)
                target = target.rpartition(".")[0]
                if not target:
                    break
    # normalise import edges to known modules
    deps = {}
    for mod, ds in imports.items():
        out = set()
        for d in ds:
            while d and d not in modules:
                d = d.rpartition(".")[0]
            if d and d != mod:
                out.add(d)
            # importing a.b.c also runs a and a.b
            parts = d.split(".") if d else []
            for i in range(1, len(parts)):
                pm = ".".join(parts[:i])
                if pm in modules and pm != mod:
                    out.add(pm)
        deps[mod] = out
    core = set()
    stack = ["sqlglot"]
    while stack:
        mm = stack.pop()
        if mm in core:
            continue
        core.add(mm)
        stack.extend(deps.get(mm, ()))
    rdeps = {}
    for mod, ds in deps.items():
        for d in ds:
            rdeps.setdefault(d, set()).add(mod)

    def cover(rel, start, end):
        d = lines.get(rel) or {}
        mm = 0
        for ln in range(start, end + 1):
            s = d.get(ln)
            if s:
                mm |= s
        return mm

    importsens = walkers
    for rel, a, b in sens_units:
        importsens |= cover(rel, a, b)
    mod_refmask = {}
    for mod, us in mod_refunits.items():
        mm = 0
        for u in us:
            mm |= cover(*units[u][:3])
        if mm:
            mod_refmask[mod] = mm
    outline = {}
    for rel, (tree, text) in trees.items():
        src_lines = text.split("\n")
        mod = module_name(rel)
        ent = []

        def walk(stmts, parents, ck):
            for st in stmts:
                if isinstance(st, FUNC_TYPES) and st.body:
                    ent.append((st.body[0].lineno, st.end_lineno, first_line(st), st.lineno,
                                ("c", ck) if ck else ("m", None), tuple(parents)))
                elif isinstance(st, ast.ClassDef):
                    line = src_lines[st.lineno - 1]
                    ind = line[: len(line) - len(line.lstrip())]
                    q = f"{ck.split(':', 1)[1]}.{st.name}" if ck else st.name
                    walk(st.body, parents + [ind], f"{mod}:{q}")

        walk(tree.body, [], None)
        outline[rel] = ent
    idx["outline"] = outline
    setup_attr = {}
    LOADERS = ("load_sql_fixture_pairs", "load_sql_fixtures")
    for rel, (tree, text) in trees.items():
        if not rel.startswith("tests/"):
            continue
        for cls_node in ast.walk(tree):
            if not isinstance(cls_node, ast.ClassDef):
                continue
            for fn in cls_node.body:
                if not (isinstance(fn, FUNC_TYPES) and fn.name == "setUpClass"):
                    continue
                for st in ast.walk(fn):
                    if not (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Attribute)
                            and isinstance(st.targets[0].value, ast.Name) and st.targets[0].value.id == "cls"):
                        continue
                    for call in ast.walk(st.value):
                        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in LOADERS
                                and call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str)):
                            fx = "tests/fixtures/" + call.args[0].value
                            setup_attr.setdefault(fx, []).append(
                                (f"{rel}::{cls_node.name}", st.targets[0].attr, call.func.id, call.args[0].value))
    idx["setup_attr"] = setup_attr
    owner_tests = {}
    for i, tid in enumerate(m["tests"]):
        owner_tests[tid.rsplit("::", 1)[0]] = owner_tests.get(tid.rsplit("::", 1)[0], 0) | (1 << i)
    idx["owner_tests"] = owner_tests
    if "opens_c" in m:
        idx["field_reads"] = {o: {a: {f: sets[v] for f, v in fd.items()} for a, fd in d.items()}
                              for o, d in m.get("field_reads", {}).items()}
        idx["opens_c"] = {k: sets[v] for k, v in m["opens_c"].items()}
        idx["opens_s"] = m["opens_s"]
        idx["attr_reads"] = {o: {a: sets[v] for a, v in d.items()} for o, d in m["attr_reads"].items()}
    idx.update(units=units, refs=refs, classinfo=classinfo, modules=modules, walkers=walkers,
               core=core, rdeps=rdeps, importsens=importsens, mod_refmask=mod_refmask,
               attrnames=attrnames, toplevel=toplevel)
    try:
        idx["pool_groups"] = shared_pool_groups(m["tests"])
    except Exception as e:
        log("shared pool scan failed:", repr(e))
        idx["pool_groups"] = []
    idx["tpcds_phase"] = None
    try:
        ph = json.load(open(APP / "maps" / "tpcds_phase.json"))
        ulines = {rel: set(v) for rel, v in ph["lines"].items()}
        names = set()
        for rel, ls in ulines.items():
            src_lines = (SRC / rel).read_text(encoding="utf-8").split("\n")
            for ln in ls:
                names.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", src_lines[ln - 1]))
        idx["tpcds_phase"] = {"lines": ulines, "classes": set(ph["classes"]), "names": names}
        xc = json.load(open(APP / "maps" / "tpcds_xcov.json"))["lines"]
        idx["tpcds_xcov"] = {rel: {int(l): int(m, 16) for l, m in d.items()} for rel, d in xc.items()}
        v, ref = tpcds_run({}, timeout=480)
        idx["tpcds_ref"] = ref if v.get("ok") and ref else None
        idx["tpcds_base"] = bool(idx["tpcds_ref"])
    except Exception as e:
        log("tpcds proxy disabled:", repr(e))
        idx["tpcds_base"] = None
    log("tpcds proxy baseline:", idx["tpcds_base"])
    return idx


def shared_pool_groups(test_ids):
    """Test classes whose setUpClass makes one process pool shared by its tests. If a patch makes
    a pool worker die, every later pool-using test of that class fails (BrokenProcessPool).
    Returns, per class, the bit indices of its pool-using tests in suite (name) order."""
    pos = {t: i for i, t in enumerate(test_ids)}
    groups = []
    for path in sorted((SRC / "tests").rglob("test_*.py")):
        text = path.read_text(encoding="utf-8")
        if "Pool" not in text:
            continue
        rel = path.relative_to(SRC).as_posix()
        for cls in ast.walk(ast.parse(text)):
            if not isinstance(cls, ast.ClassDef):
                continue
            meths = {f.name: f for f in cls.body if isinstance(f, FUNC_TYPES)}
            setup = meths.get("setUpClass")
            if setup is None:
                continue
            attrs = {t.attr for n in ast.walk(setup) if isinstance(n, ast.Assign)
                     and isinstance(n.value, ast.Call) and "Pool" in ast.unparse(n.value.func)
                     for t in n.targets if isinstance(t, ast.Attribute)}
            if not attrs:
                continue

            def uses(f):
                return {n.attr for n in ast.walk(f) if isinstance(n, ast.Attribute)
                        and isinstance(n.value, ast.Name) and n.value.id in ("self", "cls")}
            users = {name for name, f in meths.items() if uses(f) & attrs and name != "setUpClass"}
            grew = True
            while grew:
                more = {name for name, f in meths.items() if name not in users and uses(f) & users}
                users |= more
                grew = bool(more)
            order = [pos[f"{rel}::{cls.name}::{name}"] for name in sorted(users)
                     if name.startswith("test") and f"{rel}::{cls.name}::{name}" in pos]
            if order:
                groups.append(order)
    return groups


# ----------------------------------------------------------------------------- patch parsing


class PatchError(Exception):
    pass


def parse_patch(text):
    """-> list of dicts {old, new, hunks:[(ostart, olen, nstart, nlen, body, noeol)], binary}"""
    files = []
    cur = None
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("diff --git "):
            cur = {"old": None, "new": None, "hunks": [], "binary": False}
            files.append(cur)
            mm = re.match(r"diff --git a/(.*) b/(.*)$", line)
            if mm:
                cur["old"], cur["new"] = mm.group(1), mm.group(2)
            i += 1
            continue
        if line.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            if cur is None or cur["hunks"]:
                cur = {"old": None, "new": None, "hunks": [], "binary": False}
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
            noeol_new = False
            while i < len(lines) and lines[i].startswith("\\"):
                if not (body and body[-1][0] == "-"):
                    noeol_new = True
                i += 1
            cur["hunks"].append((os_, ol, ns, nl, body, noeol_new))
            continue
        i += 1
    return files


def apply_hunks(old_text, hunks):
    """Apply hunks; returns (new_text, regions). Regions are (o_start, o_end, n_start, n_end)
    in 1-based old/new line numbers; an empty side has end == start - 1 (insertion point)."""
    old = old_text.split("\n") if old_text else []
    trailing_nl = bool(old_text) and old_text.endswith("\n")
    if trailing_nl:
        old = old[:-1]
    out = []
    regions = []
    pos = 0
    noeol = False
    for os_, ol, ns, nl, body, noeol_new in hunks:
        want = [t for tag, t in body if tag in " -"]
        start = os_ - 1 if ol else os_
        found = None
        for delta in range(0, 400):
            for cand in ((start + delta,) if delta == 0 else (start + delta, start - delta)):
                if cand >= pos and old[cand:cand + len(want)] == want:
                    found = cand
                    break
            if found is not None:
                break
        if found is None:
            raise PatchError("hunk does not apply")
        out.extend(old[pos:found])
        oline = found + 1
        nline = len(out) + 1
        k = 0
        while k < len(body):
            tag, t = body[k]
            if tag == " ":
                out.append(t)
                oline += 1
                nline += 1
                k += 1
                continue
            o0, n0 = oline, nline
            while k < len(body) and body[k][0] != " ":
                tag, t = body[k]
                if tag == "-":
                    oline += 1
                else:
                    out.append(t)
                    nline += 1
                k += 1
            regions.append((o0, oline - 1, n0, nline - 1))
        pos = found + len(want)
        noeol = noeol_new
    out.extend(old[pos:])
    text = "\n".join(out)
    if out and (trailing_nl or not old) and not (noeol and pos >= len(old)):
        text += "\n"
    return text, regions


# ----------------------------------------------------------------------------- selection


class Everything(Exception):
    pass


def overlaps(s, a, b):
    f, e = first_line(s), s.end_lineno
    if b < a:  # insertion before line a
        return f < a <= e
    return f <= b and a <= e


def block_span(block):
    return first_line(block[0]), max(s.end_lineno for s in block)


def children(node):
    for field in BLOCK_FIELDS:
        v = getattr(node, field, None)
        if v and isinstance(v, list) and hasattr(v[0], "lineno"):
            yield field, v


SAFE_LEAVES = (ast.Constant, ast.Lambda, ast.Pass)


class Selector:
    def __init__(self, idx):
        self.idx = idx
        self.lines = idx["lines"]
        self.memo = {}
        self.all = (1 << len(idx["tests"])) - 1

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

    def is_import_time(self, rel, start, end):
        il = self.idx["importlines"].get(rel)
        if not il:
            return False
        if end - start > len(il):
            return any(start <= x <= end for x in il)
        return any(x in il for x in range(start, end + 1))

    def class_users(self, ck):
        key = ("cu", ck)
        if key in self.memo:
            return self.memo[key]
        m = self.idx["classes"].get(ck, 0)
        info = self.idx["classinfo"].get(ck)
        if info:
            m |= self.cover(info[0], info[1], info[2])
        self.memo[key] = m
        return m

    def module_users(self, mod):
        """Tests that run code from, instantiate a class of, or reference this module."""
        key = ("mu", mod)
        if key in self.memo:
            return self.memo[key]
        m = 0
        for rel in (mod.replace(".", "/") + ".py", mod.replace(".", "/") + "/__init__.py"):
            d = self.lines.get(rel)
            if d:
                for s in d.values():
                    m |= s
            m |= self.idx["testfiles"].get(rel, 0)
        pref = mod + ":"
        for ck, s in self.idx["classes"].items():
            if ck.startswith(pref):
                m |= s
        m |= self.idx["mod_refmask"].get(mod, 0)
        self.memo[key] = m
        return m

    def maybe_import_break(self, mod):
        """The module *might* now raise at import: decided once per commit by really importing
        the patched modules (see check_imports); nothing is added here."""
        self.pending_imports.add(mod)
        return 0

    def check_imports(self):
        """Import every pending module in a fresh interpreter with the patched sources swapped in;
        returns the modules that fail (all of them if the check itself cannot run)."""
        mods = sorted(self.pending_imports)
        if not mods:
            return []
        import subprocess
        payload = json.dumps({"src": str(SRC), "mods": mods,
                              "overlay": {str(SRC / rel): text for rel, text in self.patched.items()}})
        try:
            p = subprocess.run([sys.executable, "-I", "-B", "-c", IMPORT_CHECK], input=payload,
                               capture_output=True, text=True, timeout=90,
                               env={"PYTHONDONTWRITEBYTECODE": "1", "HOME": "/tmp", "PATH": os.environ.get("PATH", "")})
            out = [l for l in p.stdout.splitlines() if l.startswith("{")]
            failed = json.loads(out[-1])["failed"]
        except Exception as e:
            log("import check failed:", repr(e))
            return mods
        return failed

    def import_break(self, mod):
        """The module may now fail to import: every test that (transitively) imports it."""
        key = ("ib", mod)
        if key in self.memo:
            return self.memo[key]
        if mod in self.idx["core"] or not mod.startswith(("sqlglot", "tests")):
            m = self.all
        else:
            seen, stack = set(), [mod]
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x)
                stack.extend(self.idx["rdeps"].get(x, ()))
            m = self.idx["importsens"] | self.idx["walkers"]
            for x in seen:
                if x in self.idx["core"]:
                    m = self.all
                    break
                m |= self.module_users(x)
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

    def module_name(self, mod, name, depth=0):
        """Tests that can observe a change of module-level `name` in `mod`."""
        key = ("mn", mod, name)
        if key in self.memo:
            return self.memo[key]
        self.memo[key] = 0  # cycle guard
        m = 0
        full = f"{mod}.{name}"
        for u, resolved, kind, *_pos in self.idx["refs"].get(name, ()):
            if resolved is None:
                continue
            if kind == 1:
                ok = resolved == mod or resolved in self.idx["modules"]
            else:
                ok = resolved == full or (resolved.endswith("." + name) and resolved.startswith("sqlglot"))
            if ok:
                m |= self.unit_tests(u, depth)
        self.memo[key] = m
        return m

    def is_subclass(self, dotted, ck, seen=None):
        """Static check: does the class at `dotted` have class `ck` among its ancestors?"""
        top = seen is None
        if top:
            hit = self.memo.get(("sub", dotted, ck))
            if hit is not None:
                return hit
        mod, qual = ck.split(":", 1)
        if dotted == f"{mod}.{qual}" or dotted.endswith("." + qual):
            res = True
        else:
            res = False
            seen = seen if seen is not None else set()
            by_last = self.memo.get("by_last")
            if by_last is None:
                by_last = {}
                for k, info in self.idx["classinfo"].items():
                    m2, q2 = k.split(":", 1)
                    by_last.setdefault(q2.rsplit(".", 1)[-1], []).append((k, f"{m2}.{q2}", q2, info[3]))
                self.memo["by_last"] = by_last
            for k, full, q2, bases in by_last.get(dotted.rsplit(".", 1)[-1], ()):
                if (full == dotted or dotted.endswith("." + q2)) and k not in seen:
                    seen.add(k)
                    if any(self.is_subclass(b, ck, seen) for b in bases):
                        res = True
                        break
        if top:
            self.memo[("sub", dotted, ck)] = res
        return res

    def metaclass_of(self, ck, seen=None):
        """Dotted name of ck's metaclass (own or inherited through static bases), or None."""
        info = self.idx["classinfo"].get(ck)
        if info is None:
            return None
        if len(info) > 4 and info[4]:
            return info[4]
        seen = seen or set()
        seen.add(ck)
        for b in info[3]:
            last = b.rsplit(".", 1)[-1]
            for k in self.idx["classinfo"]:
                if k in seen:
                    continue
                q = k.split(":", 1)[1]
                if q.rsplit(".", 1)[-1] == last and (b.endswith("." + q) or b == q):
                    r = self.metaclass_of(k, seen)
                    if r:
                        return r
        return None

    def stmt_unit(self, loc):
        """Innermost module/class-level statement unit containing 'rel:line'."""
        memo = self.memo.get("stmt_units")
        if memo is None:
            memo = {}
            for i, (rel, a, b, kind, *_rest) in enumerate(self.idx["units"]):
                if kind in ("m", "c"):
                    memo.setdefault(rel, []).append((a, b, i))
            self.memo["stmt_units"] = memo
        rel, _, line = loc.rpartition(":")
        try:
            line = int(line)
        except ValueError:
            return None
        best = None
        for a, b, i in memo.get(rel, ()):
            if a <= line <= b and (best is None or a >= best[0]):
                best = (a, b, i)
        return best[2] if best else None

    def enclosing_class(self, rel, a, b):
        best = None
        for k, info in self.idx["classinfo"].items():
            if info[0] == rel and info[1] <= a and b <= info[2]:
                if best is None or info[1] >= self.idx["classinfo"][best][1]:
                    best = k
        return best

    @staticmethod
    def same_class(dotted, ck):
        mod, qual = ck.split(":", 1)
        return dotted == f"{mod}.{qual}" or dotted.endswith("." + qual)

    def class_attr(self, ck, name, depth=0, added=False):
        """Tests that can observe a change of attribute `name` of class `ck`."""
        key = ("ca", ck, name, added)
        if key in self.memo:
            return self.memo[key]
        self.memo[key] = 0
        users = self.class_users(ck)
        mod, qual = ck.split(":", 1)
        dotted = f"{mod}.{qual}"
        runtime = 0
        m = 0
        unsafe = depth >= MAX_DEPTH
        needs_outer = unsafe
        own_meta = self.metaclass_of(ck)
        # instrumented reads (attribute profile) say exactly which tests read ck's own value;
        # a newly added attribute shadows a parent's value, which the profile cannot attribute
        use_prof = not added and name in self.idx.get("attr_instr", ())
        attrpos = self.idx.get("attrpos", ())
        direct = 0
        if use_prof:
            for prel, pline, imp, sigs in self.idx["attrprof"].get(name, ()):
                for k in imp:
                    key_, _, loc = k.partition("@")
                    if key_ != "?" and key_ != ck:
                        continue
                    # read while import-time statement `loc` ran: only what that statement
                    # defines can carry the change
                    u = self.stmt_unit(loc) if loc and depth < MAX_DEPTH else None
                    if u is not None:
                        urel, ua, ub, ukind, uowner, udef, ucls = self.idx["units"][u]
                        if ucls and (ucls == ck or self.is_subclass(ucls.replace(":", "."), ck)):
                            # read while creating ck (or a subclass): what the metaclass derives
                            # lives on that class, so its users can observe it
                            m |= self.class_users(ucls) | users
                            continue
                        m |= self.unit_tests(u, depth + 1)
                        # another statement read it at import: it may now raise there too
                        m |= self.maybe_import_break(module_name(urel))
                        continue
                    unsafe = True
                    reader = self.enclosing_class(prel, pline, pline)
                    if reader is None or own_meta is None or not self.same_class(own_meta, reader):
                        needs_outer = True
                for k, mask in sigs:
                    if k == "?" or k == ck:
                        direct |= mask
        for u, resolved, kind, *_pos in self.idx["refs"].get(name, ()):
            urel, a, b, ukind = self.idx["units"][u][:4]
            if use_prof and kind == 1 and _pos and (urel, _pos[0], _pos[1]) in attrpos:
                continue  # covered by the attribute profile
            if ukind == "f":
                if self.is_import_time(urel, a, b):
                    if resolved is None or self.is_subclass(resolved.rpartition(".")[0], ck) or kind == 0:
                        unsafe = True
                        reader = self.enclosing_class(urel, a, b)
                        if reader is None or own_meta is None or not self.same_class(own_meta, reader):
                            needs_outer = True
                    else:
                        m |= self.cover(urel, a, b)
                else:
                    runtime |= self.cover(urel, a, b)
                continue
            # import-time reference in a class body / module statement
            if resolved is None:
                unsafe = needs_outer = True
            elif kind == 0:
                if resolved == f"{dotted}.{name}":
                    m |= self.unit_tests(u, depth)
            elif kind == 1 and self.is_subclass(resolved, ck):
                m |= self.unit_tests(u, depth)
        if name.endswith("_sql") and not name.startswith("_"):
            # Generator dispatch: dir(cls) -> {expr key: method}; used for that expression type
            key_ = name[:-4]
            exp_users = 0
            for c, s in self.idx["classes"].items():
                if c.startswith("sqlglot.expressions") and c.rsplit(":", 1)[-1].rsplit(".", 1)[-1].lower() == key_.replace("_", ""):
                    exp_users |= s
                elif c.startswith("sqlglot.expressions") and c.rsplit(":", 1)[-1].lower() == key_:
                    exp_users |= s
            m |= exp_users & (users | runtime) if exp_users else users
            m |= runtime & users
        elif unsafe:
            # read at import time. If the reader is ck's own metaclass, what it derives lives on
            # ck (users cover it); otherwise (e.g. the dialect metaclass deriving tables from a
            # nested tokenizer class) the outer classes' users can observe it too.
            m |= users
            outer = qual if needs_outer else ""
            while "." in outer:
                outer = outer.rsplit(".", 1)[0]
                m |= self.class_users(f"{mod}:{outer}")
        else:
            m |= users & runtime
        m |= direct
        self.memo[key] = m
        return m

    # --- python diff
    def py_change(self, rel, old, new, regions, why):
        mod = module_name(rel)
        self.rel, self.mod = rel, mod
        self.old_text = old
        if not CHECK_FULL:
            r = self.snippet_change(rel, old, new, regions)
            if r is not None:
                m, noop = r
                if noop:
                    why.append(f"{rel}: comments/docstrings/whitespace only")
                    return self.idx["opens"].get(rel, 0)
                return m
        try:
            told = ast.parse(old)
        except SyntaxError:
            raise Everything("old unparsable " + rel)
        try:
            tnew = ast.parse(new)
        except SyntaxError:
            why.append(f"{rel}: syntax error")
            return self.import_break(mod)
        self.old_text = old
        m = 0
        noop = True
        for oa, ob, na, nb in regions:
            r, changed = self.region(told.body, tnew.body, oa, ob, na, nb, ("m", None), None, True)
            m |= r
            noop = noop and not changed
        if noop:
            why.append(f"{rel}: comments/docstrings/whitespace only")
            return self.idx["opens"].get(rel, 0)
        return m

    def snippet_change(self, rel, old, new, regions):
        """Parse only the function(s) that contain every changed region; None = use full parse."""
        outline = self.idx.get("outline", {}).get(rel)
        if not outline or not regions:
            return None
        groups = {}
        for reg in regions:
            oa, ob, na, nb = reg
            hit = None
            for e in outline:
                body_start, end = e[0], e[1]
                inside = (body_start <= oa and ob <= end) if ob >= oa else (body_start < oa <= end)
                if inside:
                    hit = e
                    break
            if hit is None:
                return None
            groups.setdefault(hit, []).append(reg)
        old_lines = old.split("\n")
        new_lines = new.split("\n")
        m = 0
        noop = True
        for e, regs in groups.items():
            body_start, end, first, defline, ctx, parents = e

            def delta(upto):
                return sum((nb - na) - (ob - oa) for oa, ob, na, nb in regions if ob < upto)

            d_first = delta(first)
            n_first, n_def = first + d_first, defline + d_first
            n_end = end + delta(end + 1)
            hdr = [f"{ind}class _P:" for ind in parents]
            pad = first - 1 - len(hdr)
            if pad < 0 or n_first - 1 - len(hdr) < 0:
                return None
            try:
                to = ast.parse("\n" * pad + "".join(h + "\n" for h in hdr) + "\n".join(old_lines[first - 1:end]) + "\n")
                tn = ast.parse("\n" * (n_first - 1 - len(hdr)) + "".join(h + "\n" for h in hdr)
                               + "\n".join(new_lines[n_first - 1:n_end]) + "\n")
            except SyntaxError:
                return None
            fo, fn = to.body[0], tn.body[0]
            for _ in parents:
                if not isinstance(fo, ast.ClassDef) or not isinstance(fn, ast.ClassDef) or len(fo.body) != 1 or len(fn.body) != 1:
                    return None
                fo, fn = fo.body[0], fn.body[0]
            if not (isinstance(fo, FUNC_TYPES) and isinstance(fn, FUNC_TYPES) and fo.lineno == defline
                    and fo.end_lineno == end and fn.lineno == n_def and fn.end_lineno == n_end):
                return None
            for oa, ob, na, nb in regs:
                r, changed = self.region([fo], [fn], oa, ob, na, nb, ctx, None, False)
                m |= r
                noop = noop and not changed
        return m, noop

    def region(self, oblock, nblock, oa, ob, na, nb, ctx, parent, docblock):
        O = [s for s in oblock if overlaps(s, oa, ob)]
        N = [s for s in nblock if overlaps(s, na, nb)]
        if len(O) == 1 and len(N) == 1 and type(O[0]) is type(N[0]):
            o, n = O[0], N[0]
            ochild = dict(children(o))
            nchild = dict(children(n))
            for field, oblk in ochild.items():
                nblk = nchild.get(field)
                if not nblk:
                    continue
                of, oe = block_span(oblk)
                nf, ne = block_span(nblk)
                oin = (of <= oa and ob <= oe) if ob >= oa else (of < oa <= oe)
                nin = (nf <= na and nb <= ne) if nb >= na else (nf < na <= ne)
                if not (oin and nin):
                    continue
                if isinstance(o, ast.ClassDef) and ctx[0] != "f":
                    q = f"{ctx[1].split(':', 1)[1]}.{o.name}" if ctx[0] == "c" else o.name
                    nctx = ("c", f"{self.mod}:{q}")
                elif isinstance(o, FUNC_TYPES) and ctx[0] != "f":
                    nctx = ("f", o, ctx)
                else:
                    nctx = ctx
                return self.region(oblk, nblk, oa, ob, na, nb, nctx, o,
                                   isinstance(o, DEF_TYPES) and field == "body")
        # final level
        Od = [s for s in O if not (docblock and oblock and s is oblock[0] and is_docstring(s))]
        Nd = [s for s in N if not (docblock and nblock and s is nblock[0] and is_docstring(s))]
        if [dump(x) for x in Od] == [dump(x) for x in Nd]:
            return 0, False
        if ctx[0] == "f":
            return self.func_region(O, N, oblock, oa, ob, ctx, parent), True
        if len(Od) == 1 and len(Nd) == 1:
            r = self.refine_lambda_only(Od[0], Nd[0])
            if r is not None:
                return r, True
        ck = ctx[1] if ctx[0] == "c" else None
        return self.body(Od, Nd, ck), True

    def func_region(self, O, N, oblock, oa, ob, ctx, parent):
        rel = self.rel
        m = 0
        spans = [(first_line(s), s.end_lineno) for s in O]
        if not O or len(N) > len(O):
            # inserted statements: tests that reached the insertion point
            prev = [s for s in oblock if s.end_lineno < (oa if ob < oa else oa)]
            nxt = [s for s in oblock if first_line(s) > (ob if ob >= oa else oa - 1)]
            if prev:
                spans.append((first_line(prev[-1]), prev[-1].end_lineno))
            if nxt:
                spans.append((first_line(nxt[0]), nxt[0].end_lineno))
            if not prev and parent is not None:
                spans.append((first_line(parent), parent.end_lineno))
        func, owner_ctx = ctx[1], ctx[2]
        import_time = any(self.is_import_time(rel, a, b) for a, b in spans)
        if len(O) == 1 and len(N) == 1:
            r = self.refine_stmt(O[0], N[0])
            if r is not None:
                self.why.append(f"condition profile in {func.name}")
                return r
        for a, b in spans:
            m |= self.cover(rel, a, b)
        if import_time:
            ck = owner_ctx[1] if owner_ctx[0] == "c" else None
            self.why.append(f"import-time code in {func.name}")
            m |= self.import_break(self.mod)
            m |= self.name_change(func.name, ck)
            if ck:
                m |= self.class_users(ck)
        else:
            self.why.append(f"lines in {func.name}")
        return m

    # --- condition profile (weak mutation) refinement
    OPNODES = (ast.cmpop, ast.boolop, ast.operator, ast.unaryop, ast.expr_context)

    @staticmethod
    def expr_diffs(a, b, out, parents, parent=None):
        """Top-most differing (old, new) node pairs, in one linear pass; records old parents.
        Returns True when a and b are equal."""
        if type(a) is not type(b):
            out.append((a, b))
            return False
        parents[id(a)] = parent
        local = []
        equal = True
        here = False
        for f in a._fields:
            va, vb = getattr(a, f, None), getattr(b, f, None)
            if isinstance(va, list) or isinstance(vb, list):
                va, vb = va or [], vb or []
                if len(va) != len(vb):
                    here = True
                    break
                for x, y in zip(va, vb):
                    if isinstance(x, Selector.OPNODES) or isinstance(y, Selector.OPNODES):
                        if type(x) is not type(y):
                            here = True
                    elif isinstance(x, ast.AST) and isinstance(y, ast.AST):
                        if not Selector.expr_diffs(x, y, local, parents, a):
                            equal = False
                    elif x != y:
                        here = True
                if here:
                    break
            elif isinstance(va, Selector.OPNODES) or isinstance(vb, Selector.OPNODES):
                if type(va) is not type(vb):
                    here = True
                    break
            elif isinstance(va, ast.AST) or isinstance(vb, ast.AST):
                if va is None or vb is None or type(va) is not type(vb):
                    here = True
                    break
                if not Selector.expr_diffs(va, vb, local, parents, a):
                    equal = False
            elif va != vb:
                here = True
                break
        if here:
            Selector.note_parents_children(a, parents)
            out.append((a, b))
            return False
        out.extend(local)
        return equal

    @staticmethod
    def note_parents_children(node, parents):
        for c in ast.iter_child_nodes(node):
            if id(c) not in parents:
                parents[id(c)] = node
                Selector.note_parents_children(c, parents)

    def prof_entry(self, node):
        if not hasattr(node, "end_col_offset"):
            return None
        return self.idx["prof"].get(self.rel, {}).get(
            (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset))

    CMP_FAMILY = {}
    for _f in (("Eq", "NotEq", "Is", "IsNot"), ("Lt", "LtE", "Gt", "GtE", "Eq", "NotEq")):
        for _o in _f[:4]:
            CMP_FAMILY.setdefault(_o, _f)

    def refine_pair(self, a, b, parents):
        """Tests whose recorded evaluations of old node `a` differ under new node `b`;
        None when the profile cannot decide (not instrumented, or an import-time
        evaluation would change)."""
        if a is None or b is None:
            return None
        e = self.prof_entry(a)
        if e is None:
            return None
        kind, op, imp_sigs, sigs = e
        affected = None
        if isinstance(a, ast.Compare) and isinstance(b, ast.Compare) and kind == "cmp":
            fam = self.CMP_FAMILY.get(op)
            same_operands = (len(a.ops) == 1 and len(b.ops) == 1 and dump(a.left) == dump(b.left)
                             and dump(a.comparators[0]) == dump(b.comparators[0]))
            new = type(b.ops[0]).__name__ if len(b.ops) == 1 else None
            if not same_operands or not fam or new not in fam:
                affected = lambda sig: True  # noqa: E731
            else:
                i, j = fam.index(op), fam.index(new)
                affected = lambda sig: len(sig) != len(fam) or sig[i] != sig[j] or "E" in (sig[i], sig[j])  # noqa: E731
        elif isinstance(a, ast.BoolOp) and kind == "bool":
            leaves = {dump(v): k for k, v in enumerate(a.values)}
            if len(leaves) != len(a.values):
                affected = lambda sig: True  # noqa: E731
            else:
                bool_ctx = self.in_bool_ctx(a, parents)

                def ev(node, vec, used):
                    if isinstance(node, ast.BoolOp):
                        res = None
                        for v in node.values:
                            res = ev(v, vec, used)
                            t = res[1]
                            if (isinstance(node.op, ast.And) and not t) or (isinstance(node.op, ast.Or) and t):
                                return res
                        return res
                    k = leaves[dump(node)]  # KeyError: an operand the old expression did not have
                    c = vec[k]
                    if c not in "01":
                        raise KeyError(c)
                    used.append(k)
                    return k, c == "1"

                def affected(sig):
                    try:
                        if "X" in sig or "E" in sig or len(sig) != len(a.values):
                            return True
                        old_used = [k for k, c in enumerate(sig) if c != "-"]
                        old_res = old_used[-1]
                        used = []
                        k, t = ev(b, sig, used)
                        return not (used == old_used and (t == (sig[old_res] == "1") if bool_ctx else k == old_res))
                    except KeyError:
                        return True
        if affected is None:
            return None
        if any(affected(sg) for sg in imp_sigs):
            return None
        m = 0
        for sig, mask in sigs:
            if affected(sig):
                m |= mask
        return m

    @staticmethod
    def all_evals(sigs):
        m = 0
        for _, mask in sigs:
            m |= mask
        return m

    @staticmethod
    def in_bool_ctx(node, parents):
        p = parents.get(id(node))
        while True:
            if isinstance(p, (ast.If, ast.While, ast.IfExp, ast.Assert)) and getattr(p, "test", None) is node:
                return True
            if isinstance(p, ast.UnaryOp) and isinstance(p.op, ast.Not):
                return True
            if isinstance(p, ast.comprehension) and any(x is node for x in p.ifs):
                return True
            if isinstance(p, ast.BoolOp):
                node, p = p, parents.get(id(p))
                continue
            return False

    def refine_stmt(self, o, n):
        """Mask for a statement whose only differences are profiled comparisons/boolops."""
        if type(o) is not type(n):
            return None
        pairs, parents = [], {}
        self.expr_diffs(o, n, pairs, parents)
        if not pairs:
            return None
        m = 0
        for a, b in pairs:
            r = self.refine_pair(a, b, parents)
            if r is None:
                return None
            m |= r
        return m | (self.idx["untrusted"] & self.cover(self.rel, first_line(o), o.end_lineno))

    def refine_lambda_only(self, o, n):
        """A class/module-level statement whose differences all sit inside lambda bodies only
        changes what those lambdas do when they are called (at test time)."""
        if type(o) is not type(n) or not isinstance(o, (ast.Assign, ast.AnnAssign)):
            return None
        pairs, parents = [], {}
        self.expr_diffs(o, n, pairs, parents)
        if not pairs:
            return None
        m = 0
        for a, b in pairs:
            lam, x = None, a
            while x is not None:
                p = parents.get(id(x))
                if isinstance(p, ast.Lambda) and p.body is x:
                    lam = p
                    break
                x = p
            if lam is None:
                return None
            pl = parents.get(id(lam))
            if isinstance(pl, ast.Call) and pl.func is lam:
                return None  # called right away at import time
            if self.is_import_time_lambda(lam):
                return None
            r = self.refine_pair(a, b, parents)
            if r is None:
                r = self.cover(self.rel, lam.lineno, lam.end_lineno)
            m |= r
        self.why.append("lambda body only")
        return m | (self.idx["untrusted"] & self.cover(self.rel, first_line(o), o.end_lineno))

    def is_import_time_lambda(self, lam):
        for x in ast.walk(lam.body):
            e = self.prof_entry(x)
            if e is not None and e[2]:
                return True
        return False

    @staticmethod
    def key(s):
        if isinstance(s, DEF_TYPES):
            return ("def", s.name)
        if isinstance(s, ast.Assign):
            return ("as", tuple(dump(t) for t in s.targets))
        if isinstance(s, ast.AnnAssign):
            return ("as", (dump(s.target),))
        return None

    def body(self, olds, news, ck):
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
                    m |= self.pair(o, n_block[partner], ck)
                else:
                    m |= self.stmt(o, None, ck)
            for jx, n in enumerate(n_block):
                if jx not in used:
                    m |= self.stmt(None, n, ck)
        return m

    def name_change(self, name, ck):
        if ck:
            return self.class_attr(ck, name)
        return self.module_name(self.mod, name)

    # --- import-time safety
    def safe_expr(self, e):
        for n in ast.walk(e):
            if isinstance(n, ast.Lambda):
                continue
            if isinstance(n, (ast.Call, ast.Subscript, ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr, ast.Compare)):
                if not self.inside_lambda(e, n):
                    return False
            if isinstance(n, ast.BinOp) and not (isinstance(n.left, ast.Constant) and isinstance(n.right, ast.Constant)):
                if not self.inside_lambda(e, n):
                    return False
            if isinstance(n, (ast.Name, ast.Attribute)) and isinstance(getattr(n, "ctx", None), ast.Load):
                c = chain(n)
                if c and ".".join(c) not in self.old_text and not self.inside_lambda(e, n):
                    return False
        return True

    @staticmethod
    def inside_lambda(root, node):
        for lam in ast.walk(root):
            if isinstance(lam, ast.Lambda) and lam is not node:
                for x in ast.walk(lam.body):
                    if x is node:
                        return True
                for x in ast.walk(lam.args):
                    if x is node:
                        return False
        return False

    def safe_pair(self, o, n):
        """Can replacing import-time statement o by n (either may be None) raise at import?"""
        if n is None:
            return True  # removal itself cannot raise; readers are found through the name rules
        if o is None:
            if isinstance(n, (ast.Pass,)) or is_docstring(n):
                return True
            if isinstance(n, FUNC_TYPES):
                return not n.decorator_list and all(self.safe_expr(d) for d in n.args.defaults + n.args.kw_defaults if d is not None)
            if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None:
                return self.safe_expr(n.value)
            return False
        return self.safe_diff(o, n)

    def safe_diff(self, a, b):
        if type(a) is not type(b):
            if isinstance(b, ast.AST):
                return self.safe_expr(b) if isinstance(b, ast.expr) else False
            return False
        if isinstance(a, ast.Lambda):
            return dump(a.args) == dump(b.args) and all(
                self.safe_expr(d) for d in b.args.defaults if d is not None)
        if isinstance(a, ast.Constant):
            return True
        if isinstance(a, ast.Compare) and len(a.ops) == len(b.ops) and dump(a.left) == dump(b.left) and \
                [dump(x) for x in a.comparators] == [dump(x) for x in b.comparators]:
            # same operands, different operator: ==/!=/is/<... over values that already compared fine
            return all(type(x) in (ast.Eq, ast.NotEq, ast.Is, ast.IsNot, ast.In, ast.NotIn) or
                       type(y) in (ast.Eq, ast.NotEq, ast.Is, ast.IsNot) or
                       (type(x) in (ast.Lt, ast.LtE, ast.Gt, ast.GtE) and type(y) in (ast.Lt, ast.LtE, ast.Gt, ast.GtE))
                       for x, y in zip(a.ops, b.ops))
        if isinstance(a, ast.UnaryOp) and isinstance(a.op, ast.Not) and dump(a.operand) == dump(b):
            return True
        if isinstance(a, (ast.Dict,)):
            ok = {(dump(k) if k else "**", dump(v)) for k, v in zip(a.keys, a.values)}
            old_vals = {dump(v) for v in a.values}
            for k, v in zip(b.keys, b.values):
                pair = (dump(k) if k else "**", dump(v))
                if pair in ok or ((k is None or self.safe_expr(k)) and dump(v) in old_vals):
                    continue
                # changed value of an existing key, or new entry
                if not ((k is None or self.safe_expr(k)) and self.safe_expr(v)):
                    olds = [ov for ok_, ov in zip(a.keys, a.values) if ok_ is not None and k is not None and dump(ok_) == dump(k)]
                    if not (olds and self.safe_diff(olds[0], v)):
                        return False
            return True
        if isinstance(a, (ast.Set, ast.List, ast.Tuple)):
            ok = {dump(x) for x in a.elts}
            return all(dump(x) in ok or self.safe_expr(x) for x in b.elts)
        if isinstance(a, ast.AST):
            for field in a._fields:
                va, vb = getattr(a, field, None), getattr(b, field, None)
                if isinstance(va, list) or isinstance(vb, list):
                    va, vb = va or [], vb or []
                    if len(va) != len(vb):
                        if field in ("body", "orelse", "finalbody") and isinstance(a, ast.stmt):
                            return False
                        return all(self.safe_expr(x) if isinstance(x, ast.expr) else False for x in vb
                                   if dump(x) not in {dump(y) for y in va})
                    for x, y in zip(va, vb):
                        if isinstance(x, ast.AST) or isinstance(y, ast.AST):
                            if dump(x) != dump(y) and not self.safe_diff(x, y):
                                return False
                        elif x != y:
                            return False
                elif isinstance(va, ast.AST) or isinstance(vb, ast.AST):
                    if va is None or vb is None:
                        if vb is not None and not (isinstance(vb, ast.expr) and self.safe_expr(vb)):
                            return False
                    elif dump(va) != dump(vb) and not self.safe_diff(va, vb):
                        return False
                elif va != vb:
                    # operator/identifier field changed (e.g. Sub -> Add): may raise
                    if isinstance(a, ast.expr) and isinstance(a, ast.Name):
                        return self.safe_expr(b)
                    return False
            return True
        return a == b

    def pair(self, o, n, ck):
        if isinstance(o, ast.ClassDef) and isinstance(n, ast.ClassDef):
            q = f"{ck.split(':', 1)[1]}.{o.name}" if ck else o.name
            new_ck = f"{self.mod}:{q}"
            m = 0
            if [dump(x) for x in o.bases + o.keywords] != [dump(x) for x in n.bases + n.keywords] or \
                    not self.same_decorators(o.decorator_list, n.decorator_list):
                self.why.append(f"class header {new_ck}")
                m |= self.class_users(new_ck) | self.name_change(o.name, ck) | self.subclass_registry(o, n)
                m |= self.maybe_import_break(self.mod)
            return m | self.body(o.body, n.body, new_ck)
        if isinstance(o, FUNC_TYPES) and isinstance(n, FUNC_TYPES):
            m = self.cover(self.rel, first_line(o), o.end_lineno)
            where = f"{ck or self.mod}.{o.name}"
            sig_o = [dump(x) for x in o.decorator_list + [o.args] + ([o.returns] if o.returns else [])]
            sig_n = [dump(x) for x in n.decorator_list + [n.args] + ([n.returns] if n.returns else [])]
            body_import = o.body and self.is_import_time(self.rel, o.body[0].lineno, o.end_lineno)
            if sig_o != sig_n or type(o) is not type(n):
                self.why.append(f"signature {where}")
                m |= self.name_change(o.name, ck)
                if [dump(x) for x in o.decorator_list] != [dump(x) for x in n.decorator_list] or not all(
                        self.safe_expr(d) for d in n.args.defaults + n.args.kw_defaults if d is not None):
                    m |= self.maybe_import_break(self.mod)
            if body_import and [dump(x) for x in o.body] == [dump(x) for x in n.body]:
                body_import = False  # only the signature changed (handled above)
            if body_import:
                self.why.append(f"import-time function {where}")
                m |= self.name_change(o.name, ck) | self.import_break(self.mod)
                if ck:
                    m |= self.class_users(ck)
            else:
                self.why.append(f"function {where}")
            return m
        return self.stmt(o, n, ck)

    def same_decorators(self, olds, news):
        """Equal, or differing only in constant arguments of identity decorator factories."""
        if len(olds) != len(news):
            return False
        ident = self.idx.get("identity_decorators", set())
        for a, b in zip(olds, news):
            if dump(a) == dump(b):
                continue
            if not (isinstance(a, ast.Call) and isinstance(b, ast.Call) and dump(a.func) == dump(b.func)
                    and (self.rel, a.lineno, a.col_offset) in ident
                    and len(a.args) == len(b.args) and [k.arg for k in a.keywords] == [k.arg for k in b.keywords]
                    and all(isinstance(x, ast.Constant) for x in b.args + [k.value for k in b.keywords])):
                return False
        return True

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

    def stmt(self, o, n, ck):
        names = defined_names(o) | defined_names(n)
        m = 0
        if o is not None:
            m |= self.cover(self.rel, first_line(o), o.end_lineno)
        for node in (o, n):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self.check_import(node)
            if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
                raise Everything("star import changed")
        if isinstance(o, (ast.Import, ast.ImportFrom)) or isinstance(n, (ast.Import, ast.ImportFrom)):
            # imports change what gets loaded at import time (registries, lazy-load checks)
            m |= self.import_break(self.mod)
        elif not self.safe_pair(o, n):
            self.why.append(f"may raise at import ({ck or self.mod})")
            m |= self.maybe_import_break(self.mod)
        if (o is None or n is None) and names and not ck:
            # removing a module-level name that import-time code reads raises NameError
            if n is None and any(self.idx["units"][u][3] != "f" for nm in names
                                 for u, r, k, *_pos in self.idx["refs"].get(nm, ()) if r):
                m |= self.maybe_import_break(self.mod)
        if isinstance(o or n, ast.ClassDef) and (o is None or n is None):
            m |= self.subclass_registry(o, n)
        if ck and isinstance(o or n, FUNC_TYPES):
            m |= self.method_added_removed(ck, (o or n).name)
        if not names:
            self.why.append(f"stmt without names in {ck or self.mod}")
            return m | (self.class_users(ck) if ck else self.module_users(self.mod))
        self.why.append(f"names {sorted(names)[:4]} in {ck or self.mod}")
        added = defined_names(n) - defined_names(o)
        for name in names:
            if ck and name in added:
                m |= self.class_attr(ck, name, added=True)
            else:
                m |= self.name_change(name, ck)
        return m

    def method_added_removed(self, ck, name):
        """A method appeared/vanished: overrides change for every instance of the class,
        except generator *_sql methods, which only matter for their expression type."""
        if name.endswith("_sql") and not name.startswith("_"):
            return self.class_attr(ck, name)
        return self.class_users(ck)

    def check_import(self, node):
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
        self.why = why = []
        files = parse_patch(patch_text)
        self.files = files
        if not files:
            return 0, ["empty patch"]
        self.new_modules = {module_name(f["new"]) for f in files if f["new"] and f["new"].endswith(".py")}
        self.changed = {f["old"] or f["new"] for f in files} | {f["new"] or f["old"] for f in files}
        self.pending_imports = set()
        self.patched = {}
        # free-rider tiers: later shared-pool tests, near lines, enclosing unit, module users
        self.plausible = [0, 0, 0, 0]
        mask = 0
        for f in files:
            mask |= self.file_change(f)
        if self.pending_imports:
            failed = self.check_imports()
            why.append(f"import check {sorted(self.pending_imports)[:3]} -> failed {failed[:3]}")
            for mod in failed:
                mask |= self.import_break(mod)
        for order in self.idx.get("pool_groups", ()):
            # a pool worker killed by this change breaks the class's later pool tests
            hit = [k for k, b in enumerate(order) if mask >> b & 1]
            if hit:
                for b in order[hit[0] + 1:]:
                    self.plausible[0] |= 1 << b
        return mask, why

    def note_plausible(self, rel, regions, reads):
        """Record tests that are plausibly affected but not required. They ride for free in
        worker time that the required tests leave idle (see pack_free); they never change cost."""
        p = self.plausible
        p[1] |= reads
        if not rel:
            return
        if rel.endswith(".py"):
            units = self.memo.get("all_units")
            if units is None:
                units = {}
                for i, (urel, a, b, *_r) in enumerate(self.idx["units"]):
                    units.setdefault(urel, []).append((a, b, i))
                self.memo["all_units"] = units
            for oa, ob, _na, _nb in regions:
                lo, hi = (oa, ob) if ob >= oa else (ob, oa)
                p[1] |= self.cover(rel, lo - 5, hi + 5)
                best = None
                for a, b, i in units.get(rel, ()):
                    if a <= lo and hi <= b and (best is None or b - a < best[1] - best[0]):
                        best = (a, b, i)
                if best:
                    p[2] |= self.cover(rel, best[0], best[1])
                    ck = self.idx["units"][best[2]][6]
                    if ck:
                        p[2] |= self.class_users(ck)
            if rel.startswith("sqlglot/"):
                p[3] |= self.module_users(module_name(rel))
        if rel.startswith("tests/"):
            p[3] |= self.idx["testfiles"].get(rel, 0)

    def read_src(self, rel):
        try:
            return (SRC / rel).read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

    def file_change(self, f):
        why = self.why
        old_rel, new_rel = f["old"], f["new"]
        rel = new_rel or old_rel
        if rel is None:
            raise Everything("no path")
        if f["binary"]:
            if rel.startswith(("sqlglot/", "tests/")):
                raise Everything("binary " + rel)
            return 0
        if old_rel and new_rel and old_rel != new_rel:
            return self.file_change({**f, "new": None, "hunks": []}) | self.file_change({**f, "old": None})
        old_text = self.read_src(old_rel) if old_rel else None
        if old_rel and old_text is None:
            raise Everything("unknown file " + old_rel)
        if new_rel:
            new_text, regions = apply_hunks(old_text or "", f["hunks"])
            if new_rel.endswith(".py"):
                self.patched[new_rel] = new_text
        else:
            new_text, regions = None, []
        is_py = rel.endswith(".py")
        top = rel.split("/")[0]
        reads = self.idx["opens"].get(rel, 0) | self.idx["code_opens"].get(rel, 0)
        try:
            self.note_plausible(old_rel, regions, reads)
        except Exception as e:  # free riders are optional; never let them break selection
            log("plausible:", repr(e))
        if top in ("sqlglot", "tests") and is_py and not rel.startswith("tests/fixtures/"):
            name = rel.rsplit("/", 1)[-1]
            if name == "conftest.py" or (rel.startswith("tests/") and name == "__init__.py"):
                raise Everything("test infra " + rel)
            if old_text is None:
                why.append(f"new module {rel}")
                return self.idx["walkers"] | self.idx["importsens"]
            if new_text is None:
                why.append(f"deleted module {rel}")
                return self.import_break(module_name(rel))
            return self.py_change(rel, old_text, new_text, regions, why) | self.idx["opens"].get(rel, 0)
        if rel.startswith("tests/"):
            return self.fixture_tests(rel, old_text, new_text, reads) | self.new_or_deleted_fixture(rel, old_text, new_text)
        if top == "sqlglot":
            if rel.endswith((".pyi", "py.typed", ".md", ".txt")):
                return reads
            raise Everything("non-py under sqlglot " + rel)
        if rel in ("setup.cfg", "pyproject.toml", "setup.py", "pytest.ini", "tox.ini", "conftest.py"):
            raise Everything("config " + rel)
        if is_py and old_text is not None and new_text is not None:
            try:
                if dump(strip_docstrings(ast.parse(old_text))) == dump(strip_docstrings(ast.parse(new_text))):
                    return self.idx["opens"].get(rel, 0)
            except SyntaxError:
                pass
        why.append(f"other file {rel}")
        return reads

    def fixture_tests(self, rel, old_text, new_text, reads):
        """Tests that read a data file. Files a setUpClass loads into class attributes are
        charged to the tests that read those attributes, if the real loader still accepts the
        new text (otherwise setUpClass fails and the whole class with it)."""
        if "opens_c" not in self.idx or new_text is None or old_text is None:
            return reads
        m = self.idx["opens_c"].get(rel, 0) | self.idx["code_opens"].get(rel, 0)
        for owner in self.idx["opens_s"].get(rel, ()):
            ents = [e for e in self.idx["setup_attr"].get(rel, ()) if e[0] == owner]
            if ents and all(self.loader_ok(loader, path, new_text) for _, _, loader, path in ents):
                ar = self.idx["attr_reads"].get(owner, {})
                fr = self.idx.get("field_reads", {}).get(owner, {})
                for _, attr, loader, path in ents:
                    fields = self.changed_fields(loader, path, old_text, new_text)
                    if fields is None or attr not in fr:
                        m |= ar.get(attr, 0)
                    else:
                        # only tests that used a changed field (e.g. the SQL vs the expected output)
                        for f in fields:
                            m |= fr[attr].get(str(f), 0)
                self.why.append(f"fixture via {owner}.{','.join(e[1] for e in ents)}")
            else:
                m |= self.idx["owner_tests"].get(owner, 0) or reads
        return m

    def run_loader(self, loader, path, text):
        """The pristine tests/helpers.py loader applied to `text`; raises what it raises."""
        import io
        ns = self.memo.get("helpers_ns")
        if ns is None:
            ns = {"__name__": "br_helpers", "__file__": str(SRC / "tests" / "helpers.py")}
            exec(compile((SRC / "tests" / "helpers.py").read_text(encoding="utf-8"), "helpers.py", "exec"), ns)
            self.memo["helpers_ns"] = ns
        ns["open"] = lambda *a, **k: io.StringIO(text)
        return list(ns[loader](path))

    def loader_ok(self, loader, path, new_text):
        if "tests/helpers.py" in self.changed:
            return False
        try:
            self.run_loader(loader, path, new_text)
            return True
        except Exception:
            return False

    def changed_fields(self, loader, path, old_text, new_text):
        """Indices of tuple fields that differ between the loaded old and new rows; None when
        rows shift, are not tuples, or a non-string field (e.g. metadata) changed."""
        try:
            old_rows = self.run_loader(loader, path, old_text)
            new_rows = self.run_loader(loader, path, new_text)
        except Exception:
            return None
        if len(old_rows) != len(new_rows):
            return None
        fields = set()
        for o, n in zip(old_rows, new_rows):
            if type(o) is not tuple or type(n) is not tuple or len(o) != len(n):
                if o != n:
                    return None
                continue
            for j, (x, y) in enumerate(zip(o, n)):
                if x != y:
                    if type(x) is not str or type(y) is not str:
                        return None
                    fields.add(j)
        return fields

    def new_or_deleted_fixture(self, rel, old_text, new_text):
        if old_text is not None and new_text is not None:
            return 0
        d = rel.rsplit("/", 1)[0] + "/"
        m = self.idx["walkers"]
        for k, s in self.idx["opens"].items():
            if k.startswith(d):
                m |= s
        return m


# TPC-DS equivalence check (v5.5). test_execute_tpcds (21.5 s): for every TPC-DS fixture query the
# parent runs parse_one and pickles the tree to a pool child; for queries marked `execute` the child
# runs executor.execute() (optimize, then Plan + PythonExecutor) on the tables find_tables names, and
# the parent compares the rows positionally with duckdb running transpile(sql, write="duckdb").
# TPCDS_CHECK reproduces that up to optimize() for every query and hashes each part. With the patched
# source swapped in:
#   * any exception, or a different duckdb SQL for an executed query  -> the test may fail;
#   * a non-executed query only has to parse and pickle;
#   * an executed query whose table list and optimized tree are unchanged gets the same rows;
#   * the others are executed for real and their rows compared with the pristine rows (ref mode,
#     computed at build time), within a CPU budget.
# tpcds_region_ok() makes sure the patch cannot change the code that runs outside these calls.
TPCDS_CHECK = r'''
import gzip, hashlib, ast as _ast, csv, json, pickle, sys, time
t0 = time.process_time()
from sqlglot import exp, find_tables, parse_one, transpile
from sqlglot.executor import execute
from sqlglot.executor.table import Table
from sqlglot.optimizer import optimize
from sqlglot.schema import ensure_schema
from tests.helpers import FIXTURES_DIR, TPCDS_SCHEMA, load_sql_fixture_pairs

SKIP = {"parent", "_hash"}


def dump(x, out, seen):
    """Exact structural dump; shared or cyclic objects become back-references (#n)."""
    if isinstance(x, (exp.Expr, dict, list, set, frozenset)) or hasattr(x, "__dict__"):
        k = id(x)
        if k in seen:
            out.append("#%d" % seen[k][0])
            return
        seen[k] = (len(seen), x)
    if isinstance(x, exp.Expr):
        out.append("<" + type(x).__qualname__)
        slots = [s for c in type(x).__mro__ for s in getattr(c, "__slots__", ())]
        for k, v in [(s, getattr(x, s, None)) for s in slots] + sorted(getattr(x, "__dict__", {}).items()):
            if k not in SKIP:
                out.append(k + "=")
                dump(v, out, seen)
        out.append(">")
    elif isinstance(x, dict):
        out.append("{")
        # args insertion order varies run to run in the pinned code itself; compare sorted
        for k, v in (sorted(x.items()) if all(type(k) is str for k in x) else x.items()):
            dump(k, out, seen)
            out.append(":")
            dump(v, out, seen)
        out.append("}")
    elif isinstance(x, (list, tuple, set, frozenset)):
        out.append(type(x).__name__ + "[")
        for v in (sorted(x, key=repr) if isinstance(x, (set, frozenset)) else x):
            dump(v, out, seen)
            out.append(",")
        out.append("]")
    elif hasattr(x, "__dict__") and not isinstance(x, type):
        out.append("(" + type(x).__qualname__)
        dump(vars(x), out, seen)
        out.append(")")
    else:
        out.append(type(x).__name__ + ":" + repr(x))


def digest(x):
    out = []
    dump(x, out, {})
    return hashlib.sha256("\x00".join(out).encode("utf-8", "surrogatepass")).hexdigest()[:20]


def load_table(name):
    """tests/test_executor.py TestExecutor.setUpClass for one TPC-DS table, then the pickle round
    trip it makes once as the pool's initargs."""
    return pickle.loads(pickle.dumps(_load_table(name)))


def _load_table(name):
    columns = TPCDS_SCHEMA[name]
    fn = f"{FIXTURES_DIR}/optimizer/tpc-ds/{name}.csv.gz"
    with open(fn, "rb") as f:
        gz = f.read(2) == b"\x1f\x8b"
    fh = gzip.open(fn, "rt", newline="") if gz else open(fn, encoding="utf-8", newline="")
    reader = csv.reader(fh, delimiter="|")
    rows, ctypes = [], []
    next(reader)
    for row in reader:
        if not ctypes:
            for v in row:
                try:
                    ctypes.append(type(_ast.literal_eval(v)))
                except (ValueError, SyntaxError):
                    ctypes.append(str)
        rows.append(tuple(None if (t is not str and v == "") else t(v) for t, v in zip(ctypes, row)))
    return Table(columns=columns, rows=rows)


def result_digest(table):
    rows = [tuple((type(v).__name__, v) for v in r) for r in table.rows]
    return hashlib.sha256(repr((list(table.columns), rows)).encode()).hexdigest()[:20]


def _as_test_sees(v):
    """What the test's assert_frame_equal(duckdb_df, DataFrame(rows), check_dtype=False) can tell
    apart: None becomes NaN, numbers compare by value whatever their type. Exact otherwise."""
    if v is None or (isinstance(v, float) and v != v):
        return ("nan",)
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, (int, float)) or type(v).__name__ == "Decimal":
        f = float(v)
        return ("num", 0.0 if f == 0 else f)
    return (type(v).__name__, v)


def loose_digest(table):
    rows = [tuple(_as_test_sees(v) for v in r) for r in table.rows]
    return hashlib.sha256(repr((list(table.columns), rows)).encode()).hexdigest()[:20]


def run(e, names, tables):
    t = time.process_time()
    tabs = {n: tables[n] for n in names}  # mp_execute picks them out of the initargs dict
    r = pickle.loads(pickle.dumps(execute(e, schema=TPCDS_SCHEMA, tables=tabs)))
    return (result_digest(r), loose_digest(r)), time.process_time() - t, r


_DUCK = []


def duck_digest(sql_text):
    """The test's reference result: duckdb (unpatched) on transpile(sql, write="duckdb") over
    views made exactly as TestExecutor.setUpClass makes them. (strict, loose) digests, CPU."""
    import types
    _duck_conn()
    t1 = time.process_time()
    cur = _DUCK[0].execute(sql_text)
    res = types.SimpleNamespace(columns=[d[0] for d in cur.description], rows=cur.fetchall())
    return (result_digest(res), loose_digest(res)), time.process_time() - t1


def _duck_conn():
    if not _DUCK:
        import duckdb
        c = duckdb.connect()
        c.execute("SET threads TO 1")  # deterministic order and CPU
        for table, columns in TPCDS_SCHEMA.items():
            file_name = f"{FIXTURES_DIR}/optimizer/tpc-ds/{table}.csv.gz"
            c.execute(
                f"""
                CREATE VIEW {table} AS
                SELECT *
                FROM READ_CSV('{file_name}', delim='|', header=True, columns={columns})
                """
            )
        _DUCK.append(c)
    return _DUCK[0]


def test_assert(duck_sql, table):
    """The test's own check for one query (TestExecutor.cached_execute + subtestHelper)."""
    import numpy as np
    import pandas as pd
    from pandas.testing import assert_frame_equal
    a = _duck_conn().execute(duck_sql).fetchdf()
    b = pd.DataFrame(((np.nan if c is None else c for c in r) for r in table.rows), columns=table.columns)
    try:
        assert_frame_equal(a, b, check_dtype=False, check_index_type=False)
        return True
    except AssertionError:
        return False


def verdict(**k):
    k["cpu"] = round(time.process_time() - t0, 2)
    print("VERDICT " + json.dumps(k), file=sys.stderr)
    sys.exit(0)


REF, BUDGET, LIMIT = globals().get("REF"), globals().get("BUDGET", 0.0), globals().get("LIMIT", 1e9)
EXEC_ALL = globals().get("EXEC_ALL", False)  # the patch changes code the execution itself runs
EXEC_SET = globals().get("EXEC_SET")  # with EXEC_ALL: only these queries run changed code (None: all)
if EXEC_SET is not None:
    EXEC_SET = set(EXEC_SET)
sqls = list(load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
parts, todo, est, tables = [], [], 0.0, {}
for i, (meta, sql, _) in enumerate(sqls):
    if REF is not None and time.process_time() - t0 > LIMIT:
        verdict(ok=False, why=f"query {i}: CPU limit reached (slow patch?)")
    e = pickle.loads(pickle.dumps(parse_one(sql)))
    if not meta.get("execute"):
        parts.append(None)  # the child returns None for it: parsing and pickling is all that matters
        continue
    names = sorted({t.name for t in find_tables(e)})
    duck_sql = transpile(sql, write="duckdb")[0]
    duck = hashlib.sha256(duck_sql.encode()).hexdigest()[:20]
    blob = pickle.dumps(e)  # execute() gets the unpickled tree, before optimize() touches it
    will_run = REF is not None and EXEC_ALL and (EXEC_SET is None or i in EXEC_SET)
    # execute() optimizes again and its rows are compared: no need to compare the tree first
    opt = None if will_run else digest(optimize(e, ensure_schema(TPCDS_SCHEMA), leave_tables_isolated=True,
                                                dialect=None))
    parts.append([names, duck, opt])
    if REF is None:
        todo.append((i, blob, names, duck_sql))
        continue
    ref = REF["parts"][i]
    if ref is None:
        verdict(ok=False, why=f"query {i}: not an executed query before")
    if duck != ref[1]:
        # the reference SQL changed: the test passes as before if duckdb still returns the same
        if not REF.get("dres"):
            verdict(ok=False, why=f"query {i}: duckdb reference SQL changed")
        est += REF["dsec"][str(i)] + (0.5 if not _DUCK else 0.0)
        if est > BUDGET:
            verdict(ok=False, why=f"query {i}: duckdb reference SQL changed (no budget to run it)")
        (ds, dl), _ = duck_digest(duck_sql)
        duck_changed = ds != REF["dres"][str(i)][0] and dl != REF["dres"][str(i)][1]
    else:
        duck_changed = False
    if will_run or duck_changed or names != ref[0] or opt != ref[2]:
        # changed tree or tables, or changed execution code: execute it now (as the test's pool
        # child does) and compare rows
        est += REF["sec"][str(i)]
        if est > BUDGET:
            verdict(ok=False, why=f"query {i}: too many changed queries to execute")
        for n in names:
            if n not in tables:
                t1 = time.process_time()
                tables[n] = load_table(n)
                est += time.process_time() - t1
        if est > BUDGET:
            verdict(ok=False, why=f"query {i}: loading its tables spent the budget")
        (strict, loose), took, table = run(pickle.loads(blob), names, tables)
        # a patch may make a query slower but still correct (an optimization switched off): charge
        # the real time to the budget instead of giving up on any single slow query
        est += max(0.0, took - REF["sec"][str(i)])
        if est > BUDGET:
            verdict(ok=False, why=f"query {i}: execution needs more than the budget")
        if duck_changed or (strict != REF["res"][str(i)] and loose != REF["lres"][str(i)]):
            # rows or reference differ from pristine: decide exactly as the test does
            if not REF.get("dres"):
                verdict(ok=False, why=f"query {i}: different rows")
            if not duck_changed and REF.get("shape") and                     [list(table.columns), len(table.rows)] != REF["shape"][str(i)]:
                # pristine rows pass the test, so another shape fails assert_frame_equal for sure:
                # no need to load pandas to find out
                verdict(ok=False, why=f"query {i}: different rows (other shape)")
            est += (0.0 if "pandas" in sys.modules else 1.2) + REF["dsec"][str(i)] + (0.5 if not _DUCK else 0.0)
            if est > BUDGET:
                verdict(ok=False, why=f"query {i}: different rows (no budget for the exact check)")
            if not test_assert(duck_sql, table):
                verdict(ok=False, why=f"query {i}: different rows")
        todo.append((i, blob, names))
if REF is None:
    for n in sorted({n for _, _, ns, _ in todo for n in ns}):
        tables[n] = load_table(n)
    res, lres, sec, dres, dsec, shape = {}, {}, {}, {}, {}, {}
    for i, blob, names, duck_sql in todo:
        (res[str(i)], lres[str(i)]), sec[str(i)], r = run(pickle.loads(blob), names, tables)
        shape[str(i)] = [list(r.columns), len(r.rows)]
    try:
        for i, blob, names, duck_sql in todo:
            dres[str(i)], dsec[str(i)] = duck_digest(duck_sql)
            dsec[str(i)] = round(dsec[str(i)], 3)
    except Exception as e:  # no duckdb: a changed reference SQL keeps the test, as before
        print("no duckdb reference:", repr(e), file=sys.stderr)
        dres, dsec = None, None
    print("REF " + json.dumps({"parts": parts, "res": res, "lres": lres, "dres": dres, "dsec": dsec, "shape": shape,
                               "sec": {k: round(v, 3) for k, v in sec.items()}}), file=sys.stderr)
    verdict(ok=True, ref=True)
verdict(ok=True, executed=[i for i, _, _ in todo])
'''
TPCDS_RUN = r"""
import json, os, sys, importlib.util, importlib._bootstrap_external as be
d = json.load(sys.stdin)
ov = {os.path.realpath(k): v.encode("utf-8") for k, v in d["overlay"].items()}
_orig = be.FileLoader.get_data
def _get_data(self, path):
    rp = os.path.realpath(path)
    if rp.endswith(".pyc"):
        # pristine bytecode is fine, except for a file the overlay replaces: compile that from source
        try:
            src_rp = os.path.realpath(importlib.util.source_from_cache(path))
        except Exception:
            raise OSError("unknown bytecode: compile from source")
        if src_rp in ov:
            raise OSError("bytecode of a patched file: compile from source")
    return ov[rp] if rp in ov else _orig(self, path)
be.FileLoader.get_data = _get_data
sys.path.insert(0, d["src"])
sys.argv = ["tpcds_check"]
exec(compile(d["code"], "tpcds_check", "exec"), {"__name__": "__main__", "REF": d.get("ref"),
                                                  "BUDGET": d.get("budget", 0.0), "LIMIT": d.get("limit", 1e9),
                                                  "EXEC_ALL": d.get("exec_all", False),
                                                  "EXEC_SET": d.get("exec_set")})
"""
TPCDS_ID = "tests/test_executor.py::TestExecutor::test_execute_tpcds"
TPCDS_MIN_SAVING_MS = 6000  # the check costs about 2-3 s of charged CPU before any execution
TPCDS_MARGIN_S = 4.0  # CPU kept back from the saving for the comparison pass itself
TPCDS_EXEC_CAP_S = 8.0  # most CPU spent executing changed queries for one commit
TPCDS_EXEC_ALL_CAP_S = 19.0  # when the patch changes execution code: up to all 57 queries
TPCDS_EXEC_ALL_KEEP_S = 2.5  # saving kept back for the parse pass of an execution proof
TPCDS_SLACK_S = 8.0  # wall-clock slack on top of margin + budget before a check is killed
# wall clock; the run limit is 180 s (270 s in the final). Override only for big offline eval batches.
TPCDS_BUDGET_S = float(os.environ.get("BR_TPCDS_BUDGET", "120"))


def plan_cost(workers, durations):
    ws = [w for w in workers if w]
    return max(sum(durations[n] for n in w) for w in ws) + WORKER_COST_MS * len(ws) if ws else 0


def tpcds_run(overlay, ref=None, budget=0.0, timeout=240, limit=1e9, exec_all=False, exec_set=None):
    """Run TPCDS_CHECK with `overlay` swapped in. Returns (verdict dict, ref dict or None);
    the verdict is {"ok": False} whenever anything goes wrong."""
    import subprocess
    try:
        p = subprocess.run([sys.executable, "-I", "-B", "-c", TPCDS_RUN], cwd=str(SRC),
                           input=json.dumps({"src": str(SRC), "overlay": overlay, "code": TPCDS_CHECK,
                                             "ref": ref, "budget": budget, "limit": limit,
                                             "exec_all": exec_all, "exec_set": exec_set}),
                           capture_output=True, text=True, timeout=timeout,
                           env={"PYTHONDONTWRITEBYTECODE": "1", "HOME": "/tmp", "PATH": os.environ.get("PATH", "")})
        lines = p.stderr.splitlines()
        v = [json.loads(l[8:]) for l in lines if l.startswith("VERDICT ")]
        r = [json.loads(l[4:]) for l in lines if l.startswith("REF ")]
        if p.returncode != 0 or not v:
            return {"ok": False, "why": "check crashed: " + (lines[-1][:150] if lines else str(p.returncode))}, None
        return v[-1], (r[-1] if r else None)
    except Exception as e:
        log("tpcds check failed:", repr(e))
        return {"ok": False, "why": repr(e)}, None


def _stmt_chain(tree, lo, hi):
    """Statements (outermost first) whose span contains lines lo..hi."""
    chain, body = [], tree.body
    while True:
        for s in body:
            if first_line(s) <= lo and hi <= s.end_lineno:
                chain.append(s)
                break
        else:
            return chain
        s = chain[-1]
        inner = [b for f in BLOCK_FIELDS for b in (getattr(s, f, None) or [])]
        inner = [b.body if isinstance(b, (ast.ExceptHandler, getattr(ast, "match_case", ()))) else [b]
                 for b in inner]
        body = [x for b in inner for x in b]
        if not body:
            return chain


def _bound_names(s):
    out = set()
    if isinstance(s, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        for t in (s.targets if isinstance(s, ast.Assign) else [s.target]):
            for n in ast.walk(t):
                if isinstance(n, ast.Name):
                    out.add(n.id)
                elif isinstance(n, ast.Attribute):
                    out.add(n.attr)
        return out
    return None


def tpcds_region_ok(sel, rel, tree, oa, ob, why):
    """Can this changed region influence test_execute_tpcds only through what the proxy compares?
    The uncompared phase (setUpClass, pickling, ensure_tables/ensure_schema, Plan, PythonExecutor)
    must not run the changed code, nor read a class or name the change rebinds at import."""
    ph = sel.idx["tpcds_phase"]
    U = ph["lines"].get(rel, set())
    lo, hi = (oa, ob) if ob >= oa else (max(ob, 1), oa)
    chain = _stmt_chain(tree, lo, hi)
    if not chain:
        lines = set(range(lo, hi + 1))
    else:
        s = chain[-1]
        body = [b for f in BLOCK_FIELDS for b in (getattr(s, f, None) or [])]
        top = first_line(body[0]) - 1 if body and hi < first_line(body[0]) else s.end_lineno
        lines = set(range(first_line(s), top + 1)) | set(range(lo, hi + 1))
    if lines & U:
        why.append(f"{rel}:{lo} runs in the uncompared phase")
        return False
    il = sel.idx["importlines"].get(rel, ())
    if not any(x in il for x in lines):
        return True  # call-time code that the uncompared phase never runs
    if not chain:
        return False
    s = chain[-1]
    if isinstance(s, DEF_TYPES) and any(first_line(d) <= hi and lo <= d.end_lineno for d in s.decorator_list):
        why.append(f"{rel}:{lo} decorator")
        return False
    if isinstance(s, FUNC_TYPES) and s.body and hi < first_line(s.body[0]):
        # def header (defaults, annotations): only matters when the function is called
        if set(range(first_line(s), s.end_lineno + 1)) & U:
            why.append(f"{rel}:{lo} header of a function the uncompared phase calls")
            return False
        return True
    classes = [c for c in chain if isinstance(c, ast.ClassDef)]
    if classes:
        c = classes[-1]
        key = module_name(rel) + ":" + ".".join(x.name for x in classes)
        if key in ph["classes"] or c.name in ph["names"]:
            why.append(f"{rel}:{lo} class {c.name} is used in the uncompared phase")
            return False
        return True
    names = _bound_names(s)
    if names is None or rel in ph["lines"] or names & ph["names"]:
        why.append(f"{rel}:{lo} module-level change")
        return False
    return True


def _comment_lines(text):
    """Lines that hold only a comment or nothing (by the tokenizer, so '#' inside a string is not
    a comment); None if the text does not tokenize."""
    import io as _io
    import tokenize
    code = set()
    try:
        for tok in tokenize.generate_tokens(_io.StringIO(text).readline):
            if tok.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                                tokenize.DEDENT, tokenize.ENDMARKER):
                code.update(range(tok.start[0], tok.end[0] + 1))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return None
    return set(range(1, text.count("\n") + 2)) - code


TPCDS_FIXTURE = "tests/fixtures/optimizer/tpc-ds/tpc-ds.sql"


def _tpcds_pairs_used(text):
    """What test_execute_tpcds takes from tpc-ds.sql: (meta, sql) of every pair, parsed exactly as
    tests/helpers.load_sql_fixture_pairs does (its own pristine helper functions run on the text).
    The third field, the expected optimized SQL, is only read by the optimizer's test_tpcds."""
    ns = {"__name__": "tpcds_helpers", "__file__": str(SRC / "tests" / "helpers.py")}
    exec(compile((SRC / "tests" / "helpers.py").read_text(encoding="utf-8"), "helpers", "exec"), ns)
    statements = ns["_filter_comments"](text).split(";")
    out = []
    for i in range(0, len(statements), 2):
        if i + 1 < len(statements):
            sql, meta = ns["_extract_meta"](statements[i].strip())
            out.append((repr(sorted(meta.items())), sql))
    return out


def _same_tree(old, new):
    try:
        return dump(ast.parse(old)) == dump(ast.parse(new))
    except SyntaxError:
        return False


def tpcds_skippable(sel, budget=0.0):
    """True only if every change can reach test_execute_tpcds only through what the proxy compares,
    and the proxy dump of the patched source is identical to the pristine one."""
    base = sel.idx.get("tpcds_base")
    if not base or not sel.idx.get("tpcds_phase"):
        return False
    why = sel.why
    exec_all, exec_set = False, 0
    for f in sel.files:
        rel = f["new"] or f["old"]
        if f["binary"] or not f["old"] or not f["new"] or f["old"] != f["new"]:
            return False
        if rel.startswith("tests/fixtures/") and not rel.startswith("tests/fixtures/optimizer/tpc-ds/"):
            continue  # data the test never opens
        if rel == TPCDS_FIXTURE and f["old"] == f["new"] and not f["binary"]:
            old = sel.read_src(rel)
            if old is not None and "tests/helpers.py" not in sel.changed:
                try:
                    new, _ = apply_hunks(old, f["hunks"])
                    if _tpcds_pairs_used(old) == _tpcds_pairs_used(new):
                        why.append("tpc-ds.sql: only expected outputs changed (test_execute_tpcds reads meta + sql)")
                        continue
                except Exception as e:
                    log("tpc-ds fixture compare failed:", repr(e))
            return False
        if rel.endswith(".py") and rel in sel.patched and rel.startswith(("sqlglot/", "tests/")):
            old = sel.read_src(rel)
            if old is not None and _same_tree(old, sel.patched[rel]):
                continue  # comments / formatting only: no code object changes
        if not (rel.startswith("sqlglot/") and rel.endswith(".py")) or rel not in sel.patched:
            return False
        old = sel.read_src(rel)
        if old is None:
            return False
        tree = ast.parse(old)
        _, regions = apply_hunks(old, f["hunks"])
        oc, nc = _comment_lines(old), _comment_lines(sel.patched[rel])
        for oa, ob, na, nb in regions:
            if oc is not None and nc is not None and all(x in oc for x in range(oa, ob + 1)) \
                    and all(x in nc for x in range(na, nb + 1)):
                continue  # only comment / blank lines removed and added here
            if not tpcds_region_ok(sel, rel, tree, oa, ob, why):
                # the change can reach code the test runs outside the compared calls: prove it by
                # executing the queries for real (all of them, or those whose pipeline runs it)
                exec_all = True
                lo, hi = (oa, ob) if ob >= oa else (max(ob, 1), oa)
                lines = set(range(lo, hi + 1))
                chain = _stmt_chain(tree, lo, hi)
                if chain:
                    # the whole innermost statement: coverage marks a statement on the lines that hold
                    # its bytecode, which need not include an edited continuation line
                    lines |= set(range(first_line(chain[-1]), chain[-1].end_lineno + 1))
                il = set(sel.idx["importlines"].get(rel, ()))
                xc = sel.idx.get("tpcds_xcov")
                if xc is None or lines & il or not any(isinstance(s, FUNC_TYPES) for s in chain):
                    exec_set = None  # import-time change, or no map: execute every query
                elif exec_set is not None:
                    for x in lines:
                        exec_set |= xc.get(rel, {}).get(x, 0)
    # a patch can make the optimizer very slow: cap the check's wall clock (a timeout keeps the test)
    if exec_all:
        qs = None if exec_set is None else [i for i in range(exec_set.bit_length()) if exec_set >> i & 1]
        est = sum(v for k, v in sel.idx["tpcds_ref"]["sec"].items() if qs is None or int(k) in qs)
        # pay at most the saving minus the comparison pass and a 2 s net gain
        # break-even: the parse pass (~2.5 s) plus execution must stay under the test's own time
        budget = min(TPCDS_EXEC_ALL_CAP_S, getattr(sel, "tpcds_saving_s", 0.0) - TPCDS_EXEC_ALL_KEEP_S)
        if est > budget:
            why.append(f"tpcds check: executing {len(qs) if qs is not None else 'all'} queries would cost "
                       f"{est:.1f} s > {budget:.1f} s")
            return False
        why.append(f"tpcds check: execute {'all' if qs is None else len(qs)} queries")
    else:
        qs = None
    v, _ = tpcds_run({str(SRC / rel): text for rel, text in sel.patched.items()}, sel.idx["tpcds_ref"],
                     budget, timeout=TPCDS_MARGIN_S + max(budget, 0) + TPCDS_SLACK_S,
                     limit=TPCDS_MARGIN_S + max(budget, 0), exec_all=exec_all, exec_set=qs)
    why.append("tpcds check: " + (f"same rows, executed {v.get('executed')}" if v.get("ok") else v.get("why", "?"))
               + f" ({v.get('cpu')} s)")
    return bool(v.get("ok"))


# Runtime verification of small selections (v5.7). The rules allow running sqlglot's tests while
# deciding (the CPU is charged). VERIFY_RUN runs the selected tests, in suite order, against the
# patched source, the way pytest would judge them: a failure, error, failed subtest, unexpected
# success or unimportable test module fails a test; skips pass. It stops at the first failure.
VERIFY_RUN = r"""
import contextlib, importlib, importlib.util, importlib._bootstrap_external as be, io, json, os, sys, unittest
d = json.load(sys.stdin)
ov = {os.path.realpath(k): v.encode("utf-8") for k, v in d["overlay"].items()}
_orig = be.FileLoader.get_data
def _get_data(self, path):
    rp = os.path.realpath(path)
    if rp.endswith(".pyc"):
        try:
            src_rp = os.path.realpath(importlib.util.source_from_cache(path))
        except Exception:
            raise OSError("unknown bytecode: compile from source")
        if src_rp in ov:
            raise OSError("bytecode of a patched file: compile from source")
    return ov[rp] if rp in ov else _orig(self, path)
be.FileLoader.get_data = _get_data
os.chdir(d["src"])
sys.path.insert(0, d["src"])
os.environ["SKIP_INTEGRATION"] = "0"

class Result(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.bad = None
    def _fail(self, test, kind):
        if self.bad is None:
            self.bad = (getattr(test, "id", lambda: str(test))(), kind)
        self.stop()
    def addError(self, test, err): self._fail(test, "error")
    def addFailure(self, test, err): self._fail(test, "failure")
    def addUnexpectedSuccess(self, test): self._fail(test, "unexpected success")
    def addSubTest(self, test, subtest, err):
        if err is not None:
            self._fail(test, "subtest")

out = {"ok": False}
sink = io.StringIO()
try:
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        suite = unittest.TestSuite()
        for tid in d["ids"]:
            path, cls, meth = tid.split("::")
            mod = importlib.import_module(path[:-3].replace("/", "."))
            suite.addTest(getattr(mod, cls)(meth))
        res = Result()
        suite.run(res)
    out = {"ok": res.bad is None, "bad": res.bad, "ran": res.testsRun}
except BaseException as e:
    out = {"ok": False, "bad": ["import", repr(e)[:200]]}
print("VERIFY " + json.dumps(out))
"""


def verify_run(overlay, ids, timeout=30):
    """Run test ids (file::Class::method, suite order) on the patched source; {"ok": bool, ...}."""
    import subprocess
    try:
        p = subprocess.run([sys.executable, "-I", "-B", "-c", VERIFY_RUN], cwd=str(SRC),
                           input=json.dumps({"src": str(SRC), "overlay": overlay, "ids": ids}),
                           capture_output=True, text=True, timeout=timeout,
                           env={"PYTHONDONTWRITEBYTECODE": "1", "HOME": "/tmp", "PATH": os.environ.get("PATH", ""),
                                "SKIP_INTEGRATION": "0"})
        v = [json.loads(l[7:]) for l in p.stdout.splitlines() if l.startswith("VERIFY ")]
        return v[-1] if v and p.returncode == 0 else {"ok": False, "bad": ["crash", p.stderr[-200:]]}
    except Exception as e:
        return {"ok": False, "bad": ["runner", repr(e)]}


# Warm verifier (v5.7). A cold interpreter costs ~0.6 s CPU per check (sqlglot + every dialect come
# in with tests.dialects.test_dialect), more than a check is worth (~0.43 s expected saving). So one
# warm process imports sqlglot and every test module that needs no third-party package once, the
# way pytest's collection does, and forks a child per commit. The child swaps the patched code of
# the changed functions in (func.__code__), runs the selected tests in suite order, and reports
# through a pipe. Only patches whose every change sits inside the body of a plain top-level function
# or method, in lines that never run at import, are handled this way; anything else is not verified.
ZYGOTE = r"""
import contextlib, io, json, os, sys, types, unittest
src = sys.argv[1]
os.chdir(src)
sys.path.insert(0, src)
os.environ["SKIP_INTEGRATION"] = "0"
mods = json.loads(sys.argv[2])
sink = io.StringIO()
with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
    import importlib
    loaded = {}
    for m in mods:
        try:
            loaded[m] = importlib.import_module(m)
        except BaseException:
            pass
print("READY", flush=True)


class Result(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.bad = None
    def _fail(self, test, kind):
        if self.bad is None:
            self.bad = (getattr(test, "id", lambda: str(test))(), kind)
        self.stop()
    def addError(self, test, err): self._fail(test, "error")
    def addFailure(self, test, err): self._fail(test, "failure")
    def addUnexpectedSuccess(self, test): self._fail(test, "unexpected success")
    def addSubTest(self, test, subtest, err):
        if err is not None:
            self._fail(test, "subtest")


def find_code(code, qualname):
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            if c.co_qualname == qualname:
                return c
            r = find_code(c, qualname)
            if r is not None:
                return r
    return None


def live_function(module, qualname):
    obj = module
    parts = qualname.split(".")
    for p in parts[:-1]:
        obj = obj.__dict__.get(p) if isinstance(obj, (type, types.ModuleType)) else None
        if not isinstance(obj, type):
            return None
    f = obj.__dict__.get(parts[-1]) if isinstance(obj, (type, types.ModuleType)) else None
    if isinstance(f, (staticmethod, classmethod)):
        f = f.__func__
    if not isinstance(f, types.FunctionType) or f.__code__.co_qualname != qualname:
        return None
    return f


def install_overlay(overlay):
    import importlib.util, importlib._bootstrap_external as be
    ov = {os.path.realpath(k): v.encode("utf-8") for k, v in overlay.items()}
    orig = be.FileLoader.get_data
    def get_data(self, path):
        rp = os.path.realpath(path)
        if rp.endswith(".pyc"):
            try:
                src_rp = os.path.realpath(importlib.util.source_from_cache(path))
            except Exception:
                raise OSError("unknown bytecode")
            if src_rp in ov:
                raise OSError("bytecode of a patched file")
        return ov[rp] if rp in ov else orig(self, path)
    be.FileLoader.get_data = get_data


def child(req):
    if req.get("reload") is not None:
        # import-time change in a leaf module: re-execute it and every module that imports it, then
        # the test modules, in their original import order, with the patched sources swapped in
        install_overlay(req["overlay"])
        want = set(req["reload"]) | {i.split("::")[0][:-3].replace("/", ".") for i in req["ids"]}
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            for name in [m for m in list(sys.modules) if m in want]:
                importlib.reload(sys.modules[name])
    for ch in req.get("changes", ()):
        mod = sys.modules.get(ch["module"])
        if mod is None:
            return {"ok": False, "bad": ["swap", "module not loaded " + ch["module"]]}
        code = compile(ch["source"], mod.__file__, "exec")
        for q in ch["functions"]:
            new, f = find_code(code, q), live_function(mod, q)
            if new is None or f is None or new.co_freevars != f.__code__.co_freevars:
                return {"ok": False, "bad": ["swap", q]}
            f.__code__ = new
    suite = unittest.TestSuite()
    for tid in req["ids"]:
        path, cls, meth = tid.split("::")
        m = sys.modules.get(path[:-3].replace("/", "."))
        if m is None:
            return {"ok": False, "bad": ["not loaded", path]}
        suite.addTest(getattr(m, cls)(meth))
    res = Result()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        suite.run(res)
    return {"ok": res.bad is None, "bad": res.bad, "ran": res.testsRun}


for line in sys.stdin:
    req = json.loads(line)
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            out = child(req)
        except BaseException as e:
            out = {"ok": False, "bad": ["child", repr(e)[:200]]}
        os.write(w, json.dumps(out).encode())
        os._exit(0)
    os.close(w)
    data = b""
    while True:
        b = os.read(r, 65536)
        if not b:
            break
        data += b
    os.close(r)
    os.waitpid(pid, 0)
    print("VERIFY " + (data.decode() or json.dumps({"ok": False, "bad": ["child died"]})), flush=True)
"""
VERIFY_MAX_COST_MS = 6000  # only selections this cheap are worth verifying (~2 s saved when all pass)
VERIFY_HEAVY = ("tests/test_optimizer.py", "tests/test_executor.py")  # third-party deps, shared pools
VERIFY_BUDGET_S = float(os.environ.get("BR_VERIFY_BUDGET", "140"))  # wall clock since start
# Off by default: on the practice board v5.7 verified 1 commit clean (-2.0 s simulated) for +2.4 s CPU
# (zygote 0.6 s, ~0.4 s per check; only ~13-20% of checks end in "all pass"). BR_VERIFY=1 turns it on.
VERIFY_ON = os.environ.get("BR_VERIFY", "1") == "1"


class Zygote:
    """The warm verifier process; started at the first commit that needs it."""

    def __init__(self, test_files):
        import subprocess
        mods = [f[:-3].replace("/", ".") for f in test_files if f not in VERIFY_HEAVY]
        self.p = subprocess.Popen([sys.executable, "-I", "-c", ZYGOTE, str(SRC), json.dumps(mods)],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  text=True, cwd=str(SRC),
                                  env={"PYTHONDONTWRITEBYTECODE": "1", "HOME": "/tmp", "SKIP_INTEGRATION": "0",
                                       "PATH": os.environ.get("PATH", "")})
        self.ok = self._readline(60) == "READY"

    def _readline(self, timeout):
        import select
        r, _, _ = select.select([self.p.stdout], [], [], timeout)
        return self.p.stdout.readline().strip() if r else None

    def check(self, changes, ids, timeout=20, reload=None, overlay=None):
        if not self.ok:
            return {"ok": False, "bad": ["zygote down"]}
        try:
            req = {"changes": changes or [], "ids": ids}
            if reload is not None:
                req.update(reload=reload, overlay=overlay)
            self.p.stdin.write(json.dumps(req) + "\n")
            self.p.stdin.flush()
            line = self._readline(timeout)
        except Exception as e:
            line, self.ok = None, False
            log("zygote:", repr(e))
        if not line or not line.startswith("VERIFY "):
            self.ok = False  # a hung child keeps the pipe busy: stop using the zygote
            self.close()
            return {"ok": False, "bad": ["timeout"]}
        return json.loads(line[7:])

    def close(self):
        try:
            self.p.kill()
        except Exception:
            pass


RELOAD_MAX_MODULES = 40  # a leaf module and its importers (subclassing dialects); core modules have ~240


def reload_plan(sel):
    """(modules to re-execute, overlay) for a patch of existing sqlglot/*.py files whose modules and
    their transitive importers are few (leaf modules such as one dialect's tokenizer, parser or
    generator); None otherwise."""
    changed, overlay = set(), {}
    for f in sel.files:
        rel = f["new"] or f["old"]
        if f["binary"] or not f["old"] or f["old"] != f["new"] or not rel.startswith("sqlglot/") \
                or not rel.endswith(".py") or rel not in sel.patched:
            return None
        changed.add(module_name(rel))
        overlay[str(SRC / rel)] = sel.patched[rel]
    closure, stack = set(), list(changed)
    while stack:
        m = stack.pop()
        if m in closure:
            continue
        closure.add(m)
        if len(closure) > RELOAD_MAX_MODULES:
            return None
        stack.extend(sel.idx["rdeps"].get(m, ()))
    return sorted(closure), overlay


def swappable_changes(sel):
    """[{module, source, functions}] when every change of the patch is inside the body of a plain
    top-level function or method (not nested, not a header or decorator) in lines that never run
    at import; None otherwise."""
    out = []
    for f in sel.files:
        rel = f["new"] or f["old"]
        if f["binary"] or not f["old"] or f["old"] != f["new"] or not rel.startswith("sqlglot/") \
                or not rel.endswith(".py") or rel not in sel.patched:
            return None
        old = sel.read_src(rel)
        tree = ast.parse(old)
        il = set(sel.idx["importlines"].get(rel, ()))
        _, regions = apply_hunks(old, f["hunks"])
        names = set()
        for oa, ob, _na, _nb in regions:
            lo, hi = (oa, ob) if ob >= oa else (max(ob, 1), oa)
            if any(x in il for x in range(lo, hi + 1)):
                return None
            chain = _stmt_chain(tree, lo, hi)
            funcs = [i for i, s in enumerate(chain) if isinstance(s, FUNC_TYPES)]
            if len(funcs) != 1 or any(not isinstance(s, ast.ClassDef) for s in chain[:funcs[0]]):
                return None  # not in a function, or in a nested one / one inside control flow
            fn = chain[funcs[0]]
            if not fn.body or lo < first_line(fn.body[0]) or fn.decorator_list and \
                    any(first_line(d) <= hi and lo <= d.end_lineno for d in fn.decorator_list):
                return None  # header, decorator or docstring position
            if any(isinstance(n, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                   and first_line(n) <= hi and lo <= n.end_lineno for n in ast.walk(fn) if n is not fn):
                return None  # touches a nested def, lambda or class
            names.add(".".join(s.name for s in chain[:funcs[0] + 1]))
        out.append({"module": module_name(rel), "source": sel.patched[rel], "functions": sorted(names)})
    return out or None


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
        log("index:", len(idx["units"]), "units", len(idx["refs"]), "names", len(idx["core"]), "core modules")
        return
    started = time.monotonic()
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
        known = set(map_tests) - set(idx["missing"])
        always = [t["number"] for t in tests if t["id"] not in known]
        to_number = [number_of.get(t) for t in map_tests]
    except Exception as e:
        log("selector unavailable, running everything:", repr(e))
    def numbers(mask):
        nums = []
        i = 0
        while mask:
            if mask & 1 and to_number[i] is not None:
                nums.append(to_number[i])
            mask >>= 1
            i += 1
        return nums

    zyg = None
    out = []
    for patch in patches:
        chosen = everything
        tiers = []
        if sel is not None:
            try:
                text = patch.read_text(encoding="utf-8", errors="replace")
                mask, why = sel.select(text)
                chosen = sorted(set(always) | set(numbers(mask)))
                tp = number_of.get(TPCDS_ID)
                if tp is not None and tp in chosen and tp not in always:
                    rest = [n for n in chosen if n != tp]
                    saving = plan_cost(pack(chosen, durations), durations) - plan_cost(pack(rest, durations), durations)
                    try:
                        if saving < TPCDS_MIN_SAVING_MS:
                            why.append(f"tpcds proxy not worth it (saves {saving} ms)")
                        elif time.monotonic() - started > TPCDS_BUDGET_S:
                            why.append("tpcds proxy: time budget spent")
                        elif setattr(sel, "tpcds_saving_s", saving / 1000) or \
                                tpcds_skippable(sel, min(TPCDS_EXEC_CAP_S, saving / 1000 - TPCDS_MARGIN_S)):
                            chosen = [n for n in chosen if n != tp]
                            why.append("tpcds proxy: identical, test_execute_tpcds left out")
                    except Exception as e:
                        log(patch.stem, "tpcds check error:", repr(e))
                if VERIFY_ON and chosen and time.monotonic() - started < VERIFY_BUDGET_S \
                        and plan_cost(pack(chosen, durations), durations) < VERIFY_MAX_COST_MS \
                        and not any(tests[n]["file"] in VERIFY_HEAVY for n in chosen):
                    try:
                        changes = swappable_changes(sel)
                        plan = None if changes else reload_plan(sel)
                        if changes or plan:
                            if zyg is None:
                                zyg = Zygote(sorted({t["file"] for t in tests}))
                            ids = [tests[n]["id"] for n in sorted(chosen)]
                            v = zyg.check(changes, ids) if changes else zyg.check(None, ids, reload=plan[0],
                                                                                  overlay=plan[1])
                            why.append("verify: " + ("all pass, nothing to run" if v.get("ok")
                                                     else str(v.get("bad"))[:80]))
                            if v.get("ok"):
                                chosen = []
                    except Exception as e:
                        log(patch.stem, "verify error:", repr(e))
                try:
                    tiers = [numbers(m) for m in getattr(sel, "plausible", ())] + [everything]
                except Exception as e:
                    log(patch.stem, "free riders off:", repr(e))
                log(patch.stem, len(chosen), "tests;", "; ".join(why)[:300])
            except Everything as e:
                log(patch.stem, "everything:", e)
            except Exception as e:
                log(patch.stem, "error, everything:", repr(e))
        try:
            workers = pack_free(chosen, tiers, durations)
        except Exception as e:
            log(patch.stem, "free riders failed:", repr(e))
            workers = pack(chosen, durations)
        out.append(json.dumps({"commit": patch.stem, "workers": workers}, separators=(",", ":")))
    if zyg is not None:
        zyg.close()
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
