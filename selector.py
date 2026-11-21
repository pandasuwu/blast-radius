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
import contextlib, importlib, importlib._bootstrap_external as be, io, json, os, sys
d = json.load(sys.stdin)
ov = {os.path.realpath(k): v.encode("utf-8") for k, v in d["overlay"].items()}
_orig = be.FileLoader.get_data
def _get_data(self, path):
    rp = os.path.realpath(path)
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


PROBE_DROP = set()


def PROBE(chosen, durations):
    """PROBE C (practice measurement only): never run the huge TPC-DS test(s)."""
    return [n for n in chosen if n not in PROBE_DROP]


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
            if n in taken or n in PROBE_DROP:
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
    tests = json.loads((DATA / "tests.json").read_text())["tests"]
    durations = [0] * len(tests)
    number_of = {}
    for t in tests:
        durations[t["number"]] = round(t["seconds"] * 1000)
        number_of[t["id"]] = t["number"]
    everything = list(range(len(tests)))
    PROBE_DROP.update(t["number"] for t in tests if "tpcds" in t["id"] and durations[t["number"]] > 3000)
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

    out = []
    for patch in patches:
        chosen = everything
        tiers = []
        if sel is not None:
            try:
                text = patch.read_text(encoding="utf-8", errors="replace")
                mask, why = sel.select(text)
                chosen = sorted(set(always) | set(numbers(mask)))
                try:
                    tiers = [numbers(m) for m in getattr(sel, "plausible", ())] + [everything]
                except Exception as e:
                    log(patch.stem, "free riders off:", repr(e))
                log(patch.stem, len(chosen), "tests;", "; ".join(why)[:300])
            except Everything as e:
                log(patch.stem, "everything:", e)
            except Exception as e:
                log(patch.stem, "error, everything:", repr(e))
        chosen = PROBE(chosen, durations)
        tiers = [PROBE(t, durations) for t in tiers]
        try:
            workers = pack_free(chosen, tiers, durations)
        except Exception as e:
            log(patch.stem, "free riders failed:", repr(e))
            workers = pack(chosen, durations)
        out.append(json.dumps({"commit": patch.stem, "workers": workers}, separators=(",", ":")))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
