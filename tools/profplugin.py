"""pytest plugin for the condition-profile run (tools/run_prof.sh; never in the image).

sqlglot modules are compiled from an instrumented AST: inside function bodies every
single-operator comparison becomes _BRP_.cmp(nid, op, left, right) and every and/or becomes
_BRP_.boolop(nid, is_and, thunk, thunk, ...). The probe returns exactly what the original
expression returns, and records per test:
  cmp:    the outcome of each sibling operator (==, !=, is, is not / <, <=, >, >=, ==, !=)
  boolop: for each operand, '1'/'0' truthiness or '-' if not evaluated
so the selector can tell whether a mutated operator changes anything for a given test.
"""
import ast
import builtins
import importlib._bootstrap_external as bext
import json
import operator
import os
import sys

import pytest

ROOT = os.path.realpath(os.getcwd())
PKG = ROOT + "/sqlglot/"
OUT = os.environ.get("MAP_OUT", "/out")
CUR = [""]
ALT = [0]           # >0 while evaluating an alternative operator (record nothing)
NODES = []          # nid -> [rel, lineno, col, end_lineno, end_col, kind, op]
PROF = {}           # nid -> {sig: set(ctx)}
OUTCOMES = {}

OPS = {"Eq": operator.eq, "NotEq": operator.ne, "Is": operator.is_, "IsNot": operator.is_not,
       "Lt": operator.lt, "LtE": operator.le, "Gt": operator.gt, "GtE": operator.ge,
       "In": lambda a, b: a in b, "NotIn": lambda a, b: a not in b}
FAMILY = {}
for f in (("Eq", "NotEq", "Is", "IsNot"), ("Lt", "LtE", "Gt", "GtE", "Eq", "NotEq")):
    for o in f[:4]:
        FAMILY.setdefault(o, f)
FAMILY["In"] = ("In",)  # alternatives could consume iterators: record the result only
FAMILY["NotIn"] = ("NotIn",)


def _rec(nid, sig):
    d = PROF.get(nid)
    if d is None:
        d = PROF[nid] = {}
    s = d.get(sig)
    if s is None:
        s = d[sig] = set()
    s.add(CUR[0])


class Probe:
    __slots__ = ()

    @staticmethod
    def cmp(nid, op, left, right):
        if ALT[0]:  # inside an alternative evaluation: behave like the plain operator
            return OPS[op](left, right)
        try:
            result = OPS[op](left, right)
        except BaseException:
            _rec(nid, "X")
            raise
        sig = []
        for alt in FAMILY[op]:
            if alt == op:
                v = result
            else:
                ALT[0] += 1
                try:
                    v = OPS[alt](left, right)
                except Exception:
                    sig.append("E")
                    continue
                finally:
                    ALT[0] -= 1
            # only real bools are comparable; e.g. Expression builders return objects
            sig.append(("1" if v else "0") if type(v) is bool else "E")
        _rec(nid, "".join(sig))
        return result

    @staticmethod
    def boolop(nid, is_and, *thunks):
        if ALT[0]:
            v = None
            for th in thunks:
                v = th()
                if (is_and and not v) or (not is_and and v):
                    return v
            return v
        vec = []
        v = None
        n = len(thunks)
        for i, th in enumerate(thunks):
            try:
                v = th()
            except BaseException:
                vec.append("X")
                vec.extend("-" * (n - i - 1))
                _rec(nid, "".join(vec))
                raise
            try:
                t = bool(v)
            except BaseException:
                if i == n - 1:  # Python never calls bool() on the last operand
                    vec.append("E")
                    _rec(nid, "".join(vec))
                    return v
                vec.append("X")
                vec.extend("-" * (n - i - 1))
                _rec(nid, "".join(vec))
                raise
            vec.append("1" if t else "0")
            if (is_and and not t) or (not is_and and t):
                vec.extend("-" * (n - i - 1))
                break
        _rec(nid, "".join(vec))
        return v


_DEFKEY = {}


def _class_key(c):
    return f"{c.__module__}:{c.__qualname__}"


def _defkey(obj, name):
    """Where getattr(obj, name) found its value: 'module:Class' of the defining class,
    'inst' for an instance/module dict, '?' when unknown (super(), __getattr__, ...)."""
    if isinstance(obj, type):
        k = _DEFKEY.get((obj, name))
        if k is None:
            k = "?"
            for c in obj.__mro__:
                if name in c.__dict__:
                    k = _class_key(c)
                    break
            _DEFKEY[(obj, name)] = k
        return k
    if type(obj) is super:
        return "?"
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict) and name in d:
        return "inst"
    t = type(obj)
    k = _DEFKEY.get((t, name))
    if k is None:
        k = "?"
        for c in t.__mro__:
            if name in c.__dict__:
                k = _class_key(c)
                break
        _DEFKEY[(t, name)] = k
    return k


def _import_stmt():
    """At import/collection time: (rel, line) of the innermost module- or class-body frame
    in the repo, i.e. the top-level statement whose evaluation is doing this read."""
    f = sys._getframe(3)
    while f is not None:
        co = f.f_code
        if not (co.co_flags & 0x1) and co.co_filename.startswith(ROOT + "/"):
            return f"{os.path.relpath(co.co_filename, ROOT)}:{f.f_lineno}"
        f = f.f_back
    return ""


def _ga(nid, obj, name):
    v = getattr(obj, name)
    if not ALT[0]:
        try:
            k = _defkey(obj, name)
        except Exception:
            k = "?"
        if not CUR[0]:
            k = k + "@" + _import_stmt()
        _rec(nid, k)
    return v


Probe.ga = staticmethod(_ga)


class _Sentinel:
    __slots__ = ()

    def __repr__(self):
        return "<BRP default>"


def _dflt(nid, value):
    if not ALT[0]:
        _rec(nid, "1")
    return value


Probe.SENT = _Sentinel()
Probe.dflt = staticmethod(_dflt)


def _key_sig(k):
    """How the selector names a dict key from source: constants by repr, enum members and
    classes by name (a source key `exp.Foo` / `TokenType.X` maps by its last name)."""
    import enum
    if isinstance(k, enum.Enum):
        return f"E:{type(k).__name__}.{k.name}"
    if isinstance(k, type):
        return f"T:{k.__name__}"
    if isinstance(k, (str, int, float, bool, bytes, type(None))):
        return f"C:{k!r}"
    return "*"


class TDict(dict):
    """A module/class-level dict literal that records, per test, which keys' values are read.
    Reads that see every value (iteration with values, copies, merges, compares) record '*'."""
    __slots__ = ("_nid",)

    def _r(self, sig):
        if not ALT[0]:
            _rec(self._nid, sig)

    def __getitem__(self, k):
        try:
            self._r(_key_sig(k))
        except TypeError:
            self._r("*")
        return dict.__getitem__(self, k)

    def get(self, k, default=None):
        try:
            self._r(_key_sig(k))
        except TypeError:
            self._r("*")
        return dict.get(self, k, default)

    def pop(self, k, *a):
        self._r("*")
        return dict.pop(self, k, *a)

    def setdefault(self, k, default=None):
        self._r("*")
        return dict.setdefault(self, k, default)

    def __iter__(self):  # keys only; overriding it sends C-level merges through keys()+getitem
        return dict.__iter__(self)

    def items(self):
        self._r("*")
        return dict.items(self)

    def values(self):
        self._r("*")
        return dict.values(self)

    def copy(self):
        self._r("*")
        return dict.copy(self)

    def __eq__(self, o):
        self._r("*")
        return dict.__eq__(self, o)

    def __ne__(self, o):
        self._r("*")
        return dict.__ne__(self, o)

    __hash__ = None

    def __or__(self, o):
        self._r("*")
        return dict.__or__(self, o)

    def __ror__(self, o):
        self._r("*")
        return dict.__ror__(self, o)

    def __repr__(self):
        self._r("*")
        return dict.__repr__(self)

    def __reduce_ex__(self, proto):
        self._r("*")
        return (dict, (dict(dict.items(self)),))

    def popitem(self):
        self._r("*")
        return dict.popitem(self)

    def update(self, *a, **kw):
        self._r("*")
        return dict.update(self, *a, **kw)


def _tdict(nid, d):
    t = TDict(d)
    t._nid = nid
    return t


Probe.tdict = staticmethod(_tdict)
builtins._BRP_ = Probe()


def _attr_names():
    """Names assigned in class bodies of sqlglot (excluding Enum classes' members)."""
    trees = []
    for dirpath, _, files in os.walk(PKG):
        for fn in files:
            if fn.endswith(".py"):
                try:
                    trees.append(ast.parse(open(os.path.join(dirpath, fn), encoding="utf-8").read()))
                except SyntaxError:
                    pass
    enums = {"Enum", "IntEnum", "StrEnum", "Flag", "IntFlag", "AutoName"}
    changed = True
    while changed:
        changed = False
        for t in trees:
            for n in ast.walk(t):
                if isinstance(n, ast.ClassDef) and n.name not in enums:
                    for b in n.bases:
                        bn = b.attr if isinstance(b, ast.Attribute) else getattr(b, "id", None)
                        if bn in enums:
                            enums.add(n.name)
                            changed = True
                            break
    names = set()
    for t in trees:
        for n in ast.walk(t):
            if isinstance(n, ast.ClassDef) and n.name not in enums:
                for s in n.body:
                    targets = s.targets if isinstance(s, ast.Assign) else [s.target] if isinstance(s, (ast.AnnAssign, ast.AugAssign)) else []
                    for tg in targets:
                        for x in ast.walk(tg):
                            if isinstance(x, ast.Name):
                                names.add(x.id)
    return names


ATTRNAMES = _attr_names()

BAD = (ast.NamedExpr, ast.Yield, ast.YieldFrom, ast.Await)


def _unsafe_for_thunk(node):
    for n in ast.walk(node):
        if isinstance(n, BAD):
            return True
        if isinstance(n, ast.Name) and n.id in ("super", "__class__"):
            return True
    return False


class Instrument(ast.NodeTransformer):
    def __init__(self, rel):
        self.rel = rel
        self.depth = 0  # >0 inside a function or lambda body

    def _nid(self, node, kind, op):
        NODES.append([self.rel, node.lineno, node.col_offset, node.end_lineno, node.end_col_offset, kind, op])
        return len(NODES) - 1

    def visit_FunctionDef(self, node):
        # decorators/defaults/annotations run at definition time: not instrumented
        self.depth += 1
        node.body = [self.visit(s) for s in node.body]
        self.depth -= 1
        self._defaults(node)
        return node

    def _defaults(self, node):
        """Constant defaults become a sentinel; the body starts by recording that the call left
        the parameter out ("dflt" node at the default's position) and restoring the constant."""
        a = node.args
        pos = a.posonlyargs + a.args
        pairs = list(zip(pos[len(pos) - len(a.defaults):], range(len(a.defaults)), ["d"] * len(a.defaults)))
        pairs += [(p, i, "k") for i, p in enumerate(a.kwonlyargs) if a.kw_defaults[i] is not None]
        checks = []
        for p, i, kind in pairs:
            d = a.defaults[i] if kind == "d" else a.kw_defaults[i]
            if not isinstance(d, ast.Constant):
                continue
            nid = self._nid(d, "dflt", p.arg)
            sent = ast.Attribute(value=ast.Name("_BRP_", ast.Load()), attr="SENT", ctx=ast.Load())
            if kind == "d":
                a.defaults[i] = ast.copy_location(sent, d)
            else:
                a.kw_defaults[i] = ast.copy_location(sent, d)
            test = ast.Compare(left=ast.Name(p.arg, ast.Load()), ops=[ast.Is()],
                               comparators=[ast.Attribute(value=ast.Name("_BRP_", ast.Load()), attr="SENT",
                                                          ctx=ast.Load())])
            rec = ast.Call(func=ast.Attribute(value=ast.Name("_BRP_", ast.Load()), attr="dflt", ctx=ast.Load()),
                           args=[ast.Constant(nid), ast.Constant(d.value)], keywords=[])
            checks.append(ast.If(test=test, body=[ast.Assign(targets=[ast.Name(p.arg, ast.Store())], value=rec)],
                                 orelse=[]))
        if checks:
            body = node.body
            k = 1 if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str) else 0
            for c in checks:
                ast.copy_location(c, body[k] if k < len(body) else node)
                for x in ast.walk(c):
                    ast.copy_location(x, c)
            node.body = body[:k] + checks + body[k:]

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        self.depth += 1
        node.body = self.visit(node.body)
        self.depth -= 1
        return node

    def visit_ClassDef(self, node):
        saved, self.depth = self.depth, 0  # class body runs at definition time
        node.body = [self.visit(s) for s in node.body]
        self.depth = saved
        return node

    def _wrap_dict(self, node):
        if self.depth != 0 or not isinstance(node.value, ast.Dict):
            return node
        d = node.value
        nid = self._nid(d, "dict", "")
        node.value = ast.copy_location(ast.Call(
            func=ast.Attribute(value=ast.Name("_BRP_", ast.Load()), attr="tdict", ctx=ast.Load()),
            args=[ast.Constant(nid), d], keywords=[]), d)
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        return self._wrap_dict(node)

    def visit_AnnAssign(self, node):
        self.generic_visit(node)
        return self._wrap_dict(node) if node.value is not None else node

    def visit_Compare(self, node):
        self.generic_visit(node)
        if self.depth == 0 or len(node.ops) != 1:
            return node
        op = type(node.ops[0]).__name__
        if op not in OPS:
            return node
        nid = self._nid(node, "cmp", op)
        call = ast.Call(
            func=ast.Attribute(value=ast.Name("_BRP_", ast.Load()), attr="cmp", ctx=ast.Load()),
            args=[ast.Constant(nid), ast.Constant(op), node.left, node.comparators[0]], keywords=[])
        return ast.copy_location(call, node)

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if self.depth == 0 or not isinstance(node.ctx, ast.Load) or node.attr not in ATTRNAMES:
            return node
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "super":
            return node  # super().X: keep zero-arg super() in place
        nid = self._nid(node, "attr", node.attr)
        call = ast.Call(
            func=ast.Attribute(value=ast.Name("_BRP_", ast.Load()), attr="ga", ctx=ast.Load()),
            args=[ast.Constant(nid), node.value, ast.Constant(node.attr)], keywords=[])
        return ast.copy_location(call, node)

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if self.depth == 0 or any(_unsafe_for_thunk(v) for v in node.values):
            return node
        nid = self._nid(node, "bool", type(node.op).__name__)
        thunks = [ast.copy_location(ast.Lambda(args=ast.arguments(posonlyargs=[], args=[], vararg=None,
                                                                  kwonlyargs=[], kw_defaults=[], kwarg=None,
                                                                  defaults=[]), body=v), v)
                  for v in node.values]
        call = ast.Call(
            func=ast.Attribute(value=ast.Name("_BRP_", ast.Load()), attr="boolop", ctx=ast.Load()),
            args=[ast.Constant(nid), ast.Constant(isinstance(node.op, ast.And))] + thunks, keywords=[])
        return ast.copy_location(call, node)


_orig_stc = bext.SourceLoader.source_to_code


def source_to_code(self, data, path, *, _optimize=-1):
    if isinstance(path, str) and os.path.realpath(path).startswith(PKG):
        rel = os.path.relpath(os.path.realpath(path), ROOT)
        tree = ast.parse(data, path)
        tree = Instrument(rel).visit(tree)
        ast.fix_missing_locations(tree)
        return compile(tree, path, "exec", dont_inherit=True, optimize=_optimize)
    return _orig_stc(self, data, path, _optimize=_optimize)


bext.SourceLoader.source_to_code = source_to_code


class InlineProcessPool:
    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **kw):
        self._init, self._initargs, self._ready = initializer, initargs, False

    def submit(self, fn, /, *args, **kwargs):
        import pickle
        from concurrent.futures import Future
        f = Future()
        try:
            if not self._ready:
                self._ready = True
                if self._init is not None:
                    self._init(*self._initargs)
            fn, args, kwargs = pickle.loads(pickle.dumps((fn, args, kwargs)))
            f.set_result(pickle.loads(pickle.dumps(fn(*args, **kwargs))))
        except BaseException as e:  # noqa: BLE001
            f.set_exception(e)
        return f

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        return [self.submit(fn, *a).result() for a in zip(*iterables)]

    def shutdown(self, wait=True, *, cancel_futures=False):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _clear_caches():
    for name, mod in list(sys.modules.items()):
        if not name.startswith("sqlglot") or mod is None:
            continue
        for attr, value in list(vars(mod).items()):
            if "cache" in attr.lower() and isinstance(value, (dict, set)):
                value.clear()


def pytest_configure(config):
    import concurrent.futures
    import concurrent.futures.process
    import importlib
    import pkgutil
    concurrent.futures.ProcessPoolExecutor = InlineProcessPool
    concurrent.futures.process.ProcessPoolExecutor = InlineProcessPool
    import sqlglot
    for info in pkgutil.walk_packages(sqlglot.__path__, prefix="sqlglot."):
        if info.name == "sqlglot.__main__":
            continue
        try:
            importlib.import_module(info.name)
        except Exception as e:
            print("preimport failed", info.name, e, file=sys.stderr)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    _clear_caches()
    CUR[0] = "S|" + item.nodeid
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    CUR[0] = "C|" + item.nodeid
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item):
    CUR[0] = "C|" + item.nodeid
    yield
    CUR[0] = ""


def pytest_runtest_logreport(report):
    OUTCOMES.setdefault(report.nodeid, []).append(f"{report.when}:{report.outcome}")


def _identity_decorators():
    """Decorator call sites in sqlglot whose (external) factory returns the decorated object
    unchanged for the original constant arguments and every bool-flip / int+-1 variant."""
    import importlib
    out = []
    for name, mod in list(sys.modules.items()):
        if not name.startswith("sqlglot") or mod is None or not getattr(mod, "__file__", None):
            continue
        path = os.path.realpath(mod.__file__)
        if not path.startswith(PKG):
            continue
        rel = os.path.relpath(path, ROOT)
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except Exception:
            continue
        for node in ast.walk(tree):
            for dec in getattr(node, "decorator_list", []) or []:
                if not isinstance(dec, ast.Call):
                    continue
                consts = list(dec.args) + [k.value for k in dec.keywords]
                if not all(isinstance(c, ast.Constant) for c in consts) or any(k.arg is None for k in dec.keywords):
                    continue
                try:
                    fn = eval(compile(ast.Expression(dec.func), rel, "eval"), vars(mod))
                except Exception:
                    continue
                if getattr(fn, "__module__", "sqlglot").startswith("sqlglot"):
                    continue
                args = [c.value for c in dec.args]
                kws = {k.arg: k.value.value for k in dec.keywords}
                variants = [(list(args), dict(kws))]
                for i, a in enumerate(args):
                    for alt in ((not a,) if isinstance(a, bool) else (a + 1, a - 1) if isinstance(a, int) else ()):
                        variants.append((args[:i] + [alt] + args[i + 1:], dict(kws)))
                for k, a in kws.items():
                    for alt in ((not a,) if isinstance(a, bool) else (a + 1, a - 1) if isinstance(a, int) else ()):
                        variants.append((list(args), {**kws, k: alt}))
                ok = True
                for va, vk in variants:
                    try:
                        d = fn(*va, **vk)

                        class _Probe:
                            pass

                        def _probe():
                            pass

                        if d(_Probe) is not _Probe or d(_probe) is not _probe:
                            ok = False
                    except Exception:
                        ok = False
                    if not ok:
                        break
                if ok:
                    out.append([rel, dec.lineno, dec.col_offset])
    return out


def pytest_sessionfinish(session):
    prof = {str(nid): {sig: sorted(ctxs) for sig, ctxs in d.items()} for nid, d in PROF.items()}
    with open(os.path.join(OUT, "profile.json"), "w") as f:
        json.dump({"nodes": NODES, "prof": prof, "outcomes": OUTCOMES, "attrnames": sorted(ATTRNAMES),
                   "identity_decorators": _identity_decorators()}, f)
