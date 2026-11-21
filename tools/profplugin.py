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
                    v = bool(v)
                except Exception:
                    sig.append("E")
                    continue
                finally:
                    ALT[0] -= 1
            try:
                sig.append("1" if v else "0")
            except Exception:
                sig.append("E")
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


builtins._BRP_ = Probe()

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
        return node

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


def pytest_sessionfinish(session):
    prof = {str(nid): {sig: sorted(ctxs) for sig, ctxs in d.items()} for nid, d in PROF.items()}
    with open(os.path.join(OUT, "profile.json"), "w") as f:
        json.dump({"nodes": NODES, "prof": prof, "outcomes": OUTCOMES}, f)
