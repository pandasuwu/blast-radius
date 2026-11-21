"""Lines of sqlglot that test_execute_tpcds runs OUTSIDE the calls tools/tpcds_proxy.py compares.

The proxy compares parse_one, transpile(write="duckdb"), find_tables and the optimize() inside
executor.execute(). Everything else the test runs (setUpClass loading, pickling to and from the
pool child, ensure_tables/ensure_schema, Plan, PythonExecutor, result building) is uncompared:
a patch that changes any of those lines cannot be judged by the proxy.

Runs the real test in-process: the spawn pool is swapped for an inline one that keeps the pickle
round trip of the function, its arguments and its result.
usage (br-tools, SKIP_INTEGRATION=0, cwd = sqlglot checkout): python tpcds_phases.py out.json
"""
import json
import os
import pickle
import sys
import unittest
from concurrent.futures import Future

sys.path.insert(0, os.getcwd())
import sqlglot  # noqa: E402
import sqlglot.executor as sx  # noqa: E402
import tests.test_executor as te  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(sqlglot.__file__))) + os.sep
SQLGLOT = os.path.join(ROOT, "sqlglot") + os.sep
depth = 0
uncompared = set()


class InlinePool:
    def __init__(self, *a, initializer=None, initargs=(), **k):
        if initializer:
            initializer(*pickle.loads(pickle.dumps(initargs)))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def submit(self, fn, *args, **kwargs):
        fn, args, kwargs = pickle.loads(pickle.dumps((fn, args, kwargs)))
        f = Future()
        try:
            f.set_result(pickle.loads(pickle.dumps(fn(*args, **kwargs))))
        except BaseException as e:
            f.set_exception(e)
        return f

    def shutdown(self, *a, **k):
        pass


used_classes = set()  # "module:qualname" of every class (and its bases) the uncompared phase uses
_seen_types = set()


def note_type(t):
    if t in _seen_types:
        return
    _seen_types.add(t)
    for c in t.__mro__:
        used_classes.add(f"{c.__module__}:{c.__qualname__}")


def note_tree(x):
    """Expression nodes handed to the uncompared phase: their classes' attributes can be read there."""
    if isinstance(x, sqlglot.exp.Expr):
        for n in x.walk():
            note_type(type(n))
            if n.type is not None:
                note_type(type(n.type))


def compared(fn):
    def wrapper(*a, **k):
        global depth
        depth += 1
        try:
            r = fn(*a, **k)
        finally:
            depth -= 1
        note_tree(r)
        return r
    return wrapper


te.ProcessPoolExecutor = InlinePool
for name in ("parse_one", "transpile", "find_tables"):
    setattr(te, name, compared(getattr(te, name)))
sx.optimize = compared(sx.optimize)

mon = sys.monitoring
TOOL = 3


def on_line(code, line):
    if depth == 0 and code.co_filename.startswith(SQLGLOT):
        uncompared.add((code.co_filename[len(ROOT):].replace(os.sep, "/"), line))
        loc = sys._getframe(1).f_locals
        for name in ("self", "cls"):
            v = loc.get(name)
            if v is not None:
                note_type(v if isinstance(v, type) else type(v))


mon.use_tool_id(TOOL, "tpcds-phases")
mon.register_callback(TOOL, mon.events.LINE, on_line)
mon.set_events(TOOL, mon.events.LINE)
suite = unittest.TestSuite([te.TestExecutor("test_execute_tpcds")])
res = unittest.TextTestRunner(verbosity=1, stream=sys.stderr).run(suite)
mon.set_events(TOOL, 0)
assert res.wasSuccessful(), "TPC-DS must pass on the pinned commit"
out = {}
for rel, line in uncompared:
    out.setdefault(rel, []).append(line)
json.dump({"lines": {k: sorted(v) for k, v in sorted(out.items())}, "classes": sorted(used_classes)},
          open(sys.argv[1], "w"))
print(len(uncompared), "uncompared lines in", len(out), "files;", len(used_classes), "classes", file=sys.stderr)
