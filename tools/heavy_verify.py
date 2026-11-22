"""Verification of the "heavy" tests (tests/test_optimizer.py, tests/test_executor.py) in a warm process.

Written to be pasted into selector.py's ZYGOTE (stdlib only, no selector imports). Two entry points:

  heavy_prepare()                 zygote start, BEFORE the fork loop: imports the heavy test modules
                                  (and with them duckdb / pandas / numpy) with their process pools
                                  replaced by an inline pool. Pristine code only; no class setup.
  heavy_run(ids, stop_first)      in the forked child, AFTER the overlay / code swaps / reloads are in
                                  place: runs the tests in the order given (suite order), each class's
                                  setUpClass once, and returns {"results": {id: "pass"|"fail"}, ...}.

How a test is judged (as pytest would): failure, error, failed subtest or unexpected success = fail;
skip = pass. A setUpClass / module fixture error fails every test of that class. Tests not run
because stop_first stopped early are reported as "not run".

Why inline is faithful here (see work/HANDOFF.md "heavy-verify" for the measurements):
  * TestOptimizer.check_file and TestExecutor._mp_execute submit picklable module-level functions to a
    spawn ProcessPoolExecutor. The inline pool runs the initializer once, then each call in-process
    with the same pickle round trips (arguments in, result out) the real pool makes, so a result that
    does not survive pickling fails the same way.
  * Spawn workers re-import the (patched) sources; the forked child already has the patched code.
    Module caches: the warm child's caches are warm, a spawn worker's are cold; tests compare results,
    not cache state, and _clear_caches() empties sqlglot's module-level dict/set caches before each
    test as tools/profplugin.py does.
  * Not modelled: a worker that dies hard (segfault, os._exit) breaks the real pool for every later
    test of the class (BrokenProcessPool). Inline, that kills the child, which the zygote already
    treats as "keep everything" (no verdict), so it can only cost time, never a miss.
"""
import contextlib
import importlib
import io
import os
import sys
import unittest

HEAVY_FILES = ("tests/test_optimizer.py", "tests/test_executor.py")
HEAVY_SKIP = ("tests/test_executor.py::TestExecutor::test_execute_tpcds",)  # the TPC-DS proof's job


class InlineProcessPool:
    """concurrent.futures.ProcessPoolExecutor stand-in: same initializer and pickling, no processes."""

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
                    self._init(*pickle.loads(pickle.dumps(self._initargs)))
            fn, args, kwargs = pickle.loads(pickle.dumps((fn, args, kwargs)))
            f.set_result(pickle.loads(pickle.dumps(fn(*args, **kwargs))))
        except BaseException as e:  # noqa: BLE001 - the real pool hands every exception to result()
            f.set_exception(e)
        return f

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        return iter([self.submit(fn, *a).result() for a in zip(*iterables)])

    def shutdown(self, wait=True, *, cancel_futures=False):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
        return False


def _clear_caches():
    for name, mod in list(sys.modules.items()):
        if not name.startswith("sqlglot") or mod is None:
            continue
        for attr, value in list(vars(mod).items()):
            if "cache" in attr.lower() and isinstance(value, (dict, set)):
                value.clear()


def _inline_pools(mod):
    if getattr(mod, "ProcessPoolExecutor", None) is not None:
        mod.ProcessPoolExecutor = InlineProcessPool


def install_text_overlay(overlay):
    """Serve patched non-.py files (tests/fixtures/optimizer/*.sql, tpc-h.sql, ...) through open().
    The zygote's install_overlay only patches the import loader; the heavy tests read their fixtures
    with open() (tests.helpers.load_sql_fixture_pairs), so without this a fixture patch is verified
    against the pristine fixture and passes wrongly. Call it in the child, before heavy_run."""
    import builtins
    text = {os.path.realpath(k): v for k, v in overlay.items() if not k.endswith(".py")}
    if not text:
        return
    _open = builtins.open

    def _ov_open(file, mode="r", *a, **k):
        if isinstance(file, (str, os.PathLike)) and set(mode) <= set("rt"):
            rp = os.path.realpath(file)
            if rp in text:
                return io.StringIO(text[rp])
        return _open(file, mode, *a, **k)

    builtins.open = _ov_open


def heavy_prepare():
    """Import the heavy test modules once (pristine), pools inline. Returns the module names loaded."""
    import concurrent.futures
    import concurrent.futures.process
    concurrent.futures.ProcessPoolExecutor = InlineProcessPool
    concurrent.futures.process.ProcessPoolExecutor = InlineProcessPool
    loaded = []
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        for f in HEAVY_FILES:
            name = f[:-3].replace("/", ".")
            try:
                _inline_pools(importlib.import_module(name))
                loaded.append(name)
            except BaseException:  # noqa: BLE001 - the child re-imports and reports it
                pass
    return loaded


class _Result(unittest.TestResult):
    def __init__(self, stop_first):
        super().__init__()
        self.stop_first = stop_first
        self.bad = {}  # unittest id -> kind of the first failure
        self.started = set()

    def startTest(self, test):
        _clear_caches()  # before setUp: sqlglot caches start empty for every test, as in profplugin
        super().startTest(test)
        self.started.add(test.id())

    def _fail(self, test, kind):
        self.bad.setdefault(getattr(test, "id", lambda: str(test))(), kind)
        if self.stop_first:
            self.stop()

    def addError(self, test, err):
        self._fail(test, "error")

    def addFailure(self, test, err):
        self._fail(test, "failure")

    def addUnexpectedSuccess(self, test):
        self._fail(test, "unexpected success")

    def addSubTest(self, test, subtest, err):
        if err is not None:
            self._fail(test, "subtest")


def heavy_run(ids, stop_first=False):
    """Run heavy test ids ("path::Class::method") in the given order in this process.
    -> {"results": {id: "pass"|"fail"|"not run"}, "first": [id, kind] | None, "error": str | None}"""
    out = {"results": {}, "first": None, "error": None}
    suite = unittest.TestSuite()
    uid = {}
    for tid in ids:
        path, cls, meth = tid.split("::")
        name = path[:-3].replace("/", ".")
        try:
            m = sys.modules.get(name) or importlib.import_module(name)
        except BaseException as e:  # noqa: BLE001 - an unimportable test module fails its tests
            out["results"][tid] = "fail"
            out["first"] = out["first"] or [tid, "import " + repr(e)[:120]]
            continue
        _inline_pools(m)
        t = getattr(m, cls)(meth)
        uid[t.id()] = tid
        suite.addTest(t)

    res = _Result(stop_first)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        suite.run(res)
    # class / module fixture errors are reported under a _ErrorHolder id: they fail the whole class
    fixture_bad = [k for k in res.bad if k not in uid]
    for u, tid in uid.items():
        cls_key = u.rsplit(".", 1)[0]
        if u in res.bad or any(cls_key in k for k in fixture_bad):
            out["results"][tid] = "fail"
        elif u in res.started:
            out["results"][tid] = "pass"
        else:
            out["results"][tid] = "not run" if fixture_bad == [] else "fail"
    if res.bad and out["first"] is None:
        k = next(iter(res.bad))
        out["first"] = [uid.get(k, k), res.bad[k]]
    return out
