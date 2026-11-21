"""pytest plugin used only by tools/run_maps.sh (never in the submission image).

Per test it records, besides coverage (switched to our own contexts):
  * repo files opened with open/io.open/Path.open/tokenize.open/gzip ("opens")
  * repo files read as code by importlib/runpy ("code_opens")
  * sqlglot/tests classes instantiated (full MRO), via sys.monitoring on __init__/__new__
Contexts: "S|<nodeid>" = setup phase (incl. setUpClass, spread to the whole class later),
          "C|<nodeid>" = call + teardown, "" = collection / pre-import.
"""
import builtins, io, json, os, sys, tokenize, importlib, pkgutil
import pytest

ROOT = os.path.realpath(os.getcwd())
OUT = os.environ.get("MAP_OUT", "/out")
CUR = [""]
OPENS, CODE_OPENS, CLASSES, OUTCOMES = {}, {}, {}, {}
_cov = None


def _ctx():
    c = CUR[0]
    if c:
        return c
    # empty context: tag with the innermost tests/ file on the stack, if any
    f = sys._getframe(2)
    while f is not None:
        fn = f.f_code.co_filename
        if fn.startswith(ROOT + "/tests/"):
            return "|" + os.path.relpath(fn, ROOT)
        f = f.f_back
    return ""


def _log(store, p):
    try:
        if isinstance(p, int):
            return
        p = os.fsdecode(os.fspath(p))
        ap = os.path.realpath(p)
    except Exception:
        return
    if ap.startswith(ROOT + "/"):
        store.setdefault(_ctx(), set()).add(os.path.relpath(ap, ROOT))


_open, _open_code, _tok_open = builtins.open, io.open_code, tokenize._builtin_open


def open_(file, *a, **k):
    _log(OPENS, file)
    return _open(file, *a, **k)


def open_code(path):
    _log(CODE_OPENS, path)
    return _open_code(path)


builtins.open = io.open = tokenize._builtin_open = open_
io.open_code = open_code

# ---- class instantiation via sys.monitoring -------------------------------------
mon = sys.monitoring
TOOL = mon.PROFILER_ID
_mro_cache = {}


def _names(t):
    r = _mro_cache.get(t)
    if r is None:
        r = tuple(
            f"{c.__module__}:{c.__qualname__}"
            for c in getattr(t, "__mro__", ())
            if c.__module__.startswith(("sqlglot", "tests"))
        )
        _mro_cache[t] = r
    return r


def _on_start(code, offset):
    if code.co_name not in ("__init__", "__new__") or not code.co_filename.startswith(ROOT):
        return mon.DISABLE
    if not code.co_varnames:
        return
    try:
        f = sys._getframe(1)
        obj = f.f_locals.get(code.co_varnames[0])
    except Exception:
        return
    t = obj if isinstance(obj, type) and code.co_name == "__new__" else type(obj)
    names = _names(t)
    if names:
        CLASSES.setdefault(CUR[0], set()).update(names)


mon.use_tool_id(TOOL, "mapper")
mon.register_callback(TOOL, mon.events.PY_START, _on_start)
mon.set_events(TOOL, mon.events.PY_START)


def _switch(ctx):
    CUR[0] = ctx
    if _cov is not None:
        _cov.switch_context(ctx)


class InlineProcessPool:
    """Stand-in for ProcessPoolExecutor during mapping: runs work in this process so
    coverage / open log / class log see it under the current test's context."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **kw):
        self._init, self._initargs, self._ready = initializer, initargs, False

    def _ensure_init(self):
        if not self._ready:
            self._ready = True
            if self._init is not None:
                self._init(*self._initargs)

    def submit(self, fn, /, *args, **kwargs):
        from concurrent.futures import Future
        f = Future()
        import pickle
        try:
            self._ensure_init()
            # a real pool pickles work and results (sqlglot expressions go through serde)
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


def pytest_configure(config):
    global _cov
    import concurrent.futures
    import concurrent.futures.process
    concurrent.futures.ProcessPoolExecutor = InlineProcessPool
    concurrent.futures.process.ProcessPoolExecutor = InlineProcessPool
    import coverage
    _cov = coverage.Coverage.current()
    # import every sqlglot module now so import-time code lands in the "" context
    import sqlglot
    for info in pkgutil.walk_packages(sqlglot.__path__, prefix="sqlglot."):
        if info.name == "sqlglot.__main__":
            continue
        try:
            importlib.import_module(info.name)
        except Exception as e:
            print("preimport failed", info.name, e, file=sys.stderr)


def _clear_caches():
    """Module-level memo caches (e.g. generator._DISPATCH_CACHE) would credit cache-building
    code only to the first test that hits it; empty them before every test."""
    for name, mod in list(sys.modules.items()):
        if not name.startswith("sqlglot") or mod is None:
            continue
        for attr, value in list(vars(mod).items()):
            if "cache" in attr.lower() and isinstance(value, (dict, set)):
                value.clear()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    _clear_caches()
    _switch("S|" + item.nodeid)
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    _switch("C|" + item.nodeid)
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item):
    _switch("C|" + item.nodeid)
    yield
    _switch("")


def pytest_runtest_logreport(report):
    o = OUTCOMES.setdefault(report.nodeid, [])
    o.append(f"{report.when}:{report.outcome}")


def pytest_sessionfinish(session):
    mon.set_events(TOOL, 0)
    js = lambda d: {k: sorted(v) for k, v in d.items()}
    with _open(os.path.join(OUT, "runtime.json"), "w") as f:
        json.dump({"opens": js(OPENS), "code_opens": js(CODE_OPENS),
                   "classes": js(CLASSES), "outcomes": OUTCOMES}, f)
