"""Replay of test_execute_tpcds without duckdb (v5.5).

The test passes iff, for every TPC-DS fixture query marked `execute`, sqlglot's python executor
returns the same frame as duckdb running transpile(sql, write="duckdb"). On the pinned commit it
passes. So if, on the patched source, every step the test's outcome depends on gives exactly the
pristine value (table setup data, parse_one and its pickle round trip, find_tables, the duckdb
SQL, and the executor's result for each query), the test passes on the patched source too.

Steps whose pristine value is recorded and compared:
  setup   the TPC-H and TPC-DS Table objects built as setUpClass builds them, after a pickle
          round trip (the pool's initargs)
  per query: parse_one -> pickle round trip (exact structural dump), sorted find_tables names,
          transpile(write="duckdb"), the optimize() call executor.execute makes (exact dump),
          and, when needed, execute() -> pickle round trip -> (columns, repr(rows)).
A query is executed when the caller says so (its execution path reaches changed code) or when
its optimized tree differs from the pristine one. A query whose tree is identical and whose
execution path does not reach changed code returns the pristine result.

stdin JSON: {"src", "overlay": {path: text}, "mode": "record"|"check", "expect": record,
             "need": [query index...] | "all", "cap": cpu seconds}
stdout: one JSON line. record mode: the record. check mode: {"ok": bool, "why": str, "cpu": s}
"""
import ast
import csv
import gzip
import hashlib
import importlib._bootstrap_external as be
import importlib.util
import json
import os
import pickle
import sys
import time

d = json.load(sys.stdin)
ov = {os.path.realpath(k): v.encode("utf-8") for k, v in d.get("overlay", {}).items()}
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
sys.path.insert(0, d["src"])
t0 = time.process_time()
MODE = d.get("mode", "check")
EXPECT = d.get("expect") or {}
NEED = d.get("need", "all")
CAP = float(d.get("cap", 1e9))
if MODE == "check" and CAP < 1e6:
    # hard stop: a patched query can run far longer than its pristine time (the cap is only checked
    # between queries). The kernel ends the process at the limit; no output means "keep the test".
    try:
        import resource

        lim = int(CAP + 3) + 1  # runaway guard only; the cap itself is checked between queries
        resource.setrlimit(resource.RLIMIT_CPU, (lim, lim + 1))
    except Exception:
        pass


def finish(ok, why):
    print(json.dumps({"ok": ok, "why": why, "cpu": round(time.process_time() - t0, 3)}))
    sys.stdout.flush()
    os._exit(0)


try:
    from sqlglot import exp, find_tables, parse_one, transpile  # noqa: E402
    from sqlglot.executor import execute  # noqa: E402
    from sqlglot.executor.table import Table  # noqa: E402
    from sqlglot.optimizer import optimize  # noqa: E402
    from sqlglot.schema import ensure_schema  # noqa: E402
    from tests.helpers import FIXTURES_DIR, TPCDS_SCHEMA, TPCH_SCHEMA, load_sql_fixture_pairs  # noqa: E402
except BaseException as e:
    if MODE == "record":
        raise
    finish(False, "import: " + type(e).__name__)

SKIP = {"parent", "_hash"}


def dump(x, out, seen):
    """Exact structural dump (as tools/tpcds_proxy.py)."""
    if isinstance(x, (exp.Expr, dict, list, set, frozenset)) or hasattr(x, "__dict__"):
        k = id(x)
        if k in seen:
            out.append("#%d" % seen[k][0])
            return
        seen[k] = (len(seen), x)
    if isinstance(x, exp.Expr):
        out.append("<" + type(x).__qualname__)
        slots = [s for c in type(x).__mro__ for s in getattr(c, "__slots__", ())]
        items = [(s, getattr(x, s, None)) for s in slots] + sorted(getattr(x, "__dict__", {}).items())
        for k, v in items:
            if k in SKIP:
                continue
            out.append(k + "=")
            dump(v, out, seen)
        out.append(">")
    elif isinstance(x, dict):
        out.append("{")
        items = sorted(x.items()) if all(type(k) is str for k in x) else x.items()
        for k, v in items:
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


def hexdump(x):
    out = []
    dump(x, out, {})
    return hashlib.sha256("\x00".join(out).encode("utf-8", "surrogatepass")).hexdigest()[:20]


def hstr(s):
    return hashlib.sha256(s.encode("utf-8", "surrogatepass")).hexdigest()[:20]


def table_key(t):
    return hstr(repr((type(t).__qualname__, list(t.columns), [tuple(r) for r in t.rows])))


def open_file(file_name):
    with open(file_name, "rb") as f:
        gzipped = f.read(2) == b"\x1f\x8b"
    if gzipped:
        return gzip.open(file_name, "rt", newline="")
    return open(file_name, encoding="utf-8", newline="")


def load_tables(directory, schema):
    """As TestExecutor.setUpClass (minus the duckdb views)."""
    tables = {}
    for table, columns in schema.items():
        reader = csv.reader(open_file(f"{directory}{table}.csv.gz"), delimiter="|")
        rows, ctypes = [], []
        next(reader)
        for row in reader:
            if not ctypes:
                for v in row:
                    try:
                        ctypes.append(type(ast.literal_eval(v)))
                    except (ValueError, SyntaxError):
                        ctypes.append(str)
            rows.append(tuple(None if (t is not str and v == "") else t(v) for t, v in zip(ctypes, row)))
        tables[table] = Table(columns=columns, rows=rows)
    return tables


rec = {"queries": {}}
try:
    list(load_sql_fixture_pairs("optimizer/tpc-h/tpc-h.sql"))
    sqls = list(load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
except BaseException as e:
    if MODE == "record":
        raise
    finish(False, "fixtures: " + type(e).__name__)
_setup = {}


def setup():
    """setUpClass's tables (built lazily: a failing check usually stops before it needs them)."""
    if not _setup:
        try:
            D = os.path.join(FIXTURES_DIR, "optimizer") + os.sep
            tpch = load_tables(D + "tpc-h" + os.sep, TPCH_SCHEMA)
            tpcds = load_tables(D + "tpc-ds" + os.sep, TPCDS_SCHEMA)
            # the pool's initializer gets (schema, tables) pickled
            schema_c, tables_c = pickle.loads(pickle.dumps((TPCDS_SCHEMA, tpcds)))
            pickle.loads(pickle.dumps((TPCH_SCHEMA, tpch)))
            key = hstr(repr((sorted(schema_c.items()), [(k, table_key(v)) for k, v in sorted(tables_c.items())],
                             [(k, table_key(v)) for k, v in sorted(tpch.items())], len(sqls))))
        except BaseException as e:
            if MODE == "record":
                raise
            finish(False, "setup: " + type(e).__name__)
        if MODE == "check" and key != EXPECT.get("setup"):
            finish(False, "setup differs")
        rec["setup"] = key
        _setup.update(schema=schema_c, tables=tables_c)
    return _setup["schema"], _setup["tables"]

need = None if NEED == "all" else set(NEED)
for i, (meta, sql, _) in enumerate(sqls):
    q = {}
    eq = EXPECT.get("queries", {}).get(str(i), {})
    try:
        e = pickle.loads(pickle.dumps(parse_one(sql)))
        q["parse"] = hexdump(e)
        if MODE == "check" and q["parse"] != eq.get("parse"):
            finish(False, f"q{i} parse differs")
        if meta.get("execute"):
            names = sorted(t.name for t in find_tables(e))
            q["tables"] = hstr(repr(names))
            q["duckdb"] = hstr(transpile(sql, write="duckdb")[0])
            q["opt"] = hexdump(optimize(e, ensure_schema(TPCDS_SCHEMA), leave_tables_isolated=True, dialect=None))
            if MODE == "check":
                for k in ("tables", "duckdb"):
                    if q[k] != eq.get(k):
                        finish(False, f"q{i} {k} differs")
            run = MODE == "record" or need is None or i in need or q["opt"] != eq.get("opt")
            if run:
                if MODE == "check" and time.process_time() - t0 + eq.get("cpu", 0) > CAP:
                    finish(False, f"q{i} over the cpu cap")
                schema_c, tables_c = setup()
                a = time.process_time()
                sub = {}
                for n in names:
                    sub[n] = tables_c[n]
                r = pickle.loads(pickle.dumps(execute(e, schema=schema_c, tables=sub)))
                q["cpu"] = round(time.process_time() - a, 3)
                q["result"] = hstr(repr((list(r.columns), [tuple(x) for x in r.rows])))
                if MODE == "check" and q["result"] != eq.get("result"):
                    finish(False, f"q{i} result differs")
    except SystemExit:
        raise
    except BaseException as ex:
        if MODE == "record":
            raise
        finish(False, f"q{i} {type(ex).__name__}")
    rec["queries"][str(i)] = q
setup()  # the tables must match even when no query had to be executed
if MODE == "record":
    rec["cpu"] = round(time.process_time() - t0, 3)
    print(json.dumps(rec))
else:
    finish(True, "identical")
