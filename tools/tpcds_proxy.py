"""Prototype: the parts of test_execute_tpcds that the test's outcome depends on, minus execution.

For every TPC-DS fixture query, in the test's own order: parse_one (parent), the pickle round trip
to the pool child, transpile to duckdb (parent's reference result), and the optimize() call inside
executor.execute(). Prints one hash over a full structural dump of all of them.
usage: PYTHONPATH=<sqlglot checkout> python tools/tpcds_proxy.py
"""
import hashlib
import pickle
import sys
import time

t0 = time.process_time()
from sqlglot import exp, find_tables, parse_one, transpile  # noqa: E402
from sqlglot.optimizer import optimize  # noqa: E402
from sqlglot.schema import ensure_schema  # noqa: E402
from tests.helpers import TPCDS_SCHEMA, load_sql_fixture_pairs  # noqa: E402

SKIP = {"parent", "_hash"}


def dump(x, out, seen=None):
    """Exact structural dump. Shared or cyclic objects become back-references (#n), so sharing
    is part of what is compared and nothing is walked twice."""
    if seen is None:
        seen = {}
    if isinstance(x, (exp.Expr, dict, list, set, frozenset)) or hasattr(x, "__dict__"):
        k = id(x)
        if k in seen:
            out.append("#%d" % seen[k][0])
            return
        seen[k] = (len(seen), x)  # keep x alive so its id is not reused
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
        # args insertion order varies run to run in the pinned code itself (id-ordered sets in
        # the optimizer); generation reads args by name, so compare str-keyed dicts sorted
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


h = hashlib.sha256()
sqls = list(load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
schema = ensure_schema(TPCDS_SCHEMA)
n_exec = 0
for meta, sql, _ in sqls:
    out, seen = [], {}
    e = parse_one(sql)
    dump(e, out, seen)
    e = pickle.loads(pickle.dumps(e))
    if meta.get("execute"):
        n_exec += 1
        dump(sorted(t.name for t in find_tables(e)), out, seen)
        out.append(transpile(sql, write="duckdb")[0])
        dump(optimize(e, ensure_schema(TPCDS_SCHEMA), leave_tables_isolated=True, dialect=None), out, seen)
    blob = "\x00".join(out).encode("utf-8", "surrogatepass")
    h.update(blob)
    if len(sys.argv) > 1:
        print(hashlib.sha256(blob).hexdigest()[:12], file=sys.stderr)
        if sys.argv[1] != "1":
            open(f"{sys.argv[1]}.{n_exec}", "wb").write(blob)
print(h.hexdigest(), len(sqls), n_exec, round(time.process_time() - t0, 2), file=sys.stderr)
