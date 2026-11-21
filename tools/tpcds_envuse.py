"""Which executor ENV functions each executed TPC-DS fixture query calls.

The line maps cannot see this: a lambda in the ENV dict sits on a line that also runs at import,
and its body's line events are not recorded. Each ENV callable is wrapped and every executed query
runs (tables loaded as TestExecutor.setUpClass does) in one process; lazily built state does not
change which ENV functions a query's generated code calls.
usage (cwd = sqlglot checkout): python tpcds_envuse.py out.json
writes {"queries": n, "keys": {name: hexmask}}; bit i = query i in fixture order.
"""
import ast as _ast
import csv
import gzip
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import sqlglot.executor.env as envm  # noqa: E402

current, used = [None], {}
for k, f in list(envm.ENV.items()):
    if callable(f):
        def w(*a, _k=k, _f=f, **kw):
            used[_k] = used.get(_k, 0) | (1 << current[0])
            return _f(*a, **kw)
        envm.ENV[k] = w
from sqlglot import parse_one  # noqa: E402
from sqlglot.executor import execute  # noqa: E402
from sqlglot.executor.table import Table  # noqa: E402
from tests.helpers import FIXTURES_DIR, TPCDS_SCHEMA, load_sql_fixture_pairs  # noqa: E402


def load_table(name):
    fn = f"{FIXTURES_DIR}/optimizer/tpc-ds/{name}.csv.gz"
    reader = csv.reader(gzip.open(fn, "rt", newline=""), delimiter="|")
    rows, ct = [], []
    next(reader)
    for row in reader:
        if not ct:
            for v in row:
                try:
                    ct.append(type(_ast.literal_eval(v)))
                except (ValueError, SyntaxError):
                    ct.append(str)
        rows.append(tuple(None if (t is not str and v == "") else t(v) for t, v in zip(ct, row)))
    return Table(columns=TPCDS_SCHEMA[name], rows=rows)


tables = {n: load_table(n) for n in TPCDS_SCHEMA}
pairs = list(load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
for i, (meta, sql, _) in enumerate(pairs):
    if meta.get("execute"):
        current[0] = i
        execute(parse_one(sql), schema=TPCDS_SCHEMA, tables=tables)
json.dump({"queries": len(pairs), "keys": {k: hex(m) for k, m in sorted(used.items())}}, open(sys.argv[1], "w"))
print(len(used), "ENV functions used by", len(pairs), "queries", file=sys.stderr)
