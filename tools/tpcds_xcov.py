"""Per-query coverage of test_execute_tpcds's whole pipeline, execution included.

For every executed TPC-DS fixture query, in a fresh interpreter: load its tables as the test's
setUpClass does, parse_one, the pickle round trip, find_tables, transpile(write="duckdb"),
executor.execute() (ensure_tables, ensure_schema, optimize, Plan, PythonExecutor) and the result's
pickle round trip, recording every sqlglot line run after imports. A fresh process per query
charges lazily built state (dispatch caches...) to every query that needs it.
usage (cwd = sqlglot checkout): python tpcds_xcov.py out.json
writes {"queries": n, "lines": {rel: {line: hexmask}}}; bit i = query i in fixture order.
"""
import json
import os
import subprocess
import sys

CHILD = r"""
import ast as _ast, csv, gzip, os, pickle, sys
sys.path.insert(0, os.getcwd())
from sqlglot import find_tables, parse_one, transpile
from sqlglot.executor import execute
from sqlglot.executor.table import Table
from tests.helpers import FIXTURES_DIR, TPCDS_SCHEMA, load_sql_fixture_pairs
import sqlglot
root = os.path.dirname(os.path.dirname(os.path.abspath(sqlglot.__file__))) + os.sep
sqlglot_dir = os.path.join(root, "sqlglot") + os.sep
meta, sql, _ = list(load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))[int(sys.argv[1])]
hit = set()
mon = sys.monitoring
def on_line(code, line):
    if code.co_filename.startswith(sqlglot_dir):
        hit.add((code.co_filename[len(root):].replace(os.sep, "/"), line))
    return mon.DISABLE
def load_table(name):
    fn = f"{FIXTURES_DIR}/optimizer/tpc-ds/{name}.csv.gz"
    reader = csv.reader(gzip.open(fn, "rt", newline=""), delimiter="|")
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
    return Table(columns=TPCDS_SCHEMA[name], rows=rows)
mon.use_tool_id(3, "xcov")
mon.register_callback(3, mon.events.LINE, on_line)
mon.set_events(3, mon.events.LINE)
e = pickle.loads(pickle.dumps(parse_one(sql)))
names = sorted({t.name for t in find_tables(e)})
tables = {n: load_table(n) for n in names}
transpile(sql, write="duckdb")
tabs = pickle.loads(pickle.dumps(tables))
pickle.loads(pickle.dumps(execute(e, schema=TPCDS_SCHEMA, tables=tabs)))
mon.set_events(3, 0)
print("\n".join(f"{r}:{l}" for r, l in sorted(hit)))
"""

sys.path.insert(0, os.getcwd())
from tests.helpers import load_sql_fixture_pairs  # noqa: E402

pairs = list(load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
lines = {}
for i, (meta, _, _) in enumerate(pairs):
    if not meta.get("execute"):
        continue
    p = subprocess.run([sys.executable, "-I", "-B", "-c", CHILD, str(i)], capture_output=True, text=True,
                       cwd=os.getcwd(), check=True)
    for loc in p.stdout.split():
        rel, _, ln = loc.rpartition(":")
        d = lines.setdefault(rel, {})
        d[ln] = d.get(ln, 0) | (1 << i)
json.dump({"queries": len(pairs), "lines": {r: {l: hex(m) for l, m in d.items()} for r, d in lines.items()}},
          open(sys.argv[1], "w"))
print(len(pairs), "queries,", sum(len(d) for d in lines.values()), "lines", file=sys.stderr)
