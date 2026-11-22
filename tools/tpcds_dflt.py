"""Which sqlglot/executor/ function parameters test_execute_tpcds runs at their default value.

Same pipeline as tools/tpcds_xcov.py (fresh interpreter per executed query: load tables, parse_one,
pickle round trip, find_tables, transpile, executor.execute(), result pickle round trip). On every
call of a function defined in sqlglot/executor/, each parameter that has a default and whose value
at entry `is` that default object is recorded. An explicit argument equal to the default (None,
True, a small int) is also recorded: conservative for the selector, which only uses "never".
usage (cwd = sqlglot checkout): python tpcds_dflt.py out.json
writes {"queries": n, "used": {rel: {def_lineno: {param: hexmask}}}, "called": {rel: {def_lineno: hexmask}}};
bit i = query i in fixture order; def_lineno = the `def` line (co_firstlineno is the first decorator).
"""
import json
import os
import subprocess
import sys

CHILD = r"""
import ast as _ast, csv, gc, gzip, inspect, os, pickle, sys, types
sys.path.insert(0, os.getcwd())
from sqlglot import find_tables, parse_one, transpile
from sqlglot.executor import execute
from sqlglot.executor.table import Table
from tests.helpers import FIXTURES_DIR, TPCDS_SCHEMA, load_sql_fixture_pairs
import sqlglot
root = os.path.dirname(os.path.dirname(os.path.abspath(sqlglot.__file__))) + os.sep
exec_dir = os.path.join(root, "sqlglot", "executor") + os.sep
meta, sql, _ = list(load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))[int(sys.argv[1])]
used, called, funcs = set(), set(), {}

def func_of(code):
    f = funcs.get(code, 0)
    if f == 0:
        f = None
        for r in gc.get_referrers(code):
            if isinstance(r, types.FunctionType) and r.__code__ is code:
                f = r
                break
        funcs[code] = f
    return f

mon = sys.monitoring
def on_start(code, offset):
    if not code.co_filename.startswith(exec_dir):
        return mon.DISABLE
    frame = sys._getframe(1)
    f = func_of(code)
    rel = code.co_filename[len(root):].replace(os.sep, "/")
    key = (rel, code.co_name, code.co_firstlineno)
    called.add(key)
    if f is None:
        return
    pos = code.co_varnames[:code.co_argcount]
    dfl = f.__defaults__ or ()
    pairs = list(zip(pos[len(pos) - len(dfl):], dfl)) + list((f.__kwdefaults__ or {}).items())
    loc = frame.f_locals
    for name, d in pairs:
        if name in loc and loc[name] is d:
            used.add(key + (name,))

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

mon.use_tool_id(4, "dflt")
mon.register_callback(4, mon.events.PY_START, on_start)
mon.set_events(4, mon.events.PY_START)
e = pickle.loads(pickle.dumps(parse_one(sql)))
names = sorted({t.name for t in find_tables(e)})
tables = {n: load_table(n) for n in names}
transpile(sql, write="duckdb")
tabs = pickle.loads(pickle.dumps(tables))
pickle.loads(pickle.dumps(execute(e, schema=TPCDS_SCHEMA, tables=tabs)))
mon.set_events(4, 0)
import json as _j
print(_j.dumps({"called": sorted(called), "used": sorted(used)}))
"""

sys.path.insert(0, os.getcwd())
from tests.helpers import load_sql_fixture_pairs  # noqa: E402


def def_line(rel, name, first):
    """co_firstlineno is the first decorator's line: map it to the `def` line the selector sees."""
    import ast
    tree = ast.parse(open(rel).read())
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            lo = min([d.lineno for d in n.decorator_list] + [n.lineno])
            if lo == first or n.lineno == first:
                return n.lineno
    return first


pairs = list(load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
used, called = {}, {}
for i, (meta, _, _) in enumerate(pairs):
    if not meta.get("execute"):
        continue
    p = subprocess.run([sys.executable, "-I", "-B", "-c", CHILD, str(i)], capture_output=True, text=True,
                       cwd=os.getcwd(), check=True)
    r = json.loads(p.stdout.strip().splitlines()[-1])
    for rel, name, first in r["called"]:
        d = called.setdefault(rel, {})
        k = str(def_line(rel, name, first))
        d[k] = d.get(k, 0) | (1 << i)
    for rel, name, first, param in r["used"]:
        d = used.setdefault(rel, {}).setdefault(str(def_line(rel, name, first)), {})
        d[param] = d.get(param, 0) | (1 << i)
json.dump({"queries": len(pairs),
           "used": {r: {l: {p: hex(m) for p, m in ps.items()} for l, ps in d.items()} for r, d in used.items()},
           "called": {r: {l: hex(m) for l, m in d.items()} for r, d in called.items()}},
          open(sys.argv[1], "w"), indent=0, sort_keys=True)
print(len(pairs), "queries,", sum(len(d) for d in called.values()), "functions called,",
      sum(len(ps) for d in used.values() for ps in d.values()), "defaults used", file=sys.stderr)
