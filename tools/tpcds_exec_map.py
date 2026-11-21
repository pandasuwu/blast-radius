"""Per-query execution map for the TPC-DS replay (v5.5) -> maps/tpcds_exec.json.

For each TPC-DS query marked `execute`, in a fresh interpreter (no state from other queries):
the table setup and parse_one run first, then execute() runs under sys.monitoring. Recorded:
the sqlglot lines it runs outside the optimize() call it makes (optimize's output is compared by
the replay directly), and the classes (with bases) of every self/cls seen on those lines and of
every node of the optimized tree handed to the planner. Setup (table loading and the pickle
round trips of the pool) is recorded the same way under "setup".

usage (python 3.13, cwd = pristine sqlglot checkout): python tpcds_exec_map.py out.json
"""
import json
import os
import pickle
import subprocess
import sys

HERE = os.path.abspath(__file__)


def child(which):
    sys.path.insert(0, os.getcwd())
    import sqlglot
    import sqlglot.executor as sx
    from sqlglot import exp, find_tables, parse_one
    from tests.helpers import TPCDS_SCHEMA, load_sql_fixture_pairs

    sys.argv = ["replay"]
    root = os.path.dirname(os.path.dirname(os.path.abspath(sqlglot.__file__))) + os.sep
    pkg = os.path.join(root, "sqlglot") + os.sep
    lines, classes, seen_t = set(), set(), set()
    state = {"depth": 0}

    def note_type(t):
        if t in seen_t:
            return
        seen_t.add(t)
        for c in t.__mro__:
            classes.add(f"{c.__module__}:{c.__qualname__}")

    def note_tree(x):
        if isinstance(x, exp.Expr):
            for n in x.walk():
                note_type(type(n))
                if n.type is not None:
                    note_type(type(n.type))

    real_opt = sx.optimize

    def opt(*a, **k):
        state["depth"] += 1
        try:
            r = real_opt(*a, **k)
        finally:
            state["depth"] -= 1
        note_tree(r)
        return r

    mon = sys.monitoring

    def on_line(code, line):
        if state["depth"] == 0 and code.co_filename.startswith(pkg):
            lines.add((code.co_filename[len(root):].replace(os.sep, "/"), line))
            loc = sys._getframe(1).f_locals
            for name in ("self", "cls"):
                v = loc.get(name)
                if v is not None:
                    note_type(v if isinstance(v, type) else type(v))

    mon.use_tool_id(3, "tpcds-exec")
    mon.register_callback(3, mon.events.LINE, on_line)

    # the replay's own setup code, imported from the replay script by exec of its helpers
    import ast
    import csv
    import gzip
    from sqlglot.executor.table import Table
    from tests.helpers import FIXTURES_DIR, TPCH_SCHEMA

    def open_file(fn):
        with open(fn, "rb") as f:
            gz = f.read(2) == b"\x1f\x8b"
        return gzip.open(fn, "rt", newline="") if gz else open(fn, encoding="utf-8", newline="")

    def load_tables(directory, schema):
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

    D = os.path.join(FIXTURES_DIR, "optimizer") + os.sep
    if which == "setup":
        mon.set_events(3, mon.events.LINE)
    tpch = load_tables(D + "tpc-h" + os.sep, TPCH_SCHEMA)
    tpcds = load_tables(D + "tpc-ds" + os.sep, TPCDS_SCHEMA)
    schema_c, tables_c = pickle.loads(pickle.dumps((TPCDS_SCHEMA, tpcds)))
    pickle.loads(pickle.dumps((TPCH_SCHEMA, tpch)))
    mon.set_events(3, 0)
    if which == "setup":
        return {"lines": sorted(lines), "classes": sorted(classes)}
    sqls = list(load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
    meta, sql, _ = sqls[int(which)]
    e = pickle.loads(pickle.dumps(parse_one(sql)))
    names = sorted(t.name for t in find_tables(e))
    sub = {n: tables_c[n] for n in names}
    sx.optimize = opt
    mon.set_events(3, mon.events.LINE)
    r = pickle.loads(pickle.dumps(sx.execute(e, schema=schema_c, tables=sub)))
    mon.set_events(3, 0)
    assert r.rows is not None
    return {"lines": sorted(lines), "classes": sorted(classes)}


def main():
    out = sys.argv[1]
    sys.path.insert(0, os.getcwd())
    from tests.helpers import load_sql_fixture_pairs

    sqls = list(load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
    jobs = ["setup"] + [str(i) for i, (m, _, _) in enumerate(sqls) if m.get("execute")]
    res = {}
    for j in jobs:
        p = subprocess.run([sys.executable, "-I", "-B", HERE, "--child", j], cwd=os.getcwd(),
                           capture_output=True, text=True, check=True)
        res[j] = json.loads(p.stdout.strip().splitlines()[-1])
        print(j, len(res[j]["lines"]), "lines", len(res[j]["classes"]), "classes", file=sys.stderr)

    def pack(r):
        by = {}
        for rel, ln in r["lines"]:
            by.setdefault(rel, []).append(ln)
        return {"lines": {k: sorted(v) for k, v in sorted(by.items())}, "classes": r["classes"]}

    json.dump({"setup": pack(res.pop("setup")), "queries": {k: pack(v) for k, v in res.items()}},
              open(out, "w"), separators=(",", ":"))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        print(json.dumps(child(sys.argv[2])))
    else:
        main()
