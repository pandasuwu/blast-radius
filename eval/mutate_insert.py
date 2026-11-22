"""Pure-insertion mutants inside executor method bodies (evaluation only).

usage: python eval/mutate_insert.py <sqlglot_src> <out_dir>
  ins_end    `pass` added after the last statement of a function body (lands before the next def)
  ins_start  `pass` added before the first body statement (after the docstring, if any)
  ins_raise  `raise ValueError('mutant')` added before the first body statement
"""
import ast
import difflib
import json
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
(out / "data" / "commits").mkdir(parents=True, exist_ok=True)
muts = []
for name in ("context.py", "table.py", "__init__.py"):
    path = src / "sqlglot" / "executor" / name
    rel = path.relative_to(src).as_posix()
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(True)
    tree = ast.parse(text)
    for f in ast.walk(tree):
        if not isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = [s for s in f.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
        if not body:
            continue
        ind = " " * body[0].col_offset
        for kind, at, stmt in (("ins_start", body[0].lineno - 1, f"{ind}pass\n"),
                               ("ins_end", f.body[-1].end_lineno, f"{ind}pass\n"),
                               ("ins_raise", body[0].lineno - 1, f"{ind}raise ValueError('mutant')\n")):
            new = "".join(lines[:at] + [stmt] + lines[at:])
            ast.parse(new)
            mid = f"i{len(muts):04d}"
            diff = difflib.unified_diff(text.splitlines(True), new.splitlines(True), f"a/{rel}", f"b/{rel}", n=3)
            (out / "data" / "commits" / f"{mid}.patch").write_text("".join(diff))
            muts.append({"id": mid, "file": rel, "kind": kind, "func": f.name, "line": at + 1})
json.dump(muts, open(out / "mutants.json", "w"), indent=0)
print(len(muts), "mutants")
