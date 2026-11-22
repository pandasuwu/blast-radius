"""Module-level edits in sqlglot/executor/{context,table,__init__}.py (evaluation only).

usage: python eval/mutate_toplevel.py <sqlglot_src> <out_dir>
  tcdrop  one name dropped from an import inside `if t.TYPE_CHECKING:`
  tcswap  the names of such an import reversed
  tcadd   `import typing` added inside the TYPE_CHECKING block
  logname the logger name string changed
  constr  a string constant in a module-level call changed
"""
import ast
import difflib
import json
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
(out / "data" / "commits").mkdir(parents=True, exist_ok=True)
muts = []


def emit(rel, old, new, kind, line):
    if new == old:
        return
    ast.parse(new)
    mid = f"t{len(muts):04d}"
    diff = difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"a/{rel}", f"b/{rel}", n=3)
    (out / "data" / "commits" / f"{mid}.patch").write_text("".join(diff))
    muts.append({"id": mid, "file": rel, "kind": kind, "line": line})


for name in ("context.py", "table.py", "__init__.py"):
    path = src / "sqlglot" / "executor" / name
    rel = path.relative_to(src).as_posix()
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(True)
    tree = ast.parse(text)
    for st in tree.body:
        if isinstance(st, ast.If) and "TYPE_CHECKING" in ast.unparse(st.test):
            ind = " " * st.body[0].col_offset
            for imp in st.body:
                if isinstance(imp, ast.ImportFrom) and imp.lineno == imp.end_lineno:
                    names = [a.name if not a.asname else f"{a.name} as {a.asname}" for a in imp.names]
                    head = f"{ind}from {'.' * imp.level}{imp.module or ''} import "
                    for drop in range(len(names)) if len(names) > 1 else []:
                        new = lines[:imp.lineno - 1] + [head + ", ".join(n for i, n in enumerate(names) if i != drop) + "\n"] + lines[imp.lineno:]
                        emit(rel, text, "".join(new), "tcdrop", imp.lineno)
                    if len(names) > 1:
                        new = lines[:imp.lineno - 1] + [head + ", ".join(reversed(names)) + "\n"] + lines[imp.lineno:]
                        emit(rel, text, "".join(new), "tcswap", imp.lineno)
            last = st.body[-1].end_lineno
            new = lines[:last] + [f"{ind}import typing\n"] + lines[last:]
            emit(rel, text, "".join(new), "tcadd", last + 1)
        elif isinstance(st, ast.Assign):
            for c in ast.walk(st):
                if isinstance(c, ast.Constant) and isinstance(c.value, str) and c.lineno == c.end_lineno:
                    ln = lines[c.lineno - 1]
                    new = ln[:c.col_offset] + repr(c.value + "_x") + ln[c.end_col_offset:]
                    emit(rel, text, "".join(lines[:c.lineno - 1] + [new] + lines[c.lineno:]),
                         "logname" if "getLogger" in ast.unparse(st) else "constr", c.lineno)

json.dump(muts, open(out / "mutants.json", "w"), indent=0)
print(len(muts), "mutants", sorted({m["kind"] for m in muts}))
