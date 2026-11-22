"""Comment-insertion mutants in sqlglot/executor/{context,table,__init__}.py (evaluation only).

usage: python eval/mutate_comment.py <sqlglot_src> <out_dir>
  cmt    one fixed comment line inserted before a statement, at that statement's indent (as
         eval/mutate2.py does: for a decorated def, between the last decorator and `def`)
  cmtx   that insertion plus one call-time edit in the same file (a comparison swapped inside a
         function body), next to the comment (within 3 lines) or far from it
Writes <out>/data/commits/kNNNN.patch, <out>/data/tests.json is copied by the caller.
"""
import ast
import difflib
import json
import random
import re
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
(out / "data" / "commits").mkdir(parents=True, exist_ok=True)
rng = random.Random(5)
COMMENTS = ["# Intentionally not cached", "# Cheap check first", "# Keep in sync with the generator",
            "# Mirrors the behavior of the reference implementation", "# Kept for backwards compatibility"]
SWAP = {ast.Eq: ("==", "!="), ast.NotEq: ("!=", "=="), ast.Is: ("is", "is not"), ast.IsNot: ("is not", "is"),
        ast.Lt: ("<", "<="), ast.Gt: (">", ">=")}
muts = []


def emit(rel, old, new, **info):
    if new == old:
        return
    try:
        ast.parse(new)
    except SyntaxError:
        return
    mid = f"k{len(muts):04d}"
    diff = difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"a/{rel}", f"b/{rel}", n=3)
    (out / "data" / "commits" / f"{mid}.patch").write_text("".join(diff))
    muts.append({"id": mid, "file": rel, **info})


def swaps(text, tree):
    """(line, new_line_text) for single-line comparisons inside function bodies."""
    lines = text.splitlines(True)
    res = []
    for f in ast.walk(tree):
        if not isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for c in ast.walk(ast.Module(f.body, [])):
            if isinstance(c, ast.Compare) and len(c.ops) == 1 and type(c.ops[0]) in SWAP \
                    and c.lineno == c.end_lineno:
                a, b = SWAP[type(c.ops[0])]
                ln = lines[c.lineno - 1]
                seg = ln[c.left.end_col_offset:c.comparators[0].col_offset]
                if seg.strip() == a:
                    new = ln[:c.left.end_col_offset] + seg.replace(a, b, 1) + ln[c.comparators[0].col_offset:]
                    res.append((c.lineno, new))
    return res


for name in ("context.py", "table.py", "__init__.py"):
    path = src / "sqlglot" / "executor" / name
    rel = path.relative_to(src).as_posix()
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines(True)
    sw = swaps(text, tree)
    for s in sorted({s.lineno for s in ast.walk(tree) if isinstance(s, ast.stmt)}):
        ind = re.match(r"[ \t]*", lines[s - 1]).group(0)
        com = f"{ind}{rng.choice(COMMENTS)}\n"
        new = lines[:s - 1] + [com] + lines[s - 1:]
        if ast.dump(ast.parse("".join(new))) != ast.dump(tree):
            continue
        emit(rel, text, "".join(new), kind="cmt", line=s)
        near = [x for x in sw if abs(x[0] - s) <= 3]
        far = [x for x in sw if abs(x[0] - s) > 10]
        for tag, pool in (("near", near), ("far", far)):
            if pool:
                ln, nl = rng.choice(pool)
                both = list(lines)
                both[ln - 1] = nl
                both = both[:s - 1] + [com] + both[s - 1:]
                emit(rel, text, "".join(both), kind="cmtx", line=s, edit=ln, where=tag)
                alone = list(lines)
                alone[ln - 1] = nl
                emit(rel, text, "".join(alone), kind="edit", line=s, edit=ln, where=tag)

json.dump(muts, open(out / "mutants.json", "w"), indent=0)
print(len(muts), "mutants", {k: sum(m["kind"] == k for m in muts) for k in ("cmt", "cmtx")})
