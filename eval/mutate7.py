"""mutate3, restricted to the bodies of Tokenizer classes in sqlglot/dialects/*.py and sqlglot/tokens.py
(evaluation only; tests the v7.3 forwarded-setting rule).

usage: python eval/mutate7.py <sqlglot_src> <out_dir> <count> [seed]
One token per commit, outside function bodies: int +-1, True<->False, comparison or
and/or swap (e.g. inside class-level comprehensions/lambdas), files weighted by size.
"""
import ast
import difflib
import json
import random
import sys
from pathlib import Path

src, out, count = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
rng = random.Random(int(sys.argv[4]) if len(sys.argv) > 4 else 5)
(out / "commits").mkdir(parents=True, exist_ok=True)
files = sorted(p for p in (src / "sqlglot" / "dialects").glob("*.py") if "class Tokenizer" in p.read_text(encoding="utf-8"))
files.append(src / "sqlglot" / "tokens.py")
weights = [p.stat().st_size for p in files]
SWAP = {ast.Eq: ("==", "!="), ast.NotEq: ("!=", "=="), ast.Lt: ("<", "<="), ast.Gt: (">", ">="),
        ast.LtE: ("<=", "<"), ast.GtE: (">=", ">"), ast.Is: ("is", "=="), ast.IsNot: ("is not", "!=")}


def outside_functions(tree):
    """Nodes inside a `class Tokenizer` body but outside its functions."""
    out = []

    def walk(n, in_func, in_tok):
        for c in ast.iter_child_nodes(n):
            f = in_func or isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef))
            t = in_tok or (isinstance(c, ast.ClassDef) and c.name == "Tokenizer")
            if not f and in_tok:
                out.append(c)
            walk(c, f, t)

    walk(tree, False, False)
    return out


def edit(text, line, col, old, new):
    lines = text.splitlines(True)
    b = lines[line - 1].encode()
    if b[col:col + len(old)].decode(errors="replace") != old:
        return None
    lines[line - 1] = (b[:col] + new.encode() + b[col + len(old):]).decode()
    return "".join(lines)


meta = {}
seen = set()
n = tries = 0
while n < count and tries < count * 100:
    tries += 1
    p = rng.choices(files, weights)[0]
    text = p.read_text(encoding="utf-8")
    nodes = outside_functions(ast.parse(text))
    rng.shuffle(nodes)
    new = desc = None
    for node in nodes[:400]:
        if isinstance(node, ast.Constant) and node.lineno == node.end_lineno:
            seg = ast.get_source_segment(text, node)
            if type(node.value) is bool and seg in ("True", "False"):
                new, desc = edit(text, node.lineno, node.col_offset, seg, str(not node.value)), "bool_const"
            elif type(node.value) is int and seg == str(node.value):
                new, desc = edit(text, node.lineno, node.col_offset, seg, str(node.value + rng.choice((1, -1)))), "int"
        elif isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in SWAP \
                and node.left.end_lineno == node.comparators[0].lineno:
            old, rep = SWAP[type(node.ops[0])]
            new, desc = edit(text, node.left.end_lineno, node.left.end_col_offset, f" {old} ", f" {rep} "), "cmp"
        elif isinstance(node, ast.BoolOp) and node.values[0].end_lineno == node.values[1].lineno:
            a = node.values[0]
            o = " and " if isinstance(node.op, ast.And) else " or "
            new, desc = edit(text, a.end_lineno, a.end_col_offset, o, " or " if o == " and " else " and "), "bool"
        if new:
            try:
                ast.parse(new)
            except SyntaxError:
                new = None
                continue
            break
    if not new or (p, new) in seen:
        continue
    seen.add((p, new))
    rel = p.relative_to(src).as_posix()
    patch = "".join(difflib.unified_diff(text.splitlines(True), new.splitlines(True), f"a/{rel}", f"b/{rel}"))
    mid = f"c{n:04d}"
    (out / "commits" / f"{mid}.patch").write_text(patch, encoding="utf-8")
    meta[mid] = {"kind": "code", "file": rel, "desc": desc}
    n += 1
(out / "mutants.json").write_text(json.dumps(meta, indent=1))
print("mutants", n, file=sys.stderr)
