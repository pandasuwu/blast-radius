"""Synthetic commits in the style of the sample set (evaluation only, never shipped).

usage: python eval/mutate2.py <sqlglot_src> <out_dir> <count> [seed]
kinds (sample proportions): code 50%, fixture 30%, comment 10%, multi 10%
  code:    one token: is->==, is not->!=, ==<->!=, < <= > >= shifts, and<->or, drop `not`,
           int +-1, True<->False
  comment: one of the sample's comment lines inserted on its own line
  fixture: one token inserted/removed in a line of tests/fixtures/optimizer/*.sql
  multi:   comments + one code and/or fixture change, in 2-3 files
"""
import ast
import difflib
import json
import random
import re
import sys
from pathlib import Path

src, out, count = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
rng = random.Random(int(sys.argv[4]) if len(sys.argv) > 4 else 11)
(out / "commits").mkdir(parents=True, exist_ok=True)

COMMENTS = ["# Intentionally not cached", "# Cheap check first", "# Keep in sync with the generator",
            "# Mirrors the behavior of the reference implementation", "# Kept for backwards compatibility"]
py_files = sorted(p for p in (src / "sqlglot").rglob("*.py") if p.stat().st_size > 2000)
fixtures = sorted((src / "tests" / "fixtures" / "optimizer").rglob("*.sql"))
CMP_SWAP = {ast.Is: ["=="], ast.IsNot: ["!="], ast.Eq: ["!="], ast.NotEq: ["=="],
            ast.Lt: ["<="], ast.LtE: ["<"], ast.Gt: [">="], ast.GtE: [">"]}
CMP_TXT = {ast.Is: " is ", ast.IsNot: " is not ", ast.Eq: " == ", ast.NotEq: " != ",
           ast.Lt: " < ", ast.LtE: " <= ", ast.Gt: " > ", ast.GtE: " >= "}


def func_nodes(tree):
    out = []
    for f in ast.walk(tree):
        if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for n in ast.walk(f):
                out.append(n)
    return out


def edit_line(text, lineno, col, old, new):
    lines = text.splitlines(True)
    line = lines[lineno - 1]
    b = line.encode()
    if b[col:col + len(old.encode())].decode(errors="replace") != old:
        return None
    lines[lineno - 1] = (b[:col] + new.encode() + b[col + len(old.encode()):]).decode()
    return "".join(lines)


def code_mutation(path):
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    nodes = func_nodes(tree) if rng.random() < 0.85 else list(ast.walk(tree))
    op = rng.choice(["cmp", "cmp", "cmp", "bool", "bool", "not", "int", "const_bool"])
    rng.shuffle(nodes)
    for n in nodes:
        if op == "cmp" and isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in CMP_SWAP:
            seg_l = n.left
            if seg_l.end_lineno != n.comparators[0].lineno:
                continue
            line = text.splitlines(True)[seg_l.end_lineno - 1].encode()
            between = line[seg_l.end_col_offset:n.comparators[0].col_offset].decode()
            old = CMP_TXT[type(n.ops[0])]
            if between != old:
                continue
            t = edit_line(text, seg_l.end_lineno, seg_l.end_col_offset, old, " " + CMP_SWAP[type(n.ops[0])][0] + " ")
            if t:
                return t, "cmp"
        if op == "bool" and isinstance(n, ast.BoolOp):
            a, b = n.values[0], n.values[1]
            if a.end_lineno != b.lineno:
                continue
            line = text.splitlines(True)[a.end_lineno - 1].encode()
            between = line[a.end_col_offset:b.col_offset].decode()
            old = " and " if isinstance(n.op, ast.And) else " or "
            if between.strip() != old.strip():
                continue
            new = " or " if old == " and " else " and "
            t = edit_line(text, a.end_lineno, a.end_col_offset, between, between.replace(old.strip(), new.strip()))
            if t:
                return t, "bool"
        if op == "not" and isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not) and n.lineno == n.operand.lineno:
            t = edit_line(text, n.lineno, n.col_offset, "not ", "")
            if t:
                return t, "not"
        if op == "int" and isinstance(n, ast.Constant) and type(n.value) is int and n.lineno == n.end_lineno:
            seg = ast.get_source_segment(text, n)
            if seg != str(n.value):
                continue
            t = edit_line(text, n.lineno, n.col_offset, seg, str(n.value + rng.choice([1, -1])))
            if t:
                return t, "int"
        if op == "const_bool" and isinstance(n, ast.Constant) and type(n.value) is bool:
            t = edit_line(text, n.lineno, n.col_offset, str(n.value), str(not n.value))
            if t:
                return t, "bool_const"
    return None, None


def comment_mutation(path):
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    stmts = [s for s in ast.walk(tree) if isinstance(s, ast.stmt)]
    s = rng.choice(stmts)
    lines = text.splitlines(True)
    line = lines[s.lineno - 1]
    indent = re.match(r"[ \t]*", line).group(0)
    lines.insert(s.lineno - 1, f"{indent}{rng.choice(COMMENTS)}\n")
    t = "".join(lines)
    try:
        if ast.dump(ast.parse(t)) != ast.dump(tree):
            return None, None
    except SyntaxError:
        return None, None
    return t, "comment"


def fixture_mutation(path):
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(True)
    cand = [i for i, l in enumerate(lines) if l.strip() and not l.lstrip().startswith(("--", "#"))]
    if not cand:
        return None, None
    i = rng.choice(cand)
    l = lines[i].rstrip("\n")
    words = [m.start() for m in re.finditer(r"(?<=\s)\S", l)] or [0]
    r = rng.random()
    if r < 0.5:
        pos = rng.choice(words)
        nl = l[:pos] + rng.choice(["42 ", "0 ", "1 ", "NULL ", "x "]) + l[pos:]
    elif r < 0.75:
        toks = list(re.finditer(r"\S+", l))
        tk = rng.choice(toks)
        nl = l[:tk.start()] + l[tk.end():]
    else:
        chars = [k for k, ch in enumerate(l) if ch in "<>(),."]
        if not chars:
            return None, None
        k = rng.choice(chars)
        nl = l[:k] + l[k + 1:]
    if nl == l:
        return None, None
    lines[i] = nl + "\n"
    return "".join(lines), "fixture"


def diff(rel, a, b):
    return "".join(difflib.unified_diff(a.splitlines(True), b.splitlines(True), f"a/{rel}", f"b/{rel}"))


meta = {}
n = 0
while n < count:
    r = rng.random()
    parts = []
    if r < 0.5:
        kind = "code"
        p = rng.choice(py_files)
        t, d = code_mutation(p)
        parts = [(p, t, d)]
    elif r < 0.8:
        kind = "fixture"
        p = rng.choice(fixtures)
        t, d = fixture_mutation(p)
        parts = [(p, t, d)]
    elif r < 0.9:
        kind = "comment"
        p = rng.choice(py_files)
        t, d = comment_mutation(p)
        parts = [(p, t, d)]
    else:
        kind = "multi"
        ps = rng.sample(py_files, 2)
        parts = [(ps[0],) + comment_mutation(ps[0])]
        if rng.random() < 0.5:
            parts.append((ps[1],) + code_mutation(ps[1]))
        else:
            parts.append((ps[1],) + comment_mutation(ps[1]))
        fx = rng.choice(fixtures)
        parts.append((fx,) + fixture_mutation(fx))
    if any(t is None for _, t, _ in parts):
        continue
    patch = "".join(diff(p.relative_to(src).as_posix(), p.read_text(encoding="utf-8"), t) for p, t, _ in parts)
    if not patch:
        continue
    mid = f"s{n:04d}"
    (out / "commits" / f"{mid}.patch").write_text(patch, encoding="utf-8")
    meta[mid] = {"kind": kind, "file": ",".join(p.relative_to(src).as_posix() for p, _, _ in parts),
                 "desc": ",".join(d for _, _, d in parts)}
    n += 1
(out / "mutants.json").write_text(json.dumps(meta, indent=1))
print("mutants", n, file=sys.stderr)
