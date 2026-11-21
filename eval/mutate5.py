"""Mutants aimed at the v6.9 rules (evaluation only).

usage: python eval/mutate5.py <sqlglot_src> <out_dir> <per_kind> [seed]
Kinds (one token per commit, organizer style):
  dictval  a constant inside a value of a module/class-level dict literal (int +-1, True<->False)
  dflt     a constant default of a function parameter (True<->False, int +-1)
  flag     a class-level `NAME = True/False` flipped
  notflip  a `not` dropped from, or added to, an operand of an and/or inside a function
"""
import ast
import difflib
import json
import random
import sys
from pathlib import Path

src, out, per_kind = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
rng = random.Random(int(sys.argv[4]) if len(sys.argv) > 4 else 11)
(out / "commits").mkdir(parents=True, exist_ok=True)
files = sorted(p for p in (src / "sqlglot").rglob("*.py") if p.stat().st_size > 1500)


def edit(text, line, col, old, new):
    lines = text.splitlines(True)
    b = lines[line - 1].encode()
    if b[col:col + len(old)].decode(errors="replace") != old:
        return None
    lines[line - 1] = (b[:col] + new.encode() + b[col + len(old):]).decode()
    return "".join(lines)


def const_flip(text, c):
    seg = ast.get_source_segment(text, c)
    if c.lineno != c.end_lineno or seg is None:
        return None
    if type(c.value) is bool and seg in ("True", "False"):
        return edit(text, c.lineno, c.col_offset, seg, str(not c.value))
    if type(c.value) is int and seg == str(c.value):
        return edit(text, c.lineno, c.col_offset, seg, str(c.value + rng.choice((1, -1))))
    return None


def sites(text, tree, kind):
    out = []
    if kind == "dictval":
        def top(body):
            for s in body:
                if isinstance(s, (ast.Assign, ast.AnnAssign)) and isinstance(s.value, ast.Dict):
                    for v in s.value.values:
                        out.extend(x for x in ast.walk(v) if isinstance(x, ast.Constant)
                                   and not any(isinstance(l, ast.Lambda) for l in ast.walk(v)))
                if isinstance(s, ast.ClassDef):
                    top(s.body)
        top(tree.body)
    elif kind == "dflt":
        for f in ast.walk(tree):
            if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.extend(d for d in f.args.defaults + f.args.kw_defaults if isinstance(d, ast.Constant))
    elif kind == "flag":
        for c in ast.walk(tree):
            if isinstance(c, ast.ClassDef):
                for s in c.body:
                    if isinstance(s, ast.Assign) and isinstance(s.value, ast.Constant) and type(s.value.value) is bool:
                        out.append(s.value)
    elif kind == "notflip":
        for f in ast.walk(tree):
            if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for b in ast.walk(f):
                    if isinstance(b, ast.BoolOp):
                        out.extend(("op", v) for v in b.values)
    return out


meta, n = {}, 0
for kind in ("dictval", "dflt", "flag", "notflip"):
    made = tries = 0
    while made < per_kind and tries < per_kind * 200:
        tries += 1
        p = rng.choice(files)
        text = p.read_text(encoding="utf-8")
        tree = ast.parse(text)
        cands = sites(text, tree, kind)
        if not cands:
            continue
        s = rng.choice(cands)
        new = None
        if kind == "notflip":
            v = s[1]
            if isinstance(v, ast.UnaryOp) and isinstance(v.op, ast.Not):
                seg = text.splitlines(True)[v.lineno - 1].encode()[v.col_offset:v.col_offset + 4].decode(errors="replace")
                if seg == "not ":
                    new = edit(text, v.lineno, v.col_offset, "not ", "")
            elif isinstance(v, (ast.Name, ast.Attribute, ast.Call, ast.Compare)) and v.lineno == v.end_lineno:
                seg = ast.get_source_segment(text, v)
                if seg:
                    wrapped = f"not {seg}" if not isinstance(v, ast.Compare) else f"not ({seg})"
                    new = edit(text, v.lineno, v.col_offset, seg, wrapped)
        else:
            new = const_flip(text, s)
        if not new or new == text:
            continue
        try:
            ast.parse(new)
        except SyntaxError:
            continue
        rel = p.relative_to(src).as_posix()
        patch = "".join(difflib.unified_diff(text.splitlines(True), new.splitlines(True), f"a/{rel}", f"b/{rel}"))
        mid = f"t{n:04d}"
        (out / "commits" / f"{mid}.patch").write_text(patch, encoding="utf-8")
        meta[mid] = {"kind": "code", "file": rel, "desc": kind}
        n += 1
        made += 1
(out / "mutants.json").write_text(json.dumps(meta, indent=1))
print("mutants", n, file=sys.stderr)
