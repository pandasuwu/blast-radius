"""Def-header mutants in sqlglot/executor/*.py for the v7.6 TPC-DS header rule (evaluation only).

usage: python eval/mutate_header.py <sqlglot_src> <out_dir>
One edit per commit, every function of every executor module:
  dflt   a parameter default: None -> 0 / True, True <-> False, int +1, str + "x"
  undec  the first decorator removed
  prop   @property added to a method (class body only)
  stat   @staticmethod added to a method (class body only)
  ann    return annotation replaced by `-> int`
Writes <out>/commits/eNNNN.patch and <out>/mutants.json ([{id, file, line, kind, func}]).
"""
import ast
import difflib
import json
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
(out / "commits").mkdir(parents=True, exist_ok=True)
muts = []


def emit(rel, old, new, line, kind, func):
    if new == old:
        return
    try:
        ast.parse(new)
    except SyntaxError:
        return
    mid = f"e{len(muts):04d}"
    diff = difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"a/{rel}", f"b/{rel}", n=3)
    (out / "commits" / f"{mid}.patch").write_text("".join(diff))
    muts.append({"id": mid, "file": rel, "line": line, "kind": kind, "func": func})


def replace_span(text, node, new_src):
    lines = text.splitlines(True)
    a, b = node.lineno - 1, node.end_lineno - 1
    if a != b:
        return None
    s = lines[a]
    lines[a] = s[:node.col_offset] + new_src + s[node.end_col_offset:]
    return "".join(lines)


for path in sorted((src / "sqlglot" / "executor").glob("*.py")):
    rel = path.relative_to(src).as_posix()
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    in_class = set()
    for c in ast.walk(tree):
        if isinstance(c, ast.ClassDef):
            in_class.update(id(x) for x in c.body)
    for f in ast.walk(tree):
        if not isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        defs = list(f.args.defaults) + [d for d in f.args.kw_defaults if d is not None]
        for d in defs:
            if not isinstance(d, ast.Constant):
                continue
            v = d.value
            alts = ([("0"), ("True")] if v is None else ["False"] if v is True else ["True"] if v is False
                    else [repr(v + 1)] if isinstance(v, int) else [repr(v + "x")] if isinstance(v, str) else [])
            for a in alts:
                new = replace_span(text, d, a)
                if new:
                    emit(rel, text, new, d.lineno, "dflt", f.name)
        lines = text.splitlines(True)
        if f.decorator_list:
            d = f.decorator_list[0]
            if d.lineno == d.end_lineno and lines[d.lineno - 1].strip().startswith("@"):
                emit(rel, text, "".join(lines[:d.lineno - 1] + lines[d.lineno:]), d.lineno, "undec", f.name)
        if id(f) in in_class and not f.decorator_list:
            ind = lines[f.lineno - 1][: len(lines[f.lineno - 1]) - len(lines[f.lineno - 1].lstrip())]
            for kind, dec in (("prop", "property"), ("stat", "staticmethod")):
                emit(rel, text, "".join(lines[:f.lineno - 1] + [f"{ind}@{dec}\n"] + lines[f.lineno - 1:]),
                     f.lineno, kind, f.name)
        if f.returns is not None and f.returns.lineno == f.returns.end_lineno:
            new = replace_span(text, f.returns, "int")
            if new:
                emit(rel, text, new, f.returns.lineno, "ann", f.name)

json.dump(muts, open(out / "mutants.json", "w"), indent=0)
print(len(muts), "mutants", {k: sum(m["kind"] == k for m in muts) for k in ("dflt", "undec", "prop", "stat", "ann")})
