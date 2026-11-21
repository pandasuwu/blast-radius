"""Generate synthetic single-file mutations of the pinned sqlglot as unified diffs.

usage: python eval/mutate.py <sqlglot_src> <out_dir> <count> [seed]

Writes <out_dir>/commits/<id>.patch and <out_dir>/mutants.json ({id: {kind, file, desc}}).
Mutants are for evaluation only: they are never shipped or used as a lookup.
"""
import ast
import difflib
import json
import random
import re
import sys
from pathlib import Path

src, out, count = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
rng = random.Random(int(sys.argv[4]) if len(sys.argv) > 4 else 1)
(out / "commits").mkdir(parents=True, exist_ok=True)

py_files = [p for p in sorted((src / "sqlglot").rglob("*.py"))]
weights = [max(1, len(p.read_text(encoding="utf-8").splitlines())) ** 0.7 for p in py_files]
fixtures = [p for p in sorted((src / "tests" / "fixtures").rglob("*")) if p.is_file() and p.suffix in (".sql", ".json", ".csv", ".txt")]
test_py = [p for p in sorted((src / "tests").rglob("*.py")) if "fixtures" not in p.parts]

OPS = {ast.Eq: "!=", ast.NotEq: "==", ast.Lt: "<=", ast.LtE: "<", ast.Gt: ">=", ast.GtE: ">",
       ast.Is: "is not", ast.IsNot: "is", ast.In: "not in", ast.NotIn: "in"}


def diff(rel, a, b):
    return "".join(difflib.unified_diff(a.splitlines(True), b.splitlines(True), f"a/{rel}", f"b/{rel}"))


def replace_segment(text, node, new):
    lines = text.splitlines(True)
    l0, c0, l1, c1 = node.lineno - 1, node.col_offset, node.end_lineno - 1, node.end_col_offset
    if l0 != l1:
        return None
    line = lines[l0]
    b = line.encode()
    nl = (b[:c0] + new.encode() + b[c1:]).decode()
    lines[l0] = nl
    return "".join(lines)


def in_function(tree):
    """map node -> enclosing scope kind ('f' function, 'c' class body, 'm' module)."""
    kinds = {}

    def walk(n, k):
        for c in ast.iter_child_nodes(n):
            kc = k
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                kc = "f"
            elif isinstance(n, ast.ClassDef):
                kc = "c" if k != "f" else "f"
            kinds[c] = kc
            walk(c, kc)

    walk(tree, "m")
    return kinds


def mutate_py(path):
    rel = path.relative_to(src).as_posix()
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    kinds = in_function(tree)
    nodes = list(ast.walk(tree))
    op = rng.choice(["cmp", "const", "const", "bool", "delete", "dictentry", "classattr", "comment", "docstring", "boolop", "arith"])
    if op == "cmp":
        c = [n for n in nodes if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in OPS]
        rng.shuffle(c)
        for n in c:
            seg = ast.get_source_segment(text, n)
            if not seg or "\n" in seg:
                continue
            new = f"{ast.get_source_segment(text, n.left)} {OPS[type(n.ops[0])]} {ast.get_source_segment(text, n.comparators[0])}"
            t = replace_segment(text, n, new)
            if t and t != text:
                return t, f"cmp {kinds.get(n)}@{n.lineno}"
    if op in ("const", "bool"):
        c = [n for n in nodes if isinstance(n, ast.Constant) and n.lineno == n.end_lineno
             and (isinstance(n.value, (int, str)) and not isinstance(n.value, bool) if op == "const" else isinstance(n.value, bool))]
        c = [n for n in c if not (isinstance(n.value, str) and kinds.get(n) == "m" and n.col_offset == 0)]
        rng.shuffle(c)
        for n in c:
            v = n.value
            if isinstance(v, bool):
                new = repr(not v)
            elif isinstance(v, int):
                new = repr(v + rng.choice([1, -1]))
            else:
                if len(v) > 60:
                    continue
                new = repr(v + "x" if not v or rng.random() < 0.5 else v[:-1])
            t = replace_segment(text, n, new)
            if t and t != text:
                try:
                    ast.parse(t)
                except SyntaxError:
                    continue
                return t, f"{op} {kinds.get(n)}@{n.lineno}"
    if op == "boolop":
        c = [n for n in nodes if isinstance(n, ast.BoolOp) and n.lineno == n.end_lineno]
        rng.shuffle(c)
        for n in c:
            seg = ast.get_source_segment(text, n)
            new = seg.replace(" and ", " or ", 1) if isinstance(n.op, ast.And) else seg.replace(" or ", " and ", 1)
            t = replace_segment(text, n, new)
            if t and t != text:
                return t, f"boolop {kinds.get(n)}@{n.lineno}"
    if op == "arith":
        c = [n for n in nodes if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub)) and n.lineno == n.end_lineno]
        rng.shuffle(c)
        for n in c:
            l = ast.get_source_segment(text, n.left)
            r = ast.get_source_segment(text, n.right)
            new = f"{l} {'-' if isinstance(n.op, ast.Add) else '+'} {r}"
            t = replace_segment(text, n, new)
            if t and t != text:
                return t, f"arith {kinds.get(n)}@{n.lineno}"
    if op == "delete":
        bodies = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.If, ast.For)) and kinds.get(n) == "f" or isinstance(n, ast.FunctionDef)]
        rng.shuffle(bodies)
        for b in bodies:
            stmts = [s for s in b.body if isinstance(s, (ast.Assign, ast.Expr, ast.Return, ast.AugAssign, ast.If)) and not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
            if not stmts:
                continue
            s = rng.choice(stmts)
            lines = text.splitlines(True)
            indent = re.match(r"\s*", lines[s.lineno - 1]).group(0)
            new_lines = lines[: s.lineno - 1] + [indent + "pass\n"] + lines[s.end_lineno:]
            t = "".join(new_lines)
            try:
                ast.parse(t)
            except SyntaxError:
                continue
            return t, f"delete f@{s.lineno}"
    if op == "dictentry":
        c = [n for n in nodes if isinstance(n, ast.Dict) and len(n.keys) > 2 and kinds.get(n) in ("c", "m")]
        rng.shuffle(c)
        for d in c:
            i = rng.randrange(len(d.keys))
            k, v = d.keys[i], d.values[i]
            if k is None or k.lineno != v.end_lineno:
                continue
            lines = text.splitlines(True)
            line = lines[k.lineno - 1]
            seg_end = v.end_col_offset
            b = line.encode()
            rest = b[seg_end:].decode()
            if not rest.strip().startswith(",") or b[:k.col_offset].decode().strip():
                continue
            lines[k.lineno - 1] = ""
            t = "".join(lines)
            try:
                ast.parse(t)
            except SyntaxError:
                continue
            return t, f"dictentry {kinds.get(d)}@{k.lineno}"
    if op == "classattr":
        c = [n for n in nodes if isinstance(n, ast.Assign) and kinds.get(n) == "c" and n.lineno == n.end_lineno
             and isinstance(n.value, ast.Constant)]
        rng.shuffle(c)
        for n in c:
            v = n.value.value
            new = repr(not v) if isinstance(v, bool) else repr(v + 1) if isinstance(v, int) else repr(None) if v is not None else "True"
            t = replace_segment(text, n.value, new)
            if t and t != text:
                return t, f"classattr c@{n.lineno}"
    if op == "comment":
        lines = text.splitlines(True)
        if not lines:
            return None, None
        i = rng.randrange(len(lines))
        if lines[i].strip() and not lines[i].rstrip().endswith("\\"):
            j = i
            # comment on its own line, indented like the next code line
            indent = re.match(r"\s*", lines[j]).group(0)
            t = "".join(lines[:j] + [f"{indent}# TODO: revisit this\n"] + lines[j:])
            try:
                if ast.dump(ast.parse(t)) == ast.dump(ast.parse(text)):
                    return t, "comment"
            except SyntaxError:
                pass
    if op == "docstring":
        c = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.body and isinstance(n.body[0], ast.Expr)
             and isinstance(n.body[0].value, ast.Constant) and isinstance(n.body[0].value.value, str)]
        rng.shuffle(c)
        for n in c:
            s = n.body[0]
            lines = text.splitlines(True)
            line = lines[s.lineno - 1]
            if '"""' in line:
                lines[s.lineno - 1] = line.replace('"""', '"""Note: ', 1)
                t = "".join(lines)
                try:
                    ast.parse(t)
                except SyntaxError:
                    continue
                return t, "docstring"
    return None, None


def mutate_fixture(path):
    text = path.read_text(encoding="utf-8", errors="strict")
    lines = text.splitlines(True)
    cand = [i for i, l in enumerate(lines) if l.strip() and not l.lstrip().startswith(("#", "--"))]
    if not cand:
        return None, None
    i = rng.choice(cand)
    l = lines[i]
    r = rng.random()
    if r < 0.4:
        nl = re.sub(r"\b(\d+)\b", lambda m: str(int(m.group(1)) + 1), l, count=1)
    elif r < 0.7:
        nl = l.replace("a", "b", 1) if "a" in l else l.rstrip("\n") + " \n"
    else:
        nl = l.rstrip("\n") + " -- note\n" if path.suffix == ".sql" else l
    if nl == l:
        return None, None
    lines[i] = nl
    return "".join(lines), f"fixture@{i+1}"


meta = {}
n = 0
attempts = 0
while n < count and attempts < count * 50:
    attempts += 1
    r = rng.random()
    if r < 0.80:
        p = rng.choices(py_files, weights)[0]
        new, desc = mutate_py(p)
        kind = "comment" if desc in ("comment", "docstring") else "code"
    elif r < 0.90:
        p = rng.choice(fixtures)
        try:
            new, desc = mutate_fixture(p)
        except UnicodeDecodeError:
            continue
        kind = "fixture"
    else:
        p = rng.choice(test_py)
        new, desc = mutate_py(p)
        kind = "test"
    if new is None:
        continue
    rel = p.relative_to(src).as_posix()
    patch = diff(rel, p.read_text(encoding="utf-8"), new)
    if not patch:
        continue
    mid = f"m{n:04d}"
    (out / "commits" / f"{mid}.patch").write_text(patch, encoding="utf-8")
    meta[mid] = {"kind": kind, "file": rel, "desc": desc}
    n += 1
(out / "mutants.json").write_text(json.dumps(meta, indent=1))
print("mutants", n, file=sys.stderr)
