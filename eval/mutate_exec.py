"""Organizer-style mutants in the code test_execute_tpcds runs outside the proof's compared calls.

usage: python eval/mutate_exec.py <sqlglot checkout> <maps/tpcds_phase.json> <out dir> [n] [seed]
Writes <out>/commits/uXXX.patch (one-line mutations: comparison swap, and/or, dropped not,
int +-1, True/False, string literal edit) on lines the uncompared phase runs, plus class-level
settings of classes it uses. Evaluation only; never shipped.
"""
import ast
import difflib
import json
import os
import random
import re
import sys

src, phase, out = sys.argv[1:4]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 150
rnd = random.Random(int(sys.argv[5]) if len(sys.argv) > 5 else 11)
ph = json.load(open(phase))
cands = []
for rel, lines in ph["lines"].items():
    for ln in lines:
        cands.append((rel, ln, "call"))
# class-level assignments in classes the uncompared phase uses (import-time settings)
used = set(ph["classes"])
for rel in ph["lines"]:
    text = open(os.path.join(src, rel), encoding="utf-8").read()
    mod = rel[:-3].replace("/", ".")
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ClassDef) and f"{mod}:{node.name}" in used:
            for s in node.body:
                if isinstance(s, (ast.Assign, ast.AnnAssign)) and s.lineno == s.end_lineno:
                    cands.append((rel, s.lineno, "class"))

OPS = [
    (r"(?<![=!<>])==(?!=)", "!="), (r"!=", "=="), (r"(?<![<>=])<(?![<=])", "<="), (r"<=", "<"),
    (r"(?<![<>=-])>(?![>=])", ">="), (r">=", ">"), (r"\band\b", "or"), (r"\bor\b", "and"),
    (r"\bnot\s+", ""), (r"\bTrue\b", "False"), (r"\bFalse\b", "True"),
    (r"(?<![\w.])(\d+)(?![\w.])", None), (r"\"([A-Za-z_]+)\"", None),
]


def mutate(line):
    opts = []
    for pat, rep in OPS:
        for m in re.finditer(pat, line):
            if rep is None and pat.startswith("(?<![\\w.])"):
                v = int(m.group(1))
                r = str(v + rnd.choice((-1, 1)))
            elif rep is None:
                r = f'"{m.group(1)}x"'
            else:
                r = rep
            opts.append(line[:m.start()] + r + line[m.end():])
    return rnd.choice(opts) if opts else None


os.makedirs(os.path.join(out, "commits"), exist_ok=True)
rnd.shuffle(cands)
made, seen = 0, set()
for rel, ln, kind in cands:
    if made >= n:
        break
    text = open(os.path.join(src, rel), encoding="utf-8").read()
    lines = text.split("\n")
    line = lines[ln - 1]
    code = line.split("#", 1)[0]
    if not code.strip() or code.strip().startswith(("def ", "class ", "@", "import ", "from ")):
        continue
    new = mutate(code)
    if new is None or new == code:
        continue
    new_line = new + line[len(code):]
    new_text = "\n".join(lines[:ln - 1] + [new_line] + lines[ln:])
    try:
        ast.parse(new_text)
    except SyntaxError:
        continue
    if (rel, ln) in seen:
        continue
    seen.add((rel, ln))
    diff = "".join(difflib.unified_diff(text.splitlines(True), new_text.splitlines(True),
                                        f"a/{rel}", f"b/{rel}", n=3))
    name = f"u{made:03d}"
    open(os.path.join(out, "commits", name + ".patch"), "w", newline="\n").write(diff)
    made += 1
print("made", made, "mutants from", len(cands), "candidate lines", file=sys.stderr)
