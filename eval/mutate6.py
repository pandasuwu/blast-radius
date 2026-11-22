"""tpc-ds.sql corruptions that change an executed query's (meta, sql) (evaluation only, br-tools).

usage: BR_SRC=/w/work/sqlglot python eval/mutate6.py <out_dir> <count> [seed]
Same token corruptions as eval/mutate2.py's fixture kind, kept only when the pairs
test_execute_tpcds reads change, and only in a query it executes.
"""
import difflib
import json
import random
import re
import sys

sys.path.insert(0, "/w")
import selector as S  # noqa: E402

out, count = sys.argv[1], int(sys.argv[2])
rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 3)
rel = "tests/fixtures/optimizer/tpc-ds/tpc-ds.sql"
text = (S.SRC / rel).read_text(encoding="utf-8")
base = S._tpcds_pairs_used(text)


def mutate(text):
    lines = text.splitlines(True)
    cand = [i for i, l in enumerate(lines) if l.strip() and not l.lstrip().startswith(("--", "#"))]
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
            return None
        k = rng.choice(chars)
        nl = l[:k] + l[k + 1:]
    if nl == l:
        return None
    lines[i] = nl + "\n"
    return "".join(lines)


import os  # noqa: E402
os.makedirs(f"{out}/commits", exist_ok=True)
n = tries = 0
meta = {}
while n < count and tries < count * 400:
    tries += 1
    new = mutate(text)
    if new is None:
        continue
    try:
        pairs = S._tpcds_pairs_used(new)
    except Exception:
        continue
    if len(pairs) != len(base):
        continue
    changed = [k for k, (a, b) in enumerate(zip(base, pairs)) if a != b]
    if len(changed) != 1:
        continue
    import ast as _a
    m = dict(_a.literal_eval(base[changed[0]][0]))
    if str(m.get("execute", "")).lower() not in ("true", "1"):
        continue
    patch = "".join(difflib.unified_diff(text.splitlines(True), new.splitlines(True), f"a/{rel}", f"b/{rel}"))
    mid = f"q{n:03d}"
    open(f"{out}/commits/{mid}.patch", "w", encoding="utf-8").write(patch)
    meta[mid] = {"kind": "fixture", "query": changed[0]}
    n += 1
json.dump(meta, open(f"{out}/mutants.json", "w"), indent=1)
print("mutants", n, "tries", tries, file=sys.stderr)
