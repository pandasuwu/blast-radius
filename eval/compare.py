"""Compare selector versions per commit: cost, misses, and regressions vs a baseline.

usage: python eval/compare.py <cmp_dir> <tests.json> base other [other...]
"""
import json
import sys

cmp_dir, tests_json, *vers = sys.argv[1:]
tests = json.load(open(tests_json))["tests"]
dur = [round(t["seconds"] * 1000) for t in tests]
SETS = {"sample": "sample/answers.json", "mut": "work/mut/answers.json", "mut2": "work/mut2/answers.json"}


def load(ver, s):
    out = {}
    for line in open(f"{cmp_dir}/{ver}_{s}.jsonl"):
        r = json.loads(line)
        ws = [w for w in r["workers"] if w]
        cost = (max(sum(dur[n] for n in w) for w in ws) + 2000 * len(ws)) if ws else 0
        out[r["commit"]] = (cost, {n for w in ws for n in w})
    return out


base = vers[0]
for s, ans_path in SETS.items():
    ans = json.load(open(ans_path))["commits"]
    res = {v: load(v, s) for v in vers}
    print(f"== {s} ({len(ans)} commits with ground truth)")
    for v in vers:
        tot = miss = 0
        for c, a in ans.items():
            cost, sel = res[v][c]
            tot += cost
            miss += any(f not in sel for f in a["failing"])
        print(f"   {v:5} simulated {tot/1000 + 300*miss:9.3f}  misses {miss}")
    for v in vers[1:]:
        worse = [(res[v][c][0] - res[base][c][0], c) for c in ans if res[v][c][0] > res[base][c][0] + 50]
        worse.sort(reverse=True)
        print(f"   {v} worse than {base} on {len(worse)}: " + ", ".join(f"{c}+{d/1000:.1f}" for d, c in worse[:12]))
