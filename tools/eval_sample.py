"""Score an output.jsonl against answers and list per-commit misses / costs."""
import json, sys

data, answers, output = sys.argv[1:4]
tests = json.load(open(f"{data}/tests.json"))["tests"]
dur = {t["number"]: round(t["seconds"] * 1000) for t in tests}
ans = json.load(open(answers))["commits"]
tot = miss = 0
for line in open(output):
    r = json.loads(line)
    ws = [w for w in r["workers"] if w]
    cost = (max(sum(dur[n] for n in w) for w in ws) + 2000 * len(ws)) if ws else 0
    sel = {n for w in ws for n in w}
    a = ans.get(r["commit"])
    missed = [n for n in a["failing"] if n not in sel] if a else []
    tot += cost
    miss += bool(missed)
    print(f'{r["commit"]} {a["kind"] if a else "?":8} sel={len(sel):5} cost={cost/1000:7.3f} fail={len(a["failing"]) if a else "?"}'
          + (f' MISSED {[tests[n]["id"] for n in missed][:3]}' if missed else ""))
print(f"simulated {tot/1000 + 300*miss:.3f} s, misses {miss}")
