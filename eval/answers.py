"""results/*.json -> answers.json in the kit's format (failing test numbers)."""
import json, os, sys
mut, tests_json = sys.argv[1], sys.argv[2]
tests = json.load(open(tests_json))["tests"]
num = {t["id"]: t["number"] for t in tests}
meta = json.load(open(f"{mut}/mutants.json"))
ans = {}
for mid, m in meta.items():
    p = f"{mut}/results/{mid}.json"
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    failing = {num[f] for f in d["failed"] if f in num}
    for c in d["collect"]:  # a test file that cannot be imported fails all its tests
        failing |= {t["number"] for t in tests if t["id"].startswith(c.split("::")[0])}
    ans[mid] = {"failing": sorted(failing), "kind": m["kind"]}
json.dump({"commits": ans}, open(f"{mut}/answers.json", "w"))
os.makedirs(f"{mut}/data/commits", exist_ok=True)
for mid in ans:
    open(f"{mut}/data/commits/{mid}.patch", "w").write(open(f"{mut}/commits/{mid}.patch").read())
json.dump({"tests": tests}, open(f"{mut}/data/tests.json", "w"))
print("answers", len(ans), "breaking", sum(1 for a in ans.values() if a["failing"]))
