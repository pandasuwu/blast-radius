"""Fidelity of selector.verify_run on labelled commits (stage 1 of runtime verification).

usage (br-tools, repo at /w, BR_SRC=/w/work/sqlglot): python eval/verify_eval.py out.jsonl [sets...]
For every commit whose v5 selection is small (< 6 s) and avoids the optimizer/executor test modules,
run that selection on the patched source and compare with the full-suite labels:
false pass = all selected tests passed here but one of them fails in the suite (would be a miss).
"""
import json
import resource
import sys
import time

sys.path.insert(0, "/w")
import selector as S  # noqa: E402

HEAVY = {"tests/test_optimizer.py", "tests/test_executor.py"}
SETS = {"sample": "/w/sample"} | {s: f"/w/work/{s}/data" for s in ("mut", "mut2", "mut3")}
ANS = {"sample": "/w/sample/answers.json"} | {s: f"/w/work/{s}/answers.json" for s in ("mut", "mut2", "mut3")}
out_path, *sets = sys.argv[1:]
tests = json.load(open("/w/sample/tests.json"))["tests"]
dur = [round(t["seconds"] * 1000) for t in tests]
ids = [t["id"] for t in tests]
fil = [t["file"] for t in tests]
out = open(out_path, "w")
agg = {"n": 0, "false_pass": 0, "false_fail": 0, "true_pass": 0, "true_fail": 0}
for s in sets or list(SETS):
    ans = json.load(open(ANS[s]))["commits"]
    for line in open(f"/w/work/cmp/v5_{s}.jsonl"):
        r = json.loads(line)
        c = r["commit"]
        ws = [w for w in r["workers"] if w]
        if c not in ans or not ws:
            continue
        cost = max(sum(dur[n] for n in w) for w in ws) + 2000 * len(ws)
        sel = sorted({n for w in ws for n in w})
        if cost >= 6000 or any(fil[n] in HEAVY for n in sel):
            continue
        files = S.parse_patch(open(f"{SETS[s]}/commits/{c}.patch", encoding="utf-8").read())
        overlay, skip = {}, False
        for f in files:
            if f["binary"] or not f["new"]:
                skip = True
                break
            old = open(f"{S.SRC}/{f['old']}", encoding="utf-8").read() if f["old"] else ""
            new, _ = S.apply_hunks(old, f["hunks"])
            overlay[str(S.SRC / f["new"])] = new
        if skip:
            continue
        c0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        t0 = time.time()
        v = S.verify_run(overlay, [ids[n] for n in sel])
        c1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = round(c1.ru_utime + c1.ru_stime - c0.ru_utime - c0.ru_stime, 3)
        fails = set(ans[c]["failing"]) & set(sel)
        kind = ("true_pass" if not fails else "false_pass") if v["ok"] else ("true_fail" if fails else "false_fail")
        agg["n"] += 1
        agg[kind] += 1
        rec = {"set": s, "commit": c, "kind": kind, "cpu": cpu, "wall": round(time.time() - t0, 2),
               "ntests": len(sel), "bad": v.get("bad"), "failing_selected": sorted(fails)}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        if kind.startswith("false"):
            print(kind.upper(), s, c, v.get("bad"), sorted(fails)[:5], file=sys.stderr)
print(agg, file=sys.stderr)
