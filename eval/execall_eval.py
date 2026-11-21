"""Fidelity of the TPC-DS check in execute-all mode on labelled commits whose check v5.9 refuses.

usage (br-tools, repo at /w, BR_SRC=/w/work/sqlglot, BR_INDEX=<prepped index>):
    python eval/execall_eval.py out.jsonl
For every labelled commit whose v5 selection holds test_execute_tpcds and whose v5.9 log shows no
check (a guard refused it), run the check with exec_all=True and compare with the labels:
false pass = "same rows" although TPC-DS fails in the suite (would be a miss).
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, "/w")
import selector as S  # noqa: E402

out_path = sys.argv[1]
idx = S.load_index()
ref = idx["tpcds_ref"]
tests = json.load(open("/w/sample/tests.json"))["tests"]
TP = next(t["number"] for t in tests if t["id"].endswith("test_execute_tpcds"))
D = {"sample": "/w/sample"} | {s: f"/w/work/{s}/data" for s in ("mut", "mut2", "mut3")}
A = {"sample": "/w/sample/answers.json"} | {s: f"/w/work/{s}/answers.json" for s in ("mut", "mut2", "mut3")}
out = open(out_path, "w")
agg = {}
for s in D:
    ans = json.load(open(A[s]))["commits"]
    logs = {}
    for line in open(f"/w/work/cmp/{os.environ.get('BR_LOGS', 'cur')}_{s}.log", encoding="utf-8", errors="replace"):
        logs.setdefault(line.split(" ", 1)[0], []).append(line)
    for line in open(f"/w/work/cmp/v5_{s}.jsonl"):
        r = json.loads(line)
        c = r["commit"]
        if c not in ans or TP not in {n for w in r["workers"] for n in w}:
            continue
        lg = " ".join(logs.get(c, []))
        if "left out" in lg or "tpcds check:" in lg or "not worth it" in lg:
            continue
        files = S.parse_patch(open(f"{D[s]}/commits/{c}.patch", encoding="utf-8").read())
        overlay = {}
        for f in files:
            if f["new"] and f["new"].endswith(".py") and f["old"]:
                old = open(f"{S.SRC}/{f['old']}", encoding="utf-8").read()
                overlay[str(S.SRC / f["new"])] = S.apply_hunks(old, f["hunks"])[0]
        t0 = time.time()
        v, _ = S.tpcds_run(overlay, ref, budget=40.0, limit=60.0, timeout=180, exec_all=True)
        fails = TP in ans[c]["failing"]
        kind = ("false_pass" if fails else "true_pass") if v.get("ok") else ("true_fail" if fails else "false_fail")
        agg[kind] = agg.get(kind, 0) + 1
        rec = {"set": s, "commit": c, "kind": kind, "cpu": v.get("cpu"), "wall": round(time.time() - t0, 1),
               "why": v.get("why"), "files": [f["new"] or f["old"] for f in files]}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(kind, s, c, v.get("cpu"), (v.get("why") or "")[:80], file=sys.stderr)
print(agg, file=sys.stderr)
