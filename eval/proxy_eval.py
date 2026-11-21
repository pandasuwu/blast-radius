"""Run tools/tpcds_proxy.py on every labeled commit whose v5 selection includes TPC-DS.

usage (in br-tools, repo at /w): python eval/proxy_eval.py <out.jsonl> [sets...]
For each commit: apply the patch over a copy of the pinned checkout, run the proxy there,
and record whether its hash equals the pristine one and whether TPC-DS really failed.
"""
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, "/w")
from selector import apply_hunks, parse_patch  # noqa: E402

PIN = "/w/work/sqlglot"
TPCDS = "tests/test_executor.py::TestExecutor::test_execute_tpcds"
SETS = {"sample": ("/w/sample", "/w/sample/answers.json")}
for s in ("mut", "mut2", "mut3"):
    SETS[s] = (f"/w/work/{s}/data", f"/w/work/{s}/answers.json")


def proxy(root):
    env = {**os.environ, "PYTHONPATH": root}
    code = f"import sys; sys.path.insert(0, {root!r}); exec(open('/w/tools/tpcds_proxy.py').read())"
    try:
        p = subprocess.run([sys.executable, "-I", "-c", code], cwd=root, env=env, capture_output=True,
                           text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return "timeout", None
    lines = [x for x in p.stderr.splitlines() if not x.startswith("Unable to set")]
    last = lines[-1].split() if lines else []
    if p.returncode or len(last) != 4:
        return "error", (lines[-1][:200] if lines else p.returncode)
    return last[0], float(last[3])


out_path, *sets = sys.argv[1:]
sets = sets or list(SETS)
base, _ = proxy(PIN)
print("pristine", base, file=sys.stderr)
done = set()
if os.path.exists(out_path):
    done = {(r["set"], r["commit"]) for r in map(json.loads, open(out_path))}
out = open(out_path, "a")
for s in sets:
    data, ans_path = SETS[s]
    tests = json.load(open(f"{data}/tests.json"))["tests"]
    num = next(t["number"] for t in tests if t["id"] == TPCDS)
    ans = json.load(open(ans_path))["commits"]
    for line in open(f"/w/work/cmp/v5_{s}.jsonl"):
        r = json.loads(line)
        c = r["commit"]
        if c not in ans or (s, c) in done or num not in {n for w in r["workers"] for n in w}:
            continue
        ov = "/tmp/ov"
        shutil.rmtree(ov, ignore_errors=True)
        shutil.copytree(PIN, ov, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.csv.gz"),
                        symlinks=True)
        # the proxy never reads the TPC data files; link them so the copy stays small
        files = parse_patch(open(f"{data}/commits/{c}.patch", encoding="utf-8").read())
        for f in files:
            if f["old"] and f["old"] != f["new"] and os.path.exists(f"{ov}/{f['old']}"):
                os.remove(f"{ov}/{f['old']}")
            if f["new"]:
                old = open(f"{PIN}/{f['old']}", encoding="utf-8").read() if f["old"] else ""
                new, _ = apply_hunks(old, f["hunks"])
                os.makedirs(os.path.dirname(f"{ov}/{f['new']}") or ".", exist_ok=True)
                with open(f"{ov}/{f['new']}", "w", encoding="utf-8", newline="") as fh:
                    fh.write(new)
        h, cpu = proxy(ov)
        rec = {"set": s, "commit": c, "equal": h == base, "hash": h, "cpu": cpu,
               "tpcds_fails": num in ans[c]["failing"], "nfail": len(ans[c]["failing"]),
               "files": [f["new"] or f["old"] for f in files]}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(s, c, "equal" if rec["equal"] else h[:12], "FAILS" if rec["tpcds_fails"] else "", cpu,
              file=sys.stderr)
