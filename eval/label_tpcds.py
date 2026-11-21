"""Label mutants with the real test_execute_tpcds (pytest, as the organizers run it).

usage (br-tools, repo at /w): python eval/label_tpcds.py <data dir> <out.jsonl> <shard> <shards>
For every <data>/commits/*.patch in this shard: copy the pinned checkout, apply the patch, run
pytest on tests/test_executor.py::TestExecutor::test_execute_tpcds and record pass/fail.
"""
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, "/w")
from selector import apply_hunks, parse_patch  # noqa: E402

data, out_path, shard, shards = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
PIN = "/w/work/sqlglot"
TID = "tests/test_executor.py::TestExecutor::test_execute_tpcds"
done = set()
if os.path.exists(out_path):
    done = {json.loads(l)["commit"] for l in open(out_path)}
patches = sorted(os.listdir(f"{data}/commits"))[shard::shards]
out = open(out_path, "a")
for p in patches:
    c = p[:-6]
    if c in done:
        continue
    work = f"/tmp/lab{shard}"
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(PIN, work, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    for f in parse_patch(open(f"{data}/commits/{p}", encoding="utf-8").read()):
        old = open(f"{PIN}/{f['old']}", encoding="utf-8").read()
        with open(f"{work}/{f['new']}", "w", encoding="utf-8", newline="") as fh:
            fh.write(apply_hunks(old, f["hunks"])[0])
    env = {**os.environ, "SKIP_INTEGRATION": "0", "PYTHONPATH": work, "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHON_CPU_COUNT": "2"}
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", TID, "-p", "no:cacheprovider", "-q", "-x",
                            "--timeout=600", "--no-header"], cwd=work, env=env, capture_output=True,
                           text=True, timeout=900)
        tail = (r.stdout + r.stderr).strip().splitlines()[-1:] or [""]
        ok = r.returncode == 0 and "passed" in tail[0] and "fail" not in tail[0] and "error" not in tail[0]
    except subprocess.TimeoutExpired:
        ok, tail = False, ["timeout"]
    out.write(json.dumps({"commit": c, "tpcds_passes": ok, "tail": tail[0][:200]}) + "\n")
    out.flush()
    print(c, "PASS" if ok else "FAIL", tail[0][:100], file=sys.stderr, flush=True)
