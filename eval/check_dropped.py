"""Out-of-sample check of a new version's extra skips (evaluation only, runs in br-tools).

usage: python eval/check_dropped.py <data_dir> <base.jsonl> <new.jsonl> <out.jsonl> [shard shards]
For each commit: the tests base selected and new did not are the only ones new can miss (when base
misses nothing). Run exactly those on the patched source; any failure is a miss of new.
"""
import json
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, "/w")
from selector import apply_hunks, parse_patch  # noqa: E402

data, base_p, new_p, out_p = sys.argv[1:5]
shard, shards = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (0, 1)
tests = json.load(open(f"{data}/tests.json"))["tests"]
ids = {t["number"]: t["id"] for t in tests}


def sel(p):
    return {r["commit"]: {n for w in r["workers"] for n in w} for r in map(json.loads, open(p))}


base, new = sel(base_p), sel(new_p)
done = set()
if os.path.exists(out_p):
    done = {json.loads(l)["commit"] for l in open(out_p)}
out = open(out_p, "a")
PIN = "/w/work/sqlglot"
for i, c in enumerate(sorted(base)):
    if i % shards != shard or c in done:
        continue
    dropped = sorted(base[c] - new.get(c, set()))
    if not dropped:
        out.write(json.dumps({"commit": c, "dropped": 0, "failed": []}) + "\n")
        out.flush()
        continue
    work = f"/tmp/cd{shard}"
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(PIN, work, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    for f in parse_patch(open(f"{data}/commits/{c}.patch", encoding="utf-8").read()):
        old = open(f"{PIN}/{f['old']}", encoding="utf-8").read()
        with open(f"{work}/{f['new']}", "w", encoding="utf-8", newline="") as fh:
            fh.write(apply_hunks(old, f["hunks"])[0])
    env = {**os.environ, "SKIP_INTEGRATION": "0", "PYTHONPATH": work, "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHON_CPU_COUNT": "2"}
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "-rfE", "--no-header",
                            "--timeout=600"] + [ids[n] for n in dropped], cwd=work, env=env, capture_output=True,
                           text=True, timeout=1800)
        txt = r.stdout + r.stderr
        failed = sorted(set(re.findall(r"^(?:FAILED|ERROR|SUBFAIL) (\S+)", txt, re.M)))
        if r.returncode not in (0, 1) and not failed:
            failed = [f"rc={r.returncode}: " + (txt.strip().splitlines() or [""])[-1][:200]]
        tail = (txt.strip().splitlines() or [""])[-1][:200]
    except subprocess.TimeoutExpired:
        failed, tail = ["timeout"], "timeout"
    out.write(json.dumps({"commit": c, "dropped": len(dropped), "failed": failed, "tail": tail}) + "\n")
    out.flush()
    print(c, len(dropped), "FAIL" if failed else "ok", tail[:80], file=sys.stderr, flush=True)
