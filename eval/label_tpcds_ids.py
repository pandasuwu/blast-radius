"""Ground truth for test_execute_tpcds only: run it on the patched source of each listed commit.

usage (in br-tools, /w = repo): python eval/label_tpcds_ids.py <commits_dir> <ids_file> <out_jsonl>
Copies /w/work/sqlglot to /tmp/sq once (no bytecode), applies each single-file unified diff,
runs the test, restores the file. Appends {"commit", "tpcds": pass|fail|apply-failed, "sec"}.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time

commits, ids_file, out = sys.argv[1:4]
shutil.rmtree("/tmp/sq", ignore_errors=True)
shutil.copytree("/w/work/sqlglot", "/tmp/sq", ignore=shutil.ignore_patterns("__pycache__", ".git"))
env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHON_CPU_COUNT": "2"}
done = set()
if os.path.exists(out):
    done = {json.loads(l)["commit"] for l in open(out) if l.strip()}


def apply(text, patch):
    lines = text.splitlines(True)
    outl, pos = [], 0
    for m in re.finditer(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@.*\n((?:[ +\-].*\n?|\\.*\n?)*)", patch, re.M):
        start = int(m.group(1)) - (1 if m.group(2) != "0" else 0)
        outl += lines[pos:start]
        pos = start
        for h in m.group(3).splitlines(True):
            if h.startswith("\\"):
                continue
            if h[0] in " -":
                if lines[pos].rstrip("\n") != h[1:].rstrip("\n"):
                    raise ValueError(f"context mismatch at {pos + 1}")
                if h[0] == " ":
                    outl.append(lines[pos])
                pos += 1
            else:
                outl.append(h[1:] if h.endswith("\n") else h[1:] + "\n")
    return "".join(outl + lines[pos:])


for cid in [l.strip() for l in open(ids_file) if l.strip()]:
    if cid in done:
        continue
    patch = open(f"{commits}/{cid}.patch").read()
    rel = re.search(r"^\+\+\+ b/(\S+)", patch, re.M).group(1)
    path = f"/tmp/sq/{rel}"
    orig = open(path).read()
    rec = {"commit": cid}
    try:
        open(path, "w").write(apply(orig, patch))
        t0 = time.time()
        p = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                            "tests/test_executor.py::TestExecutor::test_execute_tpcds"],
                           cwd="/tmp/sq", env=env, capture_output=True, text=True, timeout=400)
        rec.update(tpcds="pass" if p.returncode == 0 else "fail", sec=round(time.time() - t0, 1))
        if p.returncode:
            rec["tail"] = p.stdout.strip().splitlines()[-1][:160] if p.stdout.strip() else p.stderr[-160:]
    except subprocess.TimeoutExpired:
        rec.update(tpcds="fail", sec=400, tail="timeout")
    except Exception as e:
        rec.update(tpcds="apply-failed", tail=repr(e)[:160])
    finally:
        open(path, "w").write(orig)
    with open(out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
