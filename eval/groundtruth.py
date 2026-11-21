"""Ground truth for mutants: apply each patch to a private sqlglot copy, run the WHOLE suite
(xdist), record failing tests. usage: groundtruth.py <mut_dir> <shard> <nshards> <nproc>"""
import json, os, shutil, subprocess, sys, time
sys.path.insert(0, "/w")
from selector import parse_patch, apply_hunks  # noqa: E402

mut, shard, nshards, nproc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
res_dir = f"{mut}/results"
os.makedirs(res_dir, exist_ok=True)
work = f"/src{shard}/sqlglot"
if not os.path.exists(work):
    shutil.copytree("/w/work/sqlglot", work)
patches = sorted(p for p in os.listdir(f"{mut}/commits") if p.endswith(".patch"))
env = dict(os.environ, PYTHONPATH=f"{work}:/w/eval", SKIP_INTEGRATION="0", PYTHONDONTWRITEBYTECODE="1",
           PYTHON_CPU_COUNT="2")  # caps the suite's ProcessPoolExecutor(spawn) pools
for i, name in enumerate(patches):
    if i % nshards != shard:
        continue
    mid = name[:-6]
    out = f"{res_dir}/{mid}.json"
    if os.path.exists(out):
        continue
    text = open(f"{mut}/commits/{name}", encoding="utf-8").read()
    saved = {}
    try:
        for f in parse_patch(text):
            p = f"{work}/{f['old']}"
            old = open(p, encoding="utf-8").read()
            saved[p] = old
            new = apply_hunks(old, f["hunks"])
            open(p, "w", encoding="utf-8").write(new)
        t0 = time.time()
        env["FAIL_OUT"] = out + ".tmp"
        r = subprocess.run([sys.executable, "-m", "pytest", "-p", "failplugin", "-p", "no:cacheprovider", "-q",
                            "-n", nproc, "--timeout=600", "tests"], cwd=work, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        os.replace(out + ".tmp", out)
        d = json.load(open(out))
        print(mid, f"{time.time()-t0:.0f}s", "failed", len(d["failed"]), "collect", len(d["collect"]), flush=True)
    finally:
        for p, old in saved.items():
            open(p, "w", encoding="utf-8").write(old)
