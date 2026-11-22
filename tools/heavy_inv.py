"""Inline heavy-test measurements in the real image.
  python -B heavy_inv.py pristine tests.json out.jsonl
  python -B heavy_inv.py mutant tests.json answers.json out.jsonl patch...
"""
import json, os, pickle, resource, sys, time
SRC = "/app/src"
sys.path.insert(0, "/w")  # heavy_verify.py
mode, tests_json = sys.argv[1], sys.argv[2]
tests = json.load(open(tests_json))["tests"]
HEAVY = [t for t in tests if t["file"] in ("tests/test_optimizer.py", "tests/test_executor.py")
         and not t["id"].endswith("::test_execute_tpcds")]
ids = [t["id"] for t in HEAVY]
num = {t["id"]: t["number"] for t in tests}
cost = {t["id"]: t["seconds"] for t in tests}


def forked(fn):
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            out = fn()
        except BaseException as e:
            out = {"error": repr(e)[:300]}
        os.write(w, json.dumps(out).encode())
        os._exit(0)
    os.close(w)
    data = b""
    while True:
        b = os.read(r, 1 << 16)
        if not b:
            break
        data += b
    os.close(r)
    _, status, ru = os.wait4(pid, 0)
    out = json.loads(data) if data else {"error": f"child died status {status}"}
    out["cpu"] = round(ru.ru_utime + ru.ru_stime, 3)
    return out


def setup_env(overlay=None):
    os.chdir(SRC)
    sys.path.insert(0, SRC)
    os.environ["SKIP_INTEGRATION"] = "0"
    if overlay:
        import importlib.util, importlib._bootstrap_external as be
        ov = {os.path.realpath(k): v.encode("utf-8") for k, v in overlay.items()}
        orig = be.FileLoader.get_data

        def get_data(self, path):
            rp = os.path.realpath(path)
            if rp.endswith(".pyc"):
                try:
                    src_rp = os.path.realpath(importlib.util.source_from_cache(path))
                except Exception:
                    raise OSError("unknown bytecode")
                if src_rp in ov:
                    raise OSError("bytecode of a patched file")
            return ov[rp] if rp in ov else orig(self, path)
        be.FileLoader.get_data = get_data


if mode == "pristine":
    out = open(sys.argv[3], "w")
    setup_env()
    t0 = time.process_time()
    import heavy_verify as H
    H.heavy_prepare()
    prep = time.process_time() - t0
    import tests.test_optimizer as TO, tests.test_executor as TE
    # class setup alone
    cs = {}
    for cls in (TO.TestOptimizer, TE.TestExecutor):
        cs[cls.__name__] = forked(lambda cls=cls: (cls.setUpClass(), {"ok": True})[1])["cpu"]
    out.write(json.dumps({"prepare_cpu": round(prep, 3), "setup_cpu": cs}) + "\n")
    for tid in ids:
        r = forked(lambda tid=tid: H.heavy_run([tid]))
        cls = tid.split("::")[1]
        rec = {"id": tid, "sim": cost[tid], "result": (r.get("results") or {}).get(tid), "first": r.get("first"),
               "error": r.get("error"), "cpu_with_setup": r["cpu"], "cpu": round(r["cpu"] - cs.get(cls, 0), 3)}
        out.write(json.dumps(rec) + "\n")
        out.flush()
    # whole files in one child each (setUpClass once): the verifier's real shape
    for f in ("tests/test_optimizer.py", "tests/test_executor.py"):
        sel = [i for i in ids if i.startswith(f)]
        r = forked(lambda sel=sel: H.heavy_run(sel))
        bad = [k for k, v in (r.get("results") or {}).items() if v != "pass"]
        out.write(json.dumps({"file": f, "n": len(sel), "sim": round(sum(cost[i] for i in sel), 3),
                              "cpu": r["cpu"], "not_pass": bad, "error": r.get("error")}) + "\n")
else:
    answers = json.load(open(sys.argv[3]))["commits"]
    out = open(sys.argv[4], "w")
    sys.path.insert(1, "/app")
    import selector as S
    for p in sys.argv[5:]:
        c = os.path.basename(p)[:-6]
        files = S.parse_patch(open(p, encoding="utf-8").read())
        ov = {}
        okp = True
        for f in files:
            rel = f["new"] or f["old"]
            if f["binary"] or not f["old"] or not f["new"] or f["old"] != f["new"]:
                okp = False
                break
            ov[os.path.join(SRC, rel)] = S.apply_hunks(open(os.path.join(SRC, rel), encoding="utf-8").read(), f["hunks"])[0]
        if not okp:
            continue

        def job(ov=ov):
            setup_env(ov)
            import heavy_verify as H
            H.install_text_overlay(ov)
            H.heavy_prepare()
            return H.heavy_run(ids)
        r = forked(job)
        res = r.get("results") or {}
        failing = set(answers[c]["failing"])
        truth = {i: ("fail" if num[i] in failing else "pass") for i in ids}
        wrong_pass = [i for i in ids if truth[i] == "fail" and res.get(i) == "pass"]
        wrong_fail = [i for i in ids if truth[i] == "pass" and res.get(i) != "pass"]
        rec = {"c": c, "files": sorted(f["new"] for f in files), "cpu": r["cpu"], "error": r.get("error"),
               "true_fail": sorted(i for i in ids if truth[i] == "fail"),
               "wrong_pass": wrong_pass, "wrong_fail": wrong_fail}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(c, "true_fail", len(rec["true_fail"]), "wrong_pass", len(wrong_pass), "wrong_fail", len(wrong_fail),
              "cpu", r["cpu"], flush=True)
