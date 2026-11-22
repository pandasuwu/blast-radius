"""Upper-bound saving from verifying heavy tests, on labelled sets.
usage: python est_saving.py pristine.jsonl tests.json VER set:answers.json:selections.jsonl ...
Class H = selections holding a heavy test (not test_execute_tpcds) and no TPC-DS. Verified selection =
the selection minus heavy tests that really pass (labels as the verifier's verdict), LPT-repacked.
CPU = per verify: class setups needed + inline CPU of the selected heavy tests (pristine) + 0.05 s fork."""
import json, sys

prist = [json.loads(l) for l in open(sys.argv[1])]
setup = prist[0]["setup_cpu"]
inl = {r["id"]: max(0.0, r["cpu"]) for r in prist[1:] if "id" in r}
tests = json.load(open(sys.argv[2]))["tests"]
by = {t["number"]: t for t in tests}
ms = {t["number"]: round(t["seconds"] * 1000) for t in tests}
TPCDS = [t["number"] for t in tests if t["id"].endswith("::test_execute_tpcds")][0]
HEAVY = {t["number"] for t in tests if t["file"] in ("tests/test_optimizer.py", "tests/test_executor.py")} - {TPCDS}


def cost(workers):
    ws = [w for w in workers if w]
    return (max(sum(ms[n] for n in w) for w in ws) + 2000 * len(ws)) / 1000 if ws else 0.0


def best_pack(nums):
    if not nums:
        return 0.0
    best = None
    for k in range(1, 9):
        loads = [0] * k
        for n in sorted(nums, key=lambda n: -ms[n]):
            i = loads.index(min(loads))
            loads[i] += ms[n]
        c = (max(loads) + 2000 * sum(1 for x in loads if x)) / 1000
        best = c if best is None else min(best, c)
    return best


per_test_gain = {}
for spec in sys.argv[3:]:
    name, ans, sel = spec.split(":")
    answers = json.load(open(ans))["commits"]
    tot_old = tot_new = tot_cpu = 0.0
    n = nfail = 0
    for line in open(sel):
        r = json.loads(line)
        nums = [x for w in r["workers"] for x in w]
        s = set(nums)
        if TPCDS in s or not (s & HEAVY):
            continue
        fail = set(answers[r["commit"]]["failing"])
        n += 1
        nfail += bool(fail)
        keep = [x for x in nums if x not in HEAVY or x in fail]
        old, new = cost(r["workers"]), best_pack(keep)
        hv = [x for x in nums if x in HEAVY]
        classes = {by[x]["id"].split("::")[1] for x in hv}
        cpu = 0.05 + sum(setup.get(c, 0) for c in classes) + sum(inl.get(by[x]["id"], 0) for x in hv)
        tot_old += old
        tot_new += new
        tot_cpu += cpu
        for x in hv:
            if x not in fail:
                per_test_gain.setdefault(by[x]["id"], [0, 0])
                per_test_gain[by[x]["id"]][0] += 1
    print(f"{name}: class H {n} commits ({nfail} really failing): sim {tot_old:.1f} -> {tot_new:.1f} "
          f"(-{tot_old - tot_new:.1f}), verify CPU ~{tot_cpu:.1f}, net -{tot_old - tot_new - tot_cpu:.1f}")

# per-test value: drop x alone (when it really passes) from each class-H selection that holds it
import os
if os.environ.get("PERTEST"):
    val = {}
    for spec in sys.argv[3:]:
        name, ans, sel = spec.split(":")
        answers = json.load(open(ans))["commits"]
        for line in open(sel):
            r = json.loads(line)
            nums = [x for w in r["workers"] for x in w]
            s = set(nums)
            if TPCDS in s or not (s & HEAVY):
                continue
            fail = set(answers[r["commit"]]["failing"])
            base = best_pack(nums)
            for x in s & HEAVY:
                tid = by[x]["id"]
                v = val.setdefault(tid, {"sel": 0, "pass": 0, "gain": 0.0})
                v["sel"] += 1
                if x not in fail:
                    v["pass"] += 1
                    v["gain"] += base - best_pack([y for y in nums if y != x])
    rows = []
    for tid, v in val.items():
        cls = tid.split("::")[1]
        c1 = inl.get(tid, 0) + 0.05
        rows.append((v["gain"] - v["sel"] * c1, tid, v, c1, setup.get(cls, 0)))
    rows.sort(reverse=True)
    print(f"{'test':55s} sel pass  gain  cpu/test setup  net(no setup)")
    for net, tid, v, c1, su in rows[:25]:
        print(f"{tid.split('::',1)[1][:55]:55s} {v['sel']:3d} {v['pass']:4d} {v['gain']:6.1f} {c1:6.2f} {su:5.2f} {net:7.1f}")
    print("tests with positive net:", sum(1 for r in rows if r[0] > 0), "of", len(rows))
