"""Evaluate 'verify only heavy tests in V' policies. usage: python policy.py pristine.jsonl tests.json set:ans:sel ..."""
import json, sys
exec(open(sys.argv[0].replace("heavy_policy.py", "heavy_saving.py")).read().split("per_test_gain = {}")[0])
inl = {r["id"]: max(0.03, r["cpu"]) for r in prist[1:] if "id" in r}
sim = {t["id"]: t["seconds"] for t in tests}
sets = []
for spec in sys.argv[3:]:
    name, ans, sel = spec.split(":")
    sets.append((name, json.load(open(ans))["commits"], [json.loads(l) for l in open(sel)]))
TOP9 = ["TestOptimizer::test_annotate_types", "TestExecutor::test_execute_tpch", "TestExecutor::test_agg_order",
        "TestOptimizer::test_merge_subqueries_copy_budget", "TestOptimizer::test_canonicalize", "TestOptimizer::test_normalize",
        "TestOptimizer::test_pushdown_projection", "TestExecutor::test_scalar_functions", "TestOptimizer::test_values_annotation"]
heavy_ids = [by[x]["id"] for x in HEAVY]
policies = {f"ratio<{q}": {i for i in heavy_ids if inl[i] < q * sim[i]} for q in (0.25, 0.5, 1.0)}
policies["top9"] = {i for i in heavy_ids if i.split("::", 1)[1] in TOP9}
policies["top9-noexec"] = {i for i in policies["top9"] if "Executor" not in i}
policies["all"] = set(heavy_ids)
for pname, V in policies.items():
    line = []
    for name, answers, sels in sets:
        dsim = cpu = 0.0
        for r in sels:
            nums = [x for w in r["workers"] for x in w]
            s = set(nums)
            if TPCDS in s or not (s & HEAVY):
                continue
            ver = [x for x in nums if x in HEAVY and by[x]["id"] in V]
            if not ver:
                continue
            fail = set(answers[r["commit"]]["failing"])
            keep = [x for x in nums if not (x in ver and x not in fail)]
            dsim += best_pack(nums) - best_pack(keep)
            cls = {by[x]["id"].split("::")[1] for x in ver}
            cpu += 0.05 + sum(setup[c] for c in cls) + sum(inl[by[x]["id"]] for x in ver)
        line.append(f"{name} -{dsim:.1f}sim +{cpu:.1f}cpu = {cpu - dsim:+.1f}")
    print(f"{pname:12s} |V|={len(V):3d}  " + " | ".join(line))
