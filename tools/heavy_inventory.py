"""Build work/heavy_inventory.json from the inline measurements (see work/HANDOFF.md "heavy-verify").

usage: python tools/heavy_inventory.py SQLGLOT_SRC tests.json pristine.jsonl OUT.json [mutants.jsonl ...]

pristine.jsonl / mutants.jsonl come from the in-image harness (heavy_inv.py, kept with the handoff notes):
one record per heavy test run inline in a forked warm process (pristine), and per labelled mutant the
heavy tests whose inline verdict disagrees with the real suite's labels.
Static columns: whether the test (or a self.<method> it calls, transitively) submits to the class's
process pool (check_file / _mp_execute / self.pool) or reads other setUpClass state.
"""
import ast
import json
import sys

src, tests_json, prist, out = sys.argv[1:5]
muts = sys.argv[5:]
POOL = {"pool", "_mp_execute"}
STATE = {"conn", "tpch_conn", "tpcds_conn", "tpch_tables", "tpcds_tables", "cache", "tpch_sqls", "tpcds_sqls",
         "cached_execute"}


def method_facts(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    facts = {}
    for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
        meths = {f.name: f for f in cls.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))}
        direct = {}
        for name, f in meths.items():
            attrs = {n.attr for n in ast.walk(f) if isinstance(n, ast.Attribute)
                     and isinstance(n.value, ast.Name) and n.value.id in ("self", "cls")}
            direct[name] = attrs
        for name in meths:
            seen, todo, attrs = set(), [name], set()
            while todo:
                m = todo.pop()
                if m in seen or m not in direct:
                    continue
                seen.add(m)
                attrs |= direct[m]
                todo.extend(a for a in direct[m] if a in meths)
            facts[f"{cls.name}::{name}"] = {
                "uses_pool": bool(attrs & POOL),
                "uses_class_state": sorted(attrs & STATE),
                "helpers": sorted(a for a in attrs if a in meths and a != name),
            }
    return facts


facts = {}
for f in ("tests/test_optimizer.py", "tests/test_executor.py"):
    for k, v in method_facts(f"{src}/{f}").items():
        facts[f"{f}::{k}"] = v

tests = {t["id"]: t for t in json.load(open(tests_json))["tests"]}
recs = [json.loads(l) for l in open(prist)]
head = recs[0]
per = {r["id"]: r for r in recs if "id" in r}
files = [r for r in recs if "file" in r]
wrong = {}
n_mut = 0
for m in muts:
    for line in open(m):
        r = json.loads(line)
        n_mut += 1
        for i in r.get("wrong_pass", []):
            wrong.setdefault(i, {"wrong_pass": [], "wrong_fail": []})["wrong_pass"].append(r["c"])
        for i in r.get("wrong_fail", []):
            wrong.setdefault(i, {"wrong_pass": [], "wrong_fail": []})["wrong_fail"].append(r["c"])
inv = {"_about": {
    "what": "heavy tests (tests/test_optimizer.py, tests/test_executor.py minus test_execute_tpcds) run inline in a forked "
            "warm process (tools/heavy_verify.py), real image, --cpus 2",
    "zygote_prepare_cpu": head["prepare_cpu"], "class_setup_cpu": head["setup_cpu"],
    "whole_file_inline": files, "mutants_checked": n_mut,
    "cpu": "inline CPU of the test alone (child CPU minus its class's setUpClass CPU, floored at 0.03; noisy at the "
           "0.1 s level)"}, "tests": {}}
for tid, r in per.items():
    t = tests[tid]
    key = tid
    cpu = max(0.03, r["cpu"])
    inv["tests"][tid] = {
        "number": t["number"], "sim_s": t["seconds"], **facts.get(key, {}),
        "inline_pristine": r["result"], "inline_cpu_s": round(cpu, 3), "inline_cpu_with_setup_s": r["cpu_with_setup"],
        "cpu_over_sim": round(cpu / t["seconds"], 2) if t["seconds"] else None,
        "mutants_wrong_pass": wrong.get(tid, {}).get("wrong_pass", []),
        "mutants_wrong_fail": wrong.get(tid, {}).get("wrong_fail", []),
    }
json.dump(inv, open(out, "w"), indent=1)
print("tests", len(inv["tests"]), "pool users", sum(1 for v in inv["tests"].values() if v.get("uses_pool")))
