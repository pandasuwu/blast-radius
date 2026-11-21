"""Which selected tests are not failing, what they cost, and which rule selected them.

usage (br-tools, repo at /w, BR_SRC=/w/work/sqlglot, BR_INDEX=<prepped index>):
    python eval/audit_extras.py <set> <data dir> <answers.json or labels> <cur log> [out.json]
For every commit: the required selection (Selector.select, plus the TPC-DS decision and the
verifier result read from the production log; no free riders), the failing tests, and for each
non-failing selected test its marginal cost (cost with it - cost without it). Extra cost is
charged to the rule that brought the test in, found by re-running select() and reading `why`.
"""
import collections
import json
import re
import sys

sys.path.insert(0, "/w")
import selector as S  # noqa: E402

name, data, answers, cur_log = sys.argv[1:5]
tests = json.load(open(f"{data}/tests.json"))["tests"]
dur = [round(t["seconds"] * 1000) for t in tests]
ids = [t["id"] for t in tests]
num = {t["id"]: t["number"] for t in tests}
if answers.endswith(".json"):
    ans = {c: set(a["failing"]) for c, a in json.load(open(answers))["commits"].items()}
else:  # mutU labels: only TPC-DS is labelled -> skip non-TPC-DS accounting
    ans = None
logs = collections.defaultdict(str)
for line in open(cur_log, encoding="utf-8", errors="replace"):
    logs[line.split(" ", 1)[0]] += line
idx = S.load_index()
sel = S.Selector(idx)
to_number = [num.get(t) for t in idx["tests"]]
TP = num["tests/test_executor.py::TestExecutor::test_execute_tpcds"]


def cost(nums):
    ws = S.pack(sorted(nums), dur)
    return S.plan_cost(ws, dur)


def numbers(mask):
    out, i = set(), 0
    while mask:
        if mask & 1 and to_number[i] is not None:
            out.add(to_number[i])
        mask >>= 1
        i += 1
    return out


by_rule = collections.Counter()
by_rule_n = collections.Counter()
examples = collections.defaultdict(list)
tot = {"commits": 0, "ours": 0, "oracle": 0, "extra": 0, "nofail_cost": 0}
for c in sorted(ans):
    text = open(f"{data}/commits/{c}.patch", encoding="utf-8").read()
    lg = logs.get(c, "")
    try:
        mask, why = sel.select(text)
        req = numbers(mask)
    except S.Everything as e:
        req, why = set(range(len(tests))), [f"everything: {e}"]
    if "test_execute_tpcds left out" in lg:
        req.discard(TP)
    if "all pass, nothing to run" in lg:
        req = set()
    fail = ans[c]
    ours, orc = cost(req), cost(fail) if fail else 0
    tot["commits"] += 1
    tot["ours"] += ours
    tot["oracle"] += orc
    tot["extra"] += ours - orc
    if not fail:
        tot["nofail_cost"] += ours
    if ours <= orc:
        continue
    # which extra tests drive the cost: drop each non-failing test and see what it saves
    extras = sorted(req - fail, key=lambda n: -dur[n])
    drivers = []
    base = ours
    for n in extras[:12]:
        d = base - cost(req - {n})
        if d > 0:
            drivers.append((n, d))
    rule = "; ".join(why)[:160]
    key = re.sub(r"\[[^\]]*\]|'[^']*'|\d+", "#", why[0] if why else "?")[:70]
    by_rule[key] += ours - orc
    by_rule_n[key] += 1
    if len(examples[key]) < 3:
        examples[key].append({"commit": c, "extra_s": round((ours - orc) / 1000, 2), "nfail": len(fail),
                              "why": rule, "drivers": [(ids[n].split("::")[-1], round(d / 1000, 2))
                                                       for n, d in drivers[:4]]})
print(json.dumps({k: round(v / 1000, 1) for k, v in tot.items() if k != "commits"} | {"commits": tot["commits"]}))
for k, v in by_rule.most_common(14):
    print(f"{v / 1000:8.1f} s  {by_rule_n[k]:4d} commits  {k}")
    for ex in examples[k][:2]:
        print("            e.g.", ex["commit"], ex["extra_s"], "s extra,", ex["nfail"], "failing; drivers", ex["drivers"])
