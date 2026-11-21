"""Compact work/prof/profile.json into maps/prof.json.gz (runs in br-tools).

usage: python build_prof.py <prof_dir> <tests.json> <dest>
Per instrumented node (keyed "rel" -> "lineno,col,end_lineno,end_col"):
  [kind, op, [import-time signatures], [[signature, set_index], ...]]
plus "untrusted": tests that did not pass in the instrumented run (their profile is not used).
"""
import gzip, json, sys
from collections import defaultdict

prof_dir, tests_json, dest = sys.argv[1:4]
tests = [t["id"] for t in json.load(open(tests_json))["tests"]]
idx = {t: i for i, t in enumerate(tests)}
by_owner, by_file = defaultdict(int), defaultdict(int)
for t, i in idx.items():
    by_owner[t.rsplit("::", 1)[0]] |= 1 << i
    by_file[t.split("::")[0]] |= 1 << i

d = json.load(open(f"{prof_dir}/profile.json"))


def ctx_mask(ctx):
    if not ctx or ctx.startswith("|"):
        return None
    kind, nodeid = ctx.split("|", 1)
    if kind == "C":
        return 1 << idx[nodeid] if nodeid in idx else 0
    return by_owner.get(nodeid.rsplit("::", 1)[0], 0) or by_file.get(nodeid.split("::")[0], 0)


sets, set_ix = [], {}


def sid(mask):
    if mask not in set_ix:
        set_ix[mask] = len(sets)
        sets.append(mask)
    return set_ix[mask]


out = defaultdict(dict)
for nid, (rel, l, c, el, ec, kind, op) in enumerate(d["nodes"]):
    entry = d["prof"].get(str(nid), {})
    imp = []  # signatures seen at import / collection time
    sigs = []
    for sig, ctxs in entry.items():
        m = 0
        for ctx in ctxs:
            cm = ctx_mask(ctx)
            if cm is None:
                if sig not in imp:
                    imp.append(sig)
            else:
                m |= cm
        if m:
            sigs.append([sig, sid(m)])
    key = f"{l},{c},{el},{ec}"
    if key in out[rel]:  # same position instrumented twice (module imported twice): merge
        prev = out[rel][key]
        prev[2].extend(x for x in imp if x not in prev[2])
        prev[3].extend(sigs)
    else:
        out[rel][key] = [kind, op, imp, sigs]

untrusted = 0
passed = set()
for nodeid, outs in d["outcomes"].items():
    if any("failed" in o for o in outs):
        if nodeid in idx:
            untrusted |= 1 << idx[nodeid]
    else:
        passed.add(nodeid)
for t in tests:
    if t not in d["outcomes"]:
        untrusted |= 1 << idx[t]
print("nodes", len(d["nodes"]), "profiled", len(d["prof"]), "sets", len(sets),
      "untrusted", [tests[i] for i in range(len(tests)) if untrusted >> i & 1][:10])
with gzip.open(dest, "wt", compresslevel=9) as fh:
    json.dump({"tests": tests, "sets": [format(m, "x") for m in sets], "nodes": out,
               "untrusted": format(untrusted, "x"), "attrnames": d.get("attrnames", []), "identity_decorators": d.get("identity_decorators", [])}, fh, separators=(",", ":"))
