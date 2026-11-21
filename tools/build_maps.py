"""Compact the coverage + runtime logs into maps/maps.json.gz (runs in br-tools).

usage: python build_maps.py <out_dir> <tests.json> <maps.json.gz>
"""
import gzip, json, sys
from collections import defaultdict
import coverage

out_dir, tests_json, dest = sys.argv[1:4]
SRC = "/src/sqlglot/"
tests = [t["id"] for t in json.load(open(tests_json))["tests"]]
idx = {t: i for i, t in enumerate(tests)}


def owner(nodeid):  # class (or file) a test belongs to
    return nodeid.rsplit("::", 1)[0]


by_owner, by_file = defaultdict(int), defaultdict(int)
for t, i in idx.items():
    by_owner[owner(t)] |= 1 << i
    by_file[t.split("::")[0]] |= 1 << i

unknown = set()


def ctx_mask(ctx):
    """'' or '|file' -> None (import time); S|id -> whole class; C|id -> the test."""
    if not ctx or ctx.startswith("|"):
        return None
    kind, nodeid = ctx.split("|", 1)
    nodeid = nodeid.split("[", 1)[0] if nodeid not in idx else nodeid
    if kind == "C":
        if nodeid in idx:
            return 1 << idx[nodeid]
        unknown.add(nodeid)
        return 0
    return by_owner.get(owner(nodeid), 0) or by_file.get(nodeid.split("::")[0], 0)


sets, set_ix = [], {}


def sid(mask):
    if mask not in set_ix:
        set_ix[mask] = len(sets)
        sets.append(mask)
    return set_ix[mask]


cov = coverage.CoverageData(basename=f"{out_dir}/.coverage")
cov.read()
lines, importlines = {}, {}
seen_ctx = set()
for f in cov.measured_files():
    rel = f[len(SRC):] if f.startswith(SRC) else f
    byl = cov.contexts_by_lineno(f)
    lm, il = {}, []
    for ln, ctxs in byl.items():
        m = 0
        for c in ctxs:
            seen_ctx.add(c)
            cm = ctx_mask(c)
            if cm is None:
                il.append(ln)
            else:
                m |= cm
        if m:
            lm[ln] = sid(m)
    lines[rel] = lm
    importlines[rel] = sorted(il)

rt = json.load(open(f"{out_dir}/runtime.json"))


def invert(d, file_ctx_ok):
    acc = defaultdict(int)
    for ctx, items in d.items():
        if ctx.startswith("|"):
            m = by_file.get(ctx[1:], 0) if file_ctx_ok else 0
        else:
            m = ctx_mask(ctx) or 0
        for it in items:
            acc[it] |= m
    return {k: sid(v) for k, v in acc.items() if v}


opens = invert(rt["opens"], True)
# opens split by phase: during a test's own call (C) vs during class setup (S -> owners)
opens_c, opens_s = defaultdict(int), defaultdict(set)
for ctx, items in rt["opens"].items():
    if ctx.startswith("C|") and ctx[2:] in idx:
        for it in items:
            opens_c[it] |= 1 << idx[ctx[2:]]
    elif ctx.startswith("S|"):
        for it in items:
            opens_s[it].add(owner(ctx[2:]))
    elif ctx.startswith("|"):
        for it in items:
            opens_c[it] |= by_file.get(ctx[1:], 0)
opens_c = {k: sid(v) for k, v in opens_c.items() if v}
opens_s = {k: sorted(v) for k, v in opens_s.items()}
attr_reads = defaultdict(lambda: defaultdict(int))
for ctx, pairs in rt.get("attr_reads", {}).items():
    cm = ctx_mask(ctx)
    if cm is None:
        continue
    for own, attr in pairs:
        attr_reads[own][attr] |= cm
attr_reads = {o: {a: sid(m) for a, m in d.items()} for o, d in attr_reads.items()}
field_reads = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
for ctx, triples in rt.get("field_reads", {}).items():
    cm = ctx_mask(ctx)
    if cm is None:
        continue
    for own, attr, field in triples:
        field_reads[own][attr][field] |= cm
field_reads = {o: {a: {str(f): sid(m) for f, m in fd.items()} for a, fd in d.items()} for o, d in field_reads.items()}
code_opens = invert(rt["code_opens"], True)
classes = invert(rt["classes"], False)
testfiles = {f: sid(m) for f, m in by_file.items()}

# checks
have = {c.split("|", 1)[1] for c in seen_ctx if c.startswith("C|")}
missing = [t for t in tests if t not in have]
bad = {k: v for k, v in rt["outcomes"].items() if any(("failed" in o) for o in v)}
print("tests", len(tests), "contexts", len(seen_ctx), "missing ctx", len(missing), missing[:10])
print("unknown nodeids", len(unknown), sorted(unknown)[:10])
print("failed in baseline", len(bad), list(bad)[:10])
print("distinct sets", len(sets), "files", len(lines), "opens", len(opens), "classes", len(classes))

doc = {"tests": tests, "sets": [format(m, "x") for m in sets], "lines": lines,
       "importlines": importlines, "opens": opens, "opens_c": opens_c, "opens_s": opens_s, "attr_reads": attr_reads, "field_reads": field_reads, "code_opens": code_opens, "classes": classes, "testfiles": testfiles,
       "missing": missing}
with gzip.open(dest, "wt", compresslevel=9) as fh:
    json.dump(doc, fh, separators=(",", ":"))
