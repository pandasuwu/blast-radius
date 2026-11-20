"""Blast Radius selector, v1: run every test, packed well.

Packs the selected tests with LPT (longest first onto the lightest worker) for
every worker count 1..8 and keeps the count with the lowest cost
(max load + 2 s per non-empty worker). All arithmetic is in whole milliseconds,
like the scorer.
"""

import json
import sys
from pathlib import Path

DATA = Path("/data")
MAX_WORKERS = 8
WORKER_COST_MS = 2000


def lpt(items: list[tuple[int, int]], k: int) -> tuple[int, list[list[int]]]:
    """items are (ms, number), sorted longest first. Returns (cost, workers)."""
    workers: list[list[int]] = [[] for _ in range(k)]
    loads = [0] * k
    for ms, number in items:
        i = loads.index(min(loads))
        workers[i].append(number)
        loads[i] += ms
    workers = [w for w in workers if w]
    return max(loads) + WORKER_COST_MS * len(workers), workers


def pack(numbers, durations: list[int]) -> list[list[int]]:
    items = sorted(((durations[n], n) for n in set(numbers)), reverse=True)
    if not items:
        return []
    best = None
    for k in range(1, min(MAX_WORKERS, len(items)) + 1):
        cost, workers = lpt(items, k)
        if best is None or cost < best[0]:
            best = (cost, workers)
    return best[1]


def main() -> None:
    tests = json.loads((DATA / "tests.json").read_text())["tests"]
    durations = [0] * len(tests)
    for t in tests:
        durations[t["number"]] = round(t["seconds"] * 1000)
    everything = pack(range(len(tests)), durations)
    out = []
    for patch in sorted((DATA / "commits").glob("*.patch")):
        out.append(json.dumps({"commit": patch.stem, "workers": everything}))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))
    print(f"v1: {len(everything)} workers, all {len(tests)} tests", file=sys.stderr)


if __name__ == "__main__":
    main()
