#!/usr/bin/env bash
# Fast loop: run selector.py from the working tree inside br-tools against sample/.
set -eu
cd /w
export BR_SRC=/w/work/sqlglot BR_INDEX=/tmp/index.pkl BR_DATA=${BR_DATA:-/w/sample}
python selector.py --prep
python - <<'PY'
import resource, subprocess, sys
with open("/w/work/out_local.jsonl", "w") as out, open("/w/work/sel_local.log", "w") as err:
    subprocess.run([sys.executable, "-I", "selector.py"], stdout=out, stderr=err)
r = resource.getrusage(resource.RUSAGE_CHILDREN)
print("selector cpu", round(r.ru_utime + r.ru_stime, 3))
PY
python tools/eval_sample.py sample sample/answers.json /w/work/out_local.jsonl
