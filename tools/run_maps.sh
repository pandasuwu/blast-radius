#!/usr/bin/env bash
# Run inside br-tools: full suite under coverage + mapplugin. Output in /w/work/out.
set -eu
rm -rf /src && mkdir -p /src /w/work/out && cp -r /w/work/sqlglot /src/sqlglot
cd /src/sqlglot
export MAP_OUT=/w/work/out PYTHONPATH=/src/sqlglot:/w/tools SKIP_INTEGRATION=0
sed -i 's#data_file = .*#data_file = /w/work/out/.coverage#' /w/tools/coveragerc
python -m coverage run --rcfile=/w/tools/coveragerc -m pytest -p mapplugin -p no:cacheprovider tests -q --no-header -rfE 2>&1 | tail -30
cd /w && python tools/build_maps.py work/out sample/tests.json maps/maps.json.gz
