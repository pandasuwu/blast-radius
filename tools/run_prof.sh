#!/usr/bin/env bash
# Condition-profile run (br-tools): whole suite with instrumented sqlglot. Output: /w/work/prof
set -eu
rm -rf /src && mkdir -p /src /w/work/prof && cp -r /w/work/sqlglot /src/sqlglot
cd /src/sqlglot
export MAP_OUT=/w/work/prof PYTHONPATH=/src/sqlglot:/w/tools SKIP_INTEGRATION=0 PYTHONDONTWRITEBYTECODE=1
python -m pytest -p profplugin -p no:cacheprovider tests -q --no-header -rfE 2>&1 | tail -30
