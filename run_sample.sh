#!/usr/bin/env bash
# `make sample` without make: build, run under the scorer's flags, score.
# The scorer itself runs in the same python:3.13-slim image (no host Python needed).
set -euo pipefail
cd "$(dirname "$0")"
IMAGE=blast-radius-entry
ROOT="$(pwd -W 2>/dev/null || pwd)"
export MSYS_NO_PATHCONV=1
docker build -q --platform linux/amd64 --tag "$IMAGE" . >&2
docker run --rm --platform linux/amd64 --network none --cpus 2 \
  --memory 4096m --memory-swap 4096m --pids-limit 256 --read-only \
  --tmpfs /tmp:rw,size=64m,mode=1777 --cap-drop ALL \
  --security-opt no-new-privileges --user 65534:65534 --env HOME=/tmp \
  --volume "$ROOT/sample/commits:/data/commits:ro" \
  --volume "$ROOT/sample/tests.json:/data/tests.json:ro" \
  "$IMAGE" > output.jsonl
docker run --rm -v "$ROOT:/w" -w /w \
  python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 \
  python score.py sample sample/answers.json output.jsonl
