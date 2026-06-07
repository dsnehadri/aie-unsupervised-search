#!/usr/bin/env bash
# refresh_charts.sh — pull current AIE timing from the live aiesim output and
# regenerate figs/. Safe to run any number of times while aiesim is in-flight.
#
# Usage:  scripts/refresh_charts.sh

set -e

REPO="$(cd "$(dirname "$0")/.." && pwd)"
TB="$REPO/src/aie_stream/tb"
PHASE3="/home/snehadri/repos/unsupervised-search/phase3_export"
TIMING_JSON="/tmp/aie_timing.json"

if [ ! -d "$TB/aiesimulator_output/data" ]; then
    echo "no aiesim output dir yet at $TB/aiesimulator_output/data"
    exit 1
fi

echo "=== refreshing AIE timing from partial aiesim output ==="
cd "$TB"
python3 check_attn_outputs.py \
    --phase3 "$PHASE3" \
    --output-dir aiesimulator_output/data \
    --tol 0.1 \
    --timing-out "$TIMING_JSON" \
    2>&1 | tail -8 || true

echo
echo "=== regenerating figs/ ==="
cd "$REPO"
python3 scripts/make_perf_charts.py 2>&1 | tail -10
