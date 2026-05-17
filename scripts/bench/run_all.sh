#!/usr/bin/env bash
# Driver: csynth each PL kernel, copy the top-level synth report into scripts/bench/reports/
# Must be run from repo root.

set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

KERNELS=(
  "embed_ffn:embed_ffn_top"
  "pairwise_mlp:pairwise_mlp_top"
  "attn_obj:attn_block_obj_top"
  "attn_cand:attn_block_cand_top"
  "attn_cross:attn_block_cross_top"
  "cand_build:candidate_build_top"
  "cand_lorentz:cand_lorentz_top"
  "autoencoder:dual_autoencoder_top"
)

mkdir -p scripts/bench/reports
mkdir -p scripts/bench/logs

for entry in "${KERNELS[@]}"; do
  name="${entry%%:*}"
  top="${entry##*:}"
  echo "=== [$name] csynth (top=$top) ==="
  tcl="scripts/bench/bench_${name}.tcl"
  log="scripts/bench/logs/${name}.log"
  proj="bench_${name}_proj"

  if [[ ! -f "$tcl" ]]; then
    echo "missing $tcl, skipping"; continue
  fi

  vitis_hls -f "$tcl" > "$log" 2>&1
  rc=$?
  rpt="${proj}/solution1/syn/report/${top}_csynth.rpt"
  if [[ -f "$rpt" ]]; then
    cp "$rpt" "scripts/bench/reports/${name}_${top}_csynth.rpt"
    echo "  -> report: scripts/bench/reports/${name}_${top}_csynth.rpt"
  else
    echo "  !! no report produced (rc=$rc), check $log"
  fi
done

echo
echo "Done. Summary:"
ls scripts/bench/reports/
