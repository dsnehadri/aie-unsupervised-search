#!/bin/bash
# Dynamic row count, single-pass layernorm variance, wide input reads and early
# four-vectors, at 156.25 MHz: 40.3 us one event, 8.93 us/event, WNS +0.250 ns,
# 705,833 LUT (78.4%), 1,709 DSP (86.8%). Same latency as t2p by a different
# route -- t2p gains it from the clock, this from the logic.
cd "$(dirname "$(readlink -f "$0")")"
./build_fabric.sh "${1:-/home/snehadri/plstream_t2r}" 156250000 \
  "-DPAIRWISE_PL_LOWDSP -DOBJ_DATAFLOW -DHEADS_BATCHED -DLIN_FABRIC_MUL -DLN_MODE=7 -DNARROW_MUL -DRESHAPE_FAST -DSOFTMAX_PIPE -DWIDE_STREAMS -DPAIRWISE_FAST -DRESHAPE_UNROLL -DLIN_J_UNROLL=4 -DREAD_WIDE -DP4_EARLY -DROWS_DYN"
