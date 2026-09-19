#!/bin/bash
# PL-only design measured in the paper: 43.9 us one event, 9.3 us/event,
# 156.25 MHz, WNS +0.568 ns, 717,550 LUT (79.7%), 1,677 DSP (85.2%).
cd "$(dirname "$(readlink -f "$0")")"
./build_fabric.sh "${1:-/home/snehadri/plstream_t2n}" 156250000 \
  "-DPAIRWISE_PL_LOWDSP -DOBJ_DATAFLOW -DHEADS_BATCHED -DLIN_FABRIC_MUL -DLN_MODE=2 -DNARROW_MUL -DRESHAPE_FAST -DSOFTMAX_PIPE -DWIDE_STREAMS -DPAIRWISE_FAST -DRESHAPE_UNROLL -DLIN_J_UNROLL=4"
