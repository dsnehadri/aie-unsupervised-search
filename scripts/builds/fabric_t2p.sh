#!/bin/bash
# t2n's logic linked at 170 MHz: 40.3 us one event, 8.74 us/event,
# WNS +0.312 ns. The design is clock-limited, so the gain is the clock ratio.
cd "$(dirname "$(readlink -f "$0")")"
./build_fabric.sh "${1:-/home/snehadri/plstream_t2p}" 170000000 \
  "-DPAIRWISE_PL_LOWDSP -DOBJ_DATAFLOW -DHEADS_BATCHED -DLIN_FABRIC_MUL -DLN_MODE=2 -DNARROW_MUL -DRESHAPE_FAST -DSOFTMAX_PIPE -DWIDE_STREAMS -DPAIRWISE_FAST -DRESHAPE_UNROLL -DLIN_J_UNROLL=4"
