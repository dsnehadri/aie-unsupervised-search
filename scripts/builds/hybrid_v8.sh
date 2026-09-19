#!/bin/bash
# v6 plus dynamic row counts and the table softmax on the array, and the
# single-pass norm, wide reads and early four-vectors in fabric: 37.4-37.8 us at
# 6-8 jets, 39.4-39.7 us at 10-12 jets, 4.4 us/event, AUC 0.9828, WNS +0.563 ns.
# Latency depends on jet multiplicity; quote the 12-jet worst case.
cd "$(dirname "$(readlink -f "$0")")"
./build_hybrid.sh "${1:-/home/snehadri/aie_hybrid_v8}" 133000000 link_chain_pair.cfg \
  "-DAIE_PLACE -DCHAIN_STREAM -DEMBED_PIPE -DHEAD_STREAM -DLN_CLZ -DPAIR_L0_WINDOW -DPAIRWISE_ON_AIE -DPOST_SPLIT_C -DPOST_STREAM -DPRE_STREAM -DPRE_STREAM_CROSS -DROWS_DYN -DSOFTMAX_LUT -DWIJ_ONE_PORT -DWIJ_PAD16" \
  "-DNARROW_MUL -DLORENTZ_PIPE -DLN_MODE=7 -DLN_SQ_FABRIC -DWIJ_ONE_PORT -DLORENTZ_WIDE -DWIJ_PAD16 -DWEIGHTS_LOCAL -DPAIRWISE_ON_AIE -DLORENTZ_WIDE_IN -DREAD_WIDE -DP4_EARLY"
