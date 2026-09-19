#!/bin/bash
# v5 plus the pairwise bias MLP on the array (48 tiles) and wide Lorentz reads.
# Linked at 133 MHz this is v6f, the hybrid measured in the paper: 42.4 us one
# event, 4.2 us/event (236,759 event/s), AUC 0.9828, 140 tiles, WNS +0.559 ns.
cd "$(dirname "$(readlink -f "$0")")"
./build_hybrid.sh "${1:-/home/snehadri/aie_hybrid_v6_133}" 133000000 link_chain_pair.cfg \
  "-DAIE_PLACE -DCHAIN_STREAM -DEMBED_PIPE -DHEAD_STREAM -DLN_CLZ -DPAIR_L0_WINDOW -DPAIRWISE_ON_AIE -DPOST_SPLIT_C -DPOST_STREAM -DPRE_STREAM -DPRE_STREAM_CROSS -DWIJ_ONE_PORT -DWIJ_PAD16" \
  "-DNARROW_MUL -DLORENTZ_PIPE -DLN_MODE=6 -DWIJ_ONE_PORT -DLORENTZ_WIDE -DWIJ_PAD16 -DWEIGHTS_LOCAL -DPAIRWISE_ON_AIE -DLORENTZ_WIDE_IN"
