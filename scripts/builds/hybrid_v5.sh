#!/bin/bash
# Whole ABC stack on the array, pairwise MLP still in fabric: 45.6 us one event,
# 5.0 us/event (201,326 event/s).
cd "$(dirname "$(readlink -f "$0")")"
./build_hybrid.sh "${1:-/home/snehadri/aie_hybrid_v5}" 125000000 link_chain_pair.cfg \
  "-DAIE_PLACE -DCHAIN_STREAM -DEMBED_PIPE -DHEAD_STREAM -DLN_CLZ -DPOST_SPLIT_C -DPOST_STREAM -DPRE_STREAM -DPRE_STREAM_CROSS -DWIJ_ONE_PORT -DWIJ_PAD16" \
  "-DNARROW_MUL -DLORENTZ_PIPE -DLN_MODE=6 -DWIJ_ONE_PORT -DLORENTZ_WIDE -DWIJ_PAD16 -DWEIGHTS_LOCAL"
