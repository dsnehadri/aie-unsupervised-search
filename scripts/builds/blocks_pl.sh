#!/bin/bash
# The same three attention blocks in fabric (Figure 7, left, red bars):
# 6.54 / 2.0 / 4.8 us per event at 156.25 MHz. Measured with host_block_sweep.
set -e
REPO=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
D=${1:-/home/snehadri/pl_attn_v5}; mkdir -p "$D"; D=$(cd "$D" && pwd)
VITIS=${VITIS_ROOT:-/code/Xilinx_2025.2/2025.2/Vitis}
P=$VITIS/base_platforms/xilinx_vck190_base_202520_1/xilinx_vck190_base_202520_1.xpfm
F="-DOBJ_DATAFLOW -DHEADS_BATCHED -DLIN_FABRIC_MUL -DLN_MODE=2 -DNARROW_MUL -DRESHAPE_FAST -DSOFTMAX_PIPE -DWIDE_STREAMS -DPAIRWISE_FAST -DPAIRWISE_PL_LOWDSP -DRESHAPE_UNROLL -DLIN_J_UNROLL=4"
source $VITIS/settings64.sh
cat > "$D/build_flags.txt" <<G
design     isolated blocks in fabric (Figure 7 left)
built      $(date -Is)
repo       $REPO @ $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)
top        src/attn_block_pl/pl_attn.cpp (objp_top, candp_top, crossp_top)
clock_hz   156250000
cflags     $F
G
cat "$D/build_flags.txt"
cp "$REPO/src/attn_block_pl/pl_attn.cpp" "$D/"
cd "$D"
for k in objp_top candp_top crossp_top; do
  v++ -c -t hw --platform $P -k $k $F --temp_dir ./_x_$k -I"$REPO/src" -o $k.xo pl_attn.cpp
done
v++ -l -t hw --platform $P --config "$REPO/src/attn_block_pl/link_pl_attn.cfg" --save-temps \
    --clock.defaultFreqHz 156250000 --temp_dir ./_x_link -o pl_attn.xsa objp_top.xo candp_top.xo crossp_top.xo
v++ -p -t hw --platform $P --package.out_dir ./package -o pl_attn.xclbin pl_attn.xsa
md5sum package/BOOT.BIN | tee -a "$D/build_flags.txt"
