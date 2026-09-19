#!/bin/bash
# One object, candidate and cross attention block on the array, each fed by its
# own PL feeder (Figure 7, left, blue bars): 4.12 / 0.98 / 3.79 us per event.
# The constraints file keeps the graph off column 0, which the platform uses.
set -e
REPO=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
D=${1:-/home/snehadri/blocks3_v2}; mkdir -p "$D"; D=$(cd "$D" && pwd)
VITIS=${VITIS_ROOT:-/code/Xilinx_2025.2/2025.2/Vitis}
P=$VITIS/base_platforms/xilinx_vck190_base_202520_1/xilinx_vck190_base_202520_1.xpfm
AIE_FLAGS="-DPOST_STREAM -DPRE_STREAM -DPOST_SPLIT_C -DHEAD_STREAM -DPRE_STREAM_CROSS -DWIJ_PAD16 -DLN_CLZ"
source $VITIS/settings64.sh
cat > "$D/build_flags.txt" <<F
design     isolated blocks on the array (Figure 7 left)
built      $(date -Is)
repo       $REPO @ $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)
aie_graph  src/aie_stream/tb/blocks3_hw_main.cc
pl_tops    src/aie_stream/pl/blocks3_top.cpp (objb_top, candb_top, crossb_top)
constraint src/aie_stream/aie/no_col0.json
clock_hz   150000000
aie_flags  $AIE_FLAGS
F
cat "$D/build_flags.txt"
cd "$REPO/src/aie_stream/tb"; rm -rf Work_hw_blocks3
aiecompiler --target=hw --platform=$P --stacksize=4096 --workdir=Work_hw_blocks3 \
  --include=. --include=kernels --constraints="$REPO/src/aie_stream/aie/no_col0.json" \
  $(for d in $AIE_FLAGS; do echo --Xpreproc=$d; done) \
  blocks3_hw_main.cc --output-archive="$D/libadf.a" > "$D/aie_build.log" 2>&1
cp "$REPO/src/aie_stream/pl/blocks3_top.cpp" "$D/"
cd "$D"
for k in objb_top candb_top crossb_top; do
  v++ -c -t hw --platform $P -k $k --save-temps --temp_dir ./_x_$k -o $k.xo blocks3_top.cpp
done
v++ -l -t hw --platform $P --config "$REPO/src/aie_stream/pl/link_blocks3.cfg" --save-temps \
    --clock.defaultFreqHz 150000000 --temp_dir ./_x_link -o blocks3.xsa objb_top.xo candb_top.xo crossb_top.xo libadf.a
v++ -p -t hw --platform $P --package.out_dir ./package -o blocks3.xclbin blocks3.xsa libadf.a
md5sum package/BOOT.BIN | tee -a "$D/build_flags.txt"
