#!/bin/bash
# 20 object-attention-block instances, 15 tiles each, every instance with its own
# feeder (Figure 7, right): 234,730 event/s at one instance to 4.11M at 20.
set -e
REPO=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
D=${1:-/home/snehadri/obj20_v6}; mkdir -p "$D"; D=$(cd "$D" && pwd)
VITIS=${VITIS_ROOT:-/code/Xilinx_2025.2/2025.2/Vitis}
P=$VITIS/base_platforms/xilinx_vck190_base_202520_1/xilinx_vck190_base_202520_1.xpfm
AIE_FLAGS="-DPOST_STREAM -DPRE_STREAM -DPOST_SPLIT_C -DHEAD_STREAM -DWIJ_PAD16 -DLN_CLZ"
source $VITIS/settings64.sh
cat > "$D/build_flags.txt" <<H
design     object-block tile-scaling sweep (Figure 7 right)
built      $(date -Is)
repo       $REPO @ $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)
aie_graph  src/aie_stream/tb/obj20_hw_main.cc
pl_top     src/aie_stream/pl/obj20_ind_top.cpp  (independent feeder per instance)
host       src/host_obj20_ind.cpp
clock_hz   150000000
aie_flags  $AIE_FLAGS
H
cat "$D/build_flags.txt"
cd "$REPO/src/aie_stream/tb"; rm -rf Work_hw_obj20
aiecompiler --target=hw --platform=$P --stacksize=4096 --workdir=Work_hw_obj20 \
  --include=. --include=kernels $(for d in $AIE_FLAGS; do echo --Xpreproc=$d; done) \
  obj20_hw_main.cc --output-archive="$D/libadf.a" > "$D/aie_build.log" 2>&1
cp "$REPO/src/aie_stream/pl/obj20_ind_top.cpp" "$D/obj24_top.cpp"
cd "$D"
v++ -c -t hw --platform $P -k obj24_top --save-temps --temp_dir ./_x_pl -o obj24_top.xo obj24_top.cpp
v++ -l -t hw --platform $P --config "$REPO/src/aie_stream/pl/link_obj20.cfg" --save-temps \
    --clock.defaultFreqHz 150000000 --temp_dir ./_x_link -o obj20.xsa obj24_top.xo libadf.a
v++ -p -t hw --platform $P --package.out_dir ./package -o obj20.xclbin obj20.xsa libadf.a
md5sum package/BOOT.BIN | tee -a "$D/build_flags.txt"
