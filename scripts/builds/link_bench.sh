#!/bin/bash
# AIE-PL link benchmark (Figure 4): a pass-through array kernel plus three PL
# bridges -- 64-bit at 100 MHz, 128-bit at 250 MHz, and four 128-bit PLIOs at
# 250 MHz (from DDR, and with the payload made in the PL). No -D flags.
# Measured: 800 MB/s, 4 GB/s per channel, 8.3 GB/s aggregate from DDR,
# 16 GB/s without it; crossing cost <= 1 us.
set -e
REPO=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
D=${1:-/home/snehadri/aie_pt_link}; mkdir -p "$D"; D=$(cd "$D" && pwd)
VITIS=${VITIS_ROOT:-/code/Xilinx_2025.2/2025.2/Vitis}
P=$VITIS/base_platforms/xilinx_vck190_base_202520_1/xilinx_vck190_base_202520_1.xpfm
source $VITIS/settings64.sh
cat > "$D/build_flags.txt" <<F
design     AIE-PL link benchmark (Figure 4)
built      $(date -Is)
repo       $REPO @ $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)
aie_graph  src/link_bench/pt_hw_main.cc
pl_tops    src/link_bench/{lb_top,p128_top,q512_top}.cpp
clock_hz   100000000 (64-bit) / 250000000 (128-bit and 4x128-bit)
cflags     none
F
cat "$D/build_flags.txt"
cp "$REPO/src/link_bench/"*.cc "$REPO/src/link_bench/"*.cpp "$REPO/src/link_bench/link.cfg" "$D/"
cd "$D"; rm -rf Work_hw_pt libadf.a
aiecompiler --target=hw --platform=$P --stacksize=2048 --workdir=Work_hw_pt --include=. \
  pt_hw_main.cc --output-archive=libadf.a > "$D/aie_build.log" 2>&1
for k in lb_top p128_top q512_top; do
  v++ -c -t hw --platform $P -k $k --save-temps --temp_dir ./_x_$k -o $k.xo $k.cpp
done
v++ -l -t hw --platform $P --config link.cfg --save-temps --temp_dir ./_x_link \
    -o pt_link.xsa lb_top.xo p128_top.xo q512_top.xo libadf.a
v++ -p -t hw --platform $P --package.out_dir ./package -o pt_link.xclbin pt_link.xsa libadf.a
md5sum package/BOOT.BIN | tee -a "$D/build_flags.txt"
