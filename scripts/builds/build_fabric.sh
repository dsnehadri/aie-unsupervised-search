#!/bin/bash
# Fabric-only (PL) build driver: compile the kernel, link at the given clock,
# package BOOT.BIN.
#
#   build_fabric.sh <build_dir> <freq_hz> "<-D flags>"
#
# The flags are written to <build_dir>/build_flags.txt before anything is
# compiled, so the finished build carries the recipe that produced it. Call this
# from one of the per-design scripts next to it rather than passing flags by
# hand.
set -e
D=$1; FREQ=$2; FLAGS="$3"
[ -n "$D" ] && [ -n "$FREQ" ] && [ -n "$FLAGS" ] || { echo "usage: $0 <build_dir> <freq_hz> \"<-D flags>\"" >&2; exit 1; }
REPO=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
VITIS=${VITIS_ROOT:-/code/Xilinx_2025.2/2025.2/Vitis}
P=$VITIS/base_platforms/xilinx_vck190_base_202520_1/xilinx_vck190_base_202520_1.xpfm
source $VITIS/settings64.sh

mkdir -p "$D"; D=$(cd "$D" && pwd)
cat > "$D/build_flags.txt" <<EOF
design     $(basename "$D")
built      $(date -Is)
repo       $REPO @ $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)
top        src/pl_stream/pl_stream_top.cpp
clock_hz   $FREQ
cflags     $FLAGS
toolchain  $VITIS
EOF
echo "=== flags recorded in $D/build_flags.txt"; cat "$D/build_flags.txt"

cp "$REPO/src/pl_stream/pl_stream_top.cpp" "$D/"
cd "$D"; rm -rf _x pl_stream.link.xsa pl_stream.xclbin package

echo "=== [1/3] kernel $(date +%T) ==="
v++ -c --save-temps -t hw --platform $P -k pl_stream_top $FLAGS \
    --temp_dir $D/_x -I$D -I$REPO/src -o $D/_x/pl_stream_top.xo $D/pl_stream_top.cpp
echo "=== [2/3] link @ ${FREQ} Hz $(date +%T) ==="
v++ -l --save-temps -t hw --platform $P --clock.freqHz ${FREQ}:pl_stream_top_1 \
    --temp_dir $D/_x -o $D/pl_stream.link.xsa $D/_x/pl_stream_top.xo
R=$(find $D/_x -name "*route_report_timing_summary*.rpt" | head -1)
[ -n "$R" ] && grep -aA3 "WNS(ns)" "$R" | tail -1 | awk '{print "  post-route WNS " $1 " ns, failing endpoints " $3}'
echo "=== [3/3] package $(date +%T) ==="
v++ -p -t hw --platform $P --package.out_dir ./package -o pl_stream.xclbin pl_stream.link.xsa
md5sum package/BOOT.BIN | tee -a "$D/build_flags.txt"
strings package/BOOT.BIN | grep -qm1 "Version=2025.2" && echo "GATE_OK 2025.2" || echo "REFUSE: not a 2025.2 image"
echo "HW_ALL_DONE"
