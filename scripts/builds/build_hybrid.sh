#!/bin/bash
# AIE-PL hybrid build driver: compile the array graph into libadf.a, compile the
# fabric kernel into a .xo, link both at the given clock, package BOOT.BIN.
#
#   build_hybrid.sh <build_dir> <freq_hz> <link.cfg> "<AIE -D flags>" "<PL -D flags>"
#
# Both flag sets are written to <build_dir>/build_flags.txt before anything is
# compiled. Earlier hybrids took their array flags from AIE_XPRE in the calling
# shell and recorded nothing; do not reintroduce that.
set -e
D=$1; FREQ=$2; CFG=$3; AIE_FLAGS="$4"; PL_FLAGS="$5"
[ -n "$D" ] && [ -n "$FREQ" ] && [ -n "$CFG" ] && [ -n "$AIE_FLAGS" ] && [ -n "$PL_FLAGS" ] \
  || { echo "usage: $0 <build_dir> <freq_hz> <link.cfg> \"<AIE flags>\" \"<PL flags>\"" >&2; exit 1; }
REPO=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
VITIS=${VITIS_ROOT:-/code/Xilinx_2025.2/2025.2/Vitis}
P=$VITIS/base_platforms/xilinx_vck190_base_202520_1/xilinx_vck190_base_202520_1.xpfm
source $VITIS/settings64.sh

mkdir -p "$D"; D=$(cd "$D" && pwd)
[ -f "$CFG" ] || CFG="$REPO/src/aie_stream/pl/$CFG"
cat > "$D/build_flags.txt" <<EOF
design     $(basename "$D")
built      $(date -Is)
repo       $REPO @ $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)
aie_graph  src/aie_stream/tb/chain_hw_main.cc
pl_top     src/aie_stream/pl/aie_stream_top_chain.cpp
clock_hz   $FREQ
link_cfg   $CFG
aie_flags  $AIE_FLAGS
pl_flags   $PL_FLAGS
toolchain  $VITIS
EOF
echo "=== flags recorded in $D/build_flags.txt"; cat "$D/build_flags.txt"

echo "=== [1/4] array graph $(date +%T) ==="
cd "$REPO/src/aie_stream/tb"
rm -rf "Work_$(basename "$D")"
aiecompiler --target=hw --platform=$P --stacksize=${AIE_STACK:-4096} \
  --workdir="Work_$(basename "$D")" --include=. --include=kernels \
  $(for d in $AIE_FLAGS; do echo --Xpreproc=$d; done) \
  chain_hw_main.cc --output-archive="$D/libadf.a" > "$D/aie_build.log" 2>&1
grep -c "cores" "$D/aie_build.log" >/dev/null && grep -oE "[0-9]+ cores" "$D/aie_build.log" | tail -1

echo "=== [2/4] fabric kernel $(date +%T) ==="
cp "$REPO/src/aie_stream/pl/aie_stream_top_chain.cpp" "$D/aie_stream_top.cpp"
cd "$D"; rm -rf _x _x_link aie_stream.xsa aie_stream.xclbin package
v++ -c --save-temps -t hw --platform $P -k aie_stream_top $PL_FLAGS \
    --temp_dir $D/_x -I$D -I$REPO/src -o $D/aie_stream_top.xo $D/aie_stream_top.cpp

echo "=== [3/4] link @ ${FREQ} Hz $(date +%T) ==="
v++ -l -t hw --platform $P --config "$CFG" --save-temps \
    --clock.defaultFreqHz $FREQ --temp_dir ./_x_link \
    -o aie_stream.xsa aie_stream_top.xo libadf.a
R=$(find $D/_x_link -name "*route*timing_summary*.rpt" 2>/dev/null | head -1)
[ -n "$R" ] && grep -a -m1 -A6 "Design Timing Summary" "$R" | tail -1 | awk '{print "  post-route WNS " $1 " ns, failing endpoints " $3}'

echo "=== [4/4] package $(date +%T) ==="
v++ -p -t hw --platform $P --package.out_dir ./package -o aie_stream.xclbin aie_stream.xsa libadf.a
md5sum package/BOOT.BIN | tee -a "$D/build_flags.txt"
strings package/BOOT.BIN | grep -qm1 "Version=2025.2" && echo "GATE_OK 2025.2" || echo "REFUSE: not a 2025.2 image"
echo "HW_ALL_DONE"
