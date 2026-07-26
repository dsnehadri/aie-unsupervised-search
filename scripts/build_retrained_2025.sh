#!/bin/bash
# Retrained pl_stream BOOT.BIN, built with the PROVEN 2025.2 toolchain.
# CRITICAL: force 2025.2 by absolute path + PATH prepend -- sourcing settings64.sh
# alone does NOT win over a 2022.2 v++ already on PATH (that mistake = the brick).
set -e
source /code/Xilinx_2025.2/2025.2/Vitis/settings64.sh
export PATH=/code/Xilinx_2025.2/2025.2/Vitis/bin:/code/Xilinx_2025.2/2025.2/Vivado/bin:$PATH
VPP=/code/Xilinx_2025.2/2025.2/Vitis/bin/v++
P=/code/Xilinx_2025.2/2025.2/Vitis/base_platforms/xilinx_vck190_base_202520_1/xilinx_vck190_base_202520_1.xpfm
cd /home/snehadri/plstream_2025_original
echo "USING v++: $VPP"; $VPP --version | head -2
rm -rf _x pl_stream.link.xsa pl_stream.xclbin package

echo "=== [1/3] PL compile (2025.2, PAIRWISE_PL_LOWDSP) $(date +%T) ==="
$VPP -c --save-temps -t hw --platform $P -k pl_stream_top -D PAIRWISE_PL_LOWDSP \
  --temp_dir ./_x -I. -o ./_x/pl_stream_top.xo pl_stream_top.cpp
echo "=== [2/3] link @80MHz (2025.2) $(date +%T) ==="
$VPP -l --save-temps -t hw --platform $P --clock.freqHz 80000000:pl_stream_top_1 \
  --temp_dir ./_x -o pl_stream.link.xsa ./_x/pl_stream_top.xo
echo "=== [3/3] package -> BOOT.BIN (2025.2) $(date +%T) ==="
$VPP -p -t hw --platform $P --package.out_dir ./package -o pl_stream.xclbin pl_stream.link.xsa
echo "=== DONE $(date +%T) ==="
ls -la package/BOOT.BIN package/sd_card/BOOT.BIN 2>/dev/null
echo "=== VERSION GATE ==="; strings package/BOOT.BIN 2>/dev/null | grep -i "Version=" | head
