#!/bin/bash
# Rebuild pl_stream (xclbin + BOOT.BIN) with retrained weights.
# Long: HLS ~30min, v++ link (place&route) ~1-2h, package ~10min.
set -e
cd "$(dirname "$0")"
HLS=/home/snehadri/Vitis_HLS/2022.2/bin/vitis_hls
VPP=/home/snehadri/Vitis/2022.2/bin/v++
XPFM=/home/snehadri/Vitis/2022.2/base_platforms/xilinx_vck190_base_202220_1/xilinx_vck190_base_202220_1.xpfm

if [ "${SKIP_HLS:-0}" != "1" ]; then
  echo "===== [1/3] HLS csynth + export .xo  $(date) ====="
  $HLS -f run_synth_export.tcl   # export_design writes pl_stream_top.xo to cwd
fi
ls -la pl_stream_top.xo

echo "===== [2/3] v++ link -> xsa/xclbin  $(date) ====="
$VPP --link --target hw --platform "$XPFM" \
  --save-temps --temp_dir _x_rebuild \
  -o pl_stream.xsa pl_stream_top.xo

echo "===== [3/3] package -> BOOT.BIN  $(date) ====="
$VPP --package --target hw --platform "$XPFM" \
  pl_stream.xsa \
  --package.boot_mode sd \
  --package.out_dir package_rebuild \
  -o pl_stream_rebuilt.xclbin

echo "===== DONE  $(date) ====="
ls -la package_rebuild/sd_card/BOOT.BIN 2>/dev/null || ls -la package_rebuild/ 2>/dev/null
