#!/bin/bash
# FLOAT_AIE x86sim: unquantized (float32) build of the SAME kernel/graph
# structure, run against the retrained PyTorch goldens. 2022.2 tools are
# sim-only here (never for BOOT.BIN).
set -e
source /home/snehadri/Vitis/2022.2/settings64.sh
PLAT=/home/snehadri/Vitis/2022.2/base_platforms/xilinx_vck190_base_202220_1/xilinx_vck190_base_202220_1.xpfm
PE=/home/snehadri/repos/unsupervised-search/phase3_export_retrained
cd /home/snehadri/repos/aie-unsupervised-search/src/aie_stream/tb
NEV=${1:-20}
echo "[1/4] gen float inputs ($NEV events)"
python3 gen_attn_inputs.py --phase3 $PE --event 0 --num-events $NEV --data-dir ./data --float
echo "[2/4] aiecompiler x86sim (FLOAT_AIE)"
rm -rf Work_x86_float x86simulator_output
aiecompiler --target=x86sim --platform=$PLAT --stacksize=8192 --workdir=Work_x86_float \
  --Xpreproc="-DFLOAT_AIE -DAIE_NUM_EVENTS=$NEV" aie_attn_test.cpp > aiec_x86_float.log 2>&1
echo "AIEC_DONE rc=$?"
echo "[3/4] run x86simulator"
x86simulator --pkg-dir=Work_x86_float > x86run_float.log 2>&1
echo "X86RUN_DONE rc=$?"
echo "[4/4] check outputs vs PyTorch golden (float, all 6 blocks)"
python3 check_attn_outputs.py --phase3 $PE --event 0 --num-events $NEV \
  --output-dir x86simulator_output/data --float --tol 1e-3 \
  --per-event-out float_aie_errors.json
