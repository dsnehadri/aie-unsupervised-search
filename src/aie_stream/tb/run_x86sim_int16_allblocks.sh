#!/bin/bash
# int16 x86sim over all 6 blocks (L0+L1) with per-block exact PyTorch inputs --
# the quantized counterpart of run_x86sim_float.sh for the unquantized-proof
# comparison. x86sim is bit-exact for the integer kernels, so these errors are
# the deployed hardware's quantization error per block.
set -e
source /home/snehadri/Vitis/2022.2/settings64.sh
PLAT=/home/snehadri/Vitis/2022.2/base_platforms/xilinx_vck190_base_202220_1/xilinx_vck190_base_202220_1.xpfm
PE=/home/snehadri/repos/unsupervised-search/phase3_export_retrained
cd /home/snehadri/repos/aie-unsupervised-search/src/aie_stream/tb
NEV=${1:-20}
echo "[1/4] gen int16 inputs ($NEV events, L0+L1)"
python3 gen_attn_inputs.py --phase3 $PE --event 0 --num-events $NEV --data-dir ./data
echo "[2/4] aiecompiler x86sim (int16)"
rm -rf Work_x86_int x86simulator_output
aiecompiler --target=x86sim --platform=$PLAT --stacksize=2048 --workdir=Work_x86_int \
  --Xpreproc="-DAIE_NUM_EVENTS=$NEV" aie_attn_test.cpp > aiec_x86_int.log 2>&1
echo "AIEC_DONE rc=$?"
echo "[3/4] run x86simulator"
x86simulator --pkg-dir=Work_x86_int > x86run_int.log 2>&1
echo "X86RUN_DONE rc=$?"
echo "[4/4] check outputs vs PyTorch golden (int16, all 6 blocks)"
python3 check_attn_outputs.py --phase3 $PE --event 0 --num-events $NEV \
  --output-dir x86simulator_output/data --all-blocks --tol 0.5 \
  --per-event-out int16_aie_errors.json
