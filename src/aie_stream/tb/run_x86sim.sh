#!/bin/bash
set -e
source /home/snehadri/Vitis/2022.2/settings64.sh
PLAT=/home/snehadri/Vitis/2022.2/base_platforms/xilinx_vck190_base_202220_1/xilinx_vck190_base_202220_1.xpfm
PE=/home/snehadri/repos/unsupervised-search/phase3_export
cd /home/snehadri/repos/aie-unsupervised-search/src/aie_stream/tb
echo "[1/4] gen inputs (designed scales: obj@2048, cand@512)"
python3 gen_attn_inputs.py --phase3 $PE --event 0 --num-events 1
echo "[2/4] aiecompiler x86sim"
rm -rf Work_x86 x86simulator_output
aiecompiler --target=x86sim --platform=$PLAT --stacksize=2048 --workdir=Work_x86 \
  --Xpreproc="-DAIE_NUM_EVENTS=1" aie_attn_test.cpp > aiec_x86_2.log 2>&1
echo "AIEC_DONE rc=$?"
echo "[3/4] run x86simulator"
x86simulator --pkg-dir=Work_x86 > x86run.log 2>&1
echo "X86RUN_DONE rc=$?"
echo "[4/4] check outputs vs PyTorch golden"
python3 check_attn_outputs.py --phase3 $PE --event 0 --num-events 1 --output-dir x86simulator_output
