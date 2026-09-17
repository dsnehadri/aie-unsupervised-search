#!/bin/bash
# x86sim of the whole-stack-on-array graph on the 20 test events, then the end-to-end golden check
source /home/snehadri/Vitis/2022.2/settings64.sh
PLAT=/home/snehadri/Vitis/2022.2/base_platforms/xilinx_vck190_base_202220_1/xilinx_vck190_base_202220_1.xpfm
cd "$(dirname "$(readlink -f "$0")")"
rm -rf Work_x86_chain x86simulator_output_chain
aiecompiler --target=x86sim --platform=$PLAT --stacksize=4096 --workdir=Work_x86_chain \
  --include=. --include=kernels --Xpreproc=-DAIE_NUM_EVENTS=20 $(for d in $AIE_XPRE; do echo "--Xpreproc=$d"; done) chain_x86_main.cc > aiec_x86_chain.log 2>&1
echo "AIEC rc=$?"
x86simulator --pkg-dir=Work_x86_chain --output-dir=x86simulator_output_chain > x86run_chain.log 2>&1
echo "X86RUN rc=$?"
python3 check_chain_outputs.py x86simulator_output_chain/data 20
