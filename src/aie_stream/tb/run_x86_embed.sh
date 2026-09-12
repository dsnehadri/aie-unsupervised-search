#!/bin/bash
# x86sim of the embedding-only graph (4 events from data/embed_jets_in.txt)
source /home/snehadri/Vitis/2022.2/settings64.sh
PLAT=/home/snehadri/Vitis/2022.2/base_platforms/xilinx_vck190_base_202220_1/xilinx_vck190_base_202220_1.xpfm
cd "$(dirname "$0")"
rm -rf Work_x86_embed x86simulator_output_embed
aiecompiler --target=x86sim --platform=$PLAT --stacksize=4096 --workdir=Work_x86_embed \
  --include=. --include=kernels embed_prof_main.cc > aiec_x86_embed.log 2>&1
echo "AIEC rc=$?"
x86simulator --pkg-dir=Work_x86_embed --output-dir=x86simulator_output_embed > x86run_embed.log 2>&1
echo "X86RUN rc=$?"
ls x86simulator_output_embed/data
