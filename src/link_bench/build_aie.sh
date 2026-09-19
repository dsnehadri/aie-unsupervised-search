#!/bin/bash
set -e
source /code/Xilinx_2025.2/2025.2/Vitis/settings64.sh
export PATH=/code/Xilinx_2025.2/2025.2/Vitis/bin:/code/Xilinx_2025.2/2025.2/Vivado/bin:$PATH
P=/code/Xilinx_2025.2/2025.2/Vitis/base_platforms/xilinx_vck190_base_202520_1/xilinx_vck190_base_202520_1.xpfm
cd /home/snehadri/aie_pt_link
rm -rf Work_hw_pt libadf.a
/code/Xilinx_2025.2/2025.2/Vitis/aietools/bin/aiecompiler --target=hw --platform=$P --stacksize=2048 --workdir=Work_hw_pt --include=. pt_hw_main.cc --output-archive=/home/snehadri/aie_pt_link/libadf.a
echo "PT_AIE_DONE rc=$?"
