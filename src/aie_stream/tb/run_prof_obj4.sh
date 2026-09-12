#!/bin/bash
# hw compile + aiesimulator profile of the object block with the vector-I/O kernels
source /code/Xilinx_2025.2/2025.2/Vitis/settings64.sh
export PATH=/code/Xilinx_2025.2/2025.2/Vitis/bin:$PATH
cd /home/snehadri/repos/aie-unsupervised-search/src/aie_stream/tb
P=/code/Xilinx_2025.2/2025.2/Vitis/base_platforms/xilinx_vck190_base_202520_1/xilinx_vck190_base_202520_1.xpfm
rm -rf Work_prof_obj4 aiesim_prof_out4
aiecompiler --target=hw --platform=$P --stacksize=4096 --workdir=Work_prof_obj4 --include=. --include=kernels --profile obj_prof_main.cc > aie_prof_build4.log 2>&1
echo "COMPILE_DONE rc=$?"
aiesimulator --pkg-dir=Work_prof_obj4 --profile --output-dir=aiesim_prof_out4 > aiesim_prof4.log 2>&1
echo "SIM_DONE rc=$?"
