#!/bin/bash
# Cycle-accurate latency of the array pairwise subgraph alone: jets in -> bias rows out.
# usage: run_pair_aiesim.sh <tag> "<extra defines>"
source /code/Xilinx_2025.2/2025.2/Vitis/settings64.sh
export PATH=/code/Xilinx_2025.2/2025.2/Vitis/bin:$PATH
P=/code/Xilinx_2025.2/2025.2/Vitis/base_platforms/xilinx_vck190_base_202520_1/xilinx_vck190_base_202520_1.xpfm
cd "$(dirname "$(readlink -f "$0")")"; T=$1
X="-DCHAIN_STREAM -DEMBED_PIPE -DHEAD_STREAM -DPOST_SPLIT_C -DPOST_STREAM -DPRE_STREAM -DPRE_STREAM_CROSS -DWIJ_ONE_PORT -DWIJ_PAD16 -DLN_CLZ -DPAIRWISE_ON_AIE -DAIE_NUM_EVENTS=4 $2"
rm -rf Work_pair_$T aiesim_pair_$T
aiecompiler --target=hw --platform=$P --stacksize=4096 --workdir=Work_pair_$T --include=. --include=kernels \
  $(for d in $X; do echo --Xpreproc=$d; done) pair_x86_main.cc > aie_pair_build_$T.log 2>&1
echo "COMPILE_DONE rc=$?"
aiesimulator --pkg-dir=Work_pair_$T --output-dir=aiesim_pair_$T > aiesim_pair_$T.log 2>&1
echo "SIM_DONE rc=$?"
