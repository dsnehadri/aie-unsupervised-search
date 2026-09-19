#!/bin/bash
# aiesim profile of the four micro-benchmark kernels; prints cycles per event per kernel
source /code/Xilinx_2025.2/2025.2/Vitis/settings64.sh
export PATH=/code/Xilinx_2025.2/2025.2/Vitis/bin:$PATH
P=/code/Xilinx_2025.2/2025.2/Vitis/base_platforms/xilinx_vck190_base_202520_1/xilinx_vck190_base_202520_1.xpfm
cd "$(dirname "$(readlink -f "$0")")"
rm -rf Work_bench aiesim_bench
aiecompiler --target=hw --platform=$P --stacksize=4096 --workdir=Work_bench --include=. --include=kernels --Xpreproc=-DLN_CLZ --profile bench_main.cc > aie_bench_build.log 2>&1
echo "COMPILE_DONE rc=$?"
aiesimulator --pkg-dir=Work_bench --profile --output-dir=aiesim_bench > aiesim_bench.log 2>&1
echo "SIM_DONE rc=$?"
python3 - <<'PY'
import re, glob, os
for f in sorted(glob.glob("aiesim_bench/profile_funct_*.txt")):
    txt=open(f).read().split("          Calls  Cycles tot")[0]
    for m in re.finditer(r"^\s*(\d+)\s+(\d+)\s+[\d.]+%\s+\d+\s+\d+\s+\d+\s+(\d+)\s+[\d.]+%.*?(_Z\d+bench_\w+?)P", txt, re.M):
        print(f"{m.group(4)[3:]:16s} {int(m.group(3))/int(m.group(1)):8.0f} cycles per call  ({int(m.group(3))/int(m.group(1))/1250:.2f} us)")
import numpy as np
rd=lambda p: np.array([int(t) for l in open(p) for t in l.split() if not l.startswith('T')])
a,b=rd("aiesim_bench/data/bench_ln.txt"),rd("aiesim_bench/data/bench_ln_il.txt"); print("ln_il identical to ln:", bool((a==b).all()), len(a))
a,b=rd("aiesim_bench/data/bench_sm_vec.txt"),rd("aiesim_bench/data/bench_sm_lut.txt"); print("sm_lut vs sm_vec: max |diff| %d LSB (of 512), mean %.2f" % (np.abs(a-b).max(), np.abs(a-b).mean()))
PY
echo BENCH_DONE
