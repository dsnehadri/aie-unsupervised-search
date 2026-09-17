#!/bin/bash
# x86sim of the array pairwise MLP on 20 events, compared with the fabric's bias slice
source /code/Xilinx_2025.2/2025.2/Vitis/settings64.sh
P=/code/Xilinx_2025.2/2025.2/Vitis/base_platforms/xilinx_vck190_base_202520_1/xilinx_vck190_base_202520_1.xpfm
cd "$(dirname "$(readlink -f "$0")")"
X="-DAIE_PLACE -DCHAIN_STREAM -DEMBED_PIPE -DHEAD_STREAM -DPOST_SPLIT_C -DPOST_STREAM -DPRE_STREAM -DPRE_STREAM_CROSS -DWIJ_ONE_PORT -DWIJ_PAD16 -DLN_CLZ -DPAIRWISE_ON_AIE -DAIE_NUM_EVENTS=20"
rm -rf Work_x86_pair x86simulator_output_pair
aiecompiler --target=x86sim --platform=$P --stacksize=4096 --workdir=Work_x86_pair --include=. --include=kernels \
  $(for d in $X; do echo --Xpreproc=$d; done) pair_x86_main.cc > aiec_x86_pair.log 2>&1
echo "AIEC rc=$?"
x86simulator --pkg-dir=Work_x86_pair --output-dir=x86simulator_output_pair > x86run_pair.log 2>&1
echo "X86RUN rc=$?"
python3 - <<'PY'
import numpy as np
def rd(p): return np.array([int(t) for l in open(p) for t in l.split() if not l.startswith('T')], dtype=np.int64)
ref=rd("data/obj_wij_h0_L0.txt").reshape(-1,12,16); out=rd("x86simulator_output_pair/data/pair_out.txt").reshape(-1,12,16)
n=min(len(ref),len(out)); d=(out[:n]-ref[:n]).astype(float)/128.0
print(f"events {n}: max |diff| {np.abs(d).max():.4f} (score units), mean {np.abs(d).mean():.4f}, identical values {(d==0).sum()}/{d.size}, lanes 12-15 all zero: {bool((out[:n,:,12:]==0).all())}")
PY
echo PAIR_X86_DONE
