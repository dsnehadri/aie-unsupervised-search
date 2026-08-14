#!/bin/bash
# Stepped hybrid golden: predict the exact aie_stream hardware output for one
# event by chaining native PL stages with x86sim per attention block.
set -e
source /home/snehadri/Vitis/2022.2/settings64.sh
PLAT=/home/snehadri/Vitis/2022.2/base_platforms/xilinx_vck190_base_202220_1/xilinx_vck190_base_202220_1.xpfm
cd /home/snehadri/repos/aie-unsupervised-search/src/aie_stream/tb
IN=${1:?input.bin}; EV=${2:-0}
mkdir -p golden_state data

# zero-init every PLIO input so uncomputed blocks run on junk without hanging
python3 - <<'PY'
import os
sizes = {"obj_x_in_L0":208, "obj_x_in_L1":208,
         "cand_c_in_L0":48, "cand_c_in_L1":48,
         "cross_x_in_L0":192, "cross_x_in_L1":192,
         "cross_c_in_L0":48, "cross_c_in_L1":48}
for h in range(4): sizes[f"obj_wij_h{h}_L0"]=156
for n,c in sizes.items():
    with open(f"data/{n}.txt","w") as f:
        for i in range(0,c,4): f.write("0 0 0 0\n")
PY

SIM="x86simulator --pkg-dir=Work_x86_1ev"
G=./hybrid_golden_steps
$G p1 "$IN" "$EV"
$SIM > /dev/null 2>&1   # obj0
$G p2
$SIM > /dev/null 2>&1   # cand0
$G p3
$SIM > /dev/null 2>&1   # cross0
$G p4
$SIM > /dev/null 2>&1   # obj1
$G p5
$SIM > /dev/null 2>&1   # cand1
$G p6
$SIM > /dev/null 2>&1   # cross1
$G p7
