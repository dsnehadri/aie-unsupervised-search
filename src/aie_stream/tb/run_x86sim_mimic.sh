#!/bin/bash
set -e
source /home/snehadri/Vitis/2022.2/settings64.sh
PLAT=/home/snehadri/Vitis/2022.2/base_platforms/xilinx_vck190_base_202220_1/xilinx_vck190_base_202220_1.xpfm
PE=/home/snehadri/repos/unsupervised-search/phase3_export
cd /home/snehadri/repos/aie-unsupervised-search/src/aie_stream/tb
echo "[1] gen designed-scale inputs"
python3 gen_attn_inputs.py --phase3 $PE --event 0 --num-events 1 >/dev/null
echo "[2] rescale obj_x_in 2048 -> 512 (mimic hybrid bridge feeding obj at data_t<16,7>)"
python3 - <<'PY'
src="data/obj_x_in_L0.txt"
nums=[]
with open(src) as f:
    for line in f: nums += [int(t) for t in line.split()]
sc=[int(round(n/4.0)) for n in nums]
with open(src,"w") as f:
    for i in range(0,len(sc),4): f.write(" ".join(str(x) for x in sc[i:i+4])+"\n")
print("obj rescaled, first:", sc[:4])
PY
echo "[3] clean aiecompile x86sim (snapshots inputs)"
rm -rf Work_x86 x86simulator_output
aiecompiler --target=x86sim --platform=$PLAT --stacksize=2048 --workdir=Work_x86 \
  --Xpreproc="-DAIE_NUM_EVENTS=1" aie_attn_test.cpp > aiec_mimic.log 2>&1
echo "AIEC rc=$?"
echo "[4] run + check"
x86simulator --pkg-dir=Work_x86 > x86run_mimic2.log 2>&1
python3 check_attn_outputs.py --phase3 $PE --event 0 --num-events 1 --data-dir ./x86simulator_output/data
