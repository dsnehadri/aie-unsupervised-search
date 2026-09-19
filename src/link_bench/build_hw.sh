#!/bin/bash
set -e
source /code/Xilinx_2025.2/2025.2/Vitis/settings64.sh
export PATH=/code/Xilinx_2025.2/2025.2/Vitis/bin:/code/Xilinx_2025.2/2025.2/Vivado/bin:$PATH
VPP=/code/Xilinx_2025.2/2025.2/Vitis/bin/v++
$VPP --version | grep -q 2025.2 || { echo "WRONG_VPP"; exit 1; }
P=/code/Xilinx_2025.2/2025.2/Vitis/base_platforms/xilinx_vck190_base_202520_1/xilinx_vck190_base_202520_1.xpfm
cd /home/snehadri/aie_pt_link
if [ ! -f p128_top.xo ]; then
  $VPP -c -t hw --platform $P -k p128_top --save-temps --temp_dir ./_x_p128 -o p128_top.xo p128_top.cpp
fi
echo "P128C_DONE rc=$?"
if [ ! -f q512_top.xo ]; then
  $VPP -c -t hw --platform $P -k q512_top --save-temps --temp_dir ./_x_q512 -o q512_top.xo q512_top.cpp
fi
echo "Q512C_DONE rc=$?"
if [ ! -f lb_top.xo ]; then
  $VPP -c -t hw --platform $P -k lb_top --save-temps --temp_dir ./_x_pl2 -o lb_top.xo lb_top.cpp
fi
echo "LBC_DONE rc=$?"
while ! grep -q PT_AIE_DONE aie_build.log 2>/dev/null; do sleep 10; done
rm -rf pt_link.xsa pt_link.xclbin package
$VPP -l -t hw --platform $P --config link.cfg --save-temps --clock.defaultFreqHz 250000000 --temp_dir ./_x_link -o pt_link.xsa p128_top.xo q512_top.xo lb_top.xo libadf.a
echo "LINK_DONE rc=$?"
$VPP -p -t hw --platform $P --package.out_dir ./package -o pt_link.xclbin pt_link.xsa libadf.a
echo "PT_ALL_DONE rc=$?"
