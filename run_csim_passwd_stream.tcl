# usage: /home/snehadri/Vitis_HLS/2022.2/bin/vitis_hls -f run_csim_passwd_stream.tcl

open_project passwd_stream_proj
set_top passwd_stream_top
open_solution "solution1"
set_part {xcvc1902-vsva2197-2MP-e-S}
create_clock -period 5 -name default

# source files
add_files passwd_stream_source/passwd_stream_top.cpp

# testbench files

add_files -tb test_benches/passwd_stream_tb.cpp
add_files -tb test_benches/tb_helpers.h
add_files -tb cnpy/cnpy.cpp
add_files -tb cnpy/cnpy.h

# run csim

csim_design -ldflags "-lz" -clean
exit
