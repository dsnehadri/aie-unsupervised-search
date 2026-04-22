# run_csim.tcl
open_project attn_block_proj
set_top attn_block_obj_top
open_solution "solution1"
set_part xcvc1902-vsva2197-2MP-e-S
create_clock -period 5 -name default

add_files attn_block_pl/attn_block_obj_top.cpp
add_files attn_block_pl/attn_block_cand_top.cpp
add_files attn_block_pl/attn_block_cross_top.cpp

add_files -tb test_benches/attn_block_tb.cpp
add_files -tb cnpy/cnpy.cpp
add_files -tb cnpy/cnpy.h
add_files -tb test_benches/tb_helpers.h

csim_design -ldflags "-lz"

exit