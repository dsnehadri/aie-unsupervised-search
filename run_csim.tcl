# run_csim.tcl
open_project attn_block_proj
set_top attn_block_obj
add_files attn_block_obj.cpp
add_files attn_helpers.h
add_files attn_block_types.h
add_files -tb attn_block_tb.cpp
add_files -tb cnpy.cpp
add_files -tb cnpy.h
open_solution "solution1"
set_part xcvc1902-vsva2197-2MP-e-S
csim_design -ldflags "-lz"