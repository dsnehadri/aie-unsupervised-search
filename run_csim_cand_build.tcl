open_project cand_build_proj
set_top candidate_build_top
add_files cand_build.cpp
add_files attn_helpers.h
add_files attn_block_types.h
add_files -tb cand_build_tb.cpp
add_files -tb tb_helpers.h 
add_files -tb cnpy.cpp
add_files -tb cnpy.h 
open_solution "solution1"
set_part xcvc1902-vsva2197-2MP-e-S
csim_design -ldflags "-lz"