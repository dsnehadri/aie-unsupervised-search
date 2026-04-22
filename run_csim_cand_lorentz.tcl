open_project cand_lorentz_proj
set_top cand_lorentz_top
add_files cand_lorentz/cand_lorentz.cpp
add_files cand_lorentz/cand_lorentz.h
add_files attn_block_pl/attn_block_types.h
add_files -tb test_benches/cand_lorentz_tb.cpp
add_files -tb test_benches/tb_helpers.h 
add_files -tb cnpy/cnpy.cpp
add_files -tb cnpy/cnpy.h 
open_solution "solution1"
set_part xcvc1902-vsva2197-2MP-e-S
csim_design -ldflags "-lz"