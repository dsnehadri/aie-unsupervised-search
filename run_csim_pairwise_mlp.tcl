open_project pairwise_mlp_proj
set_top pairwise_mlp_top
add_files pairwise_mlp/pairwise_mlpcpp 
add_files pairwise_mlp/pairwise_mlp.h 
add_files dnn_block/dnn_block.h
add_files attn_block_pl/attn_helpers.h
add_files attn_block_pl/attn_block_types.h
add_files -tb test_benches/pairwise_mlp_tb.cpp
add_files -tb test_benches/tb_helpers.h 
add_files -tb cnpy/cnpy.cpp
add_files -tb cnpy/cnpy.h
open_solution "solution1"   
set_part xcvc1902-vsva2197-2MP-e-S
csim_design -ldflags "-lz" 