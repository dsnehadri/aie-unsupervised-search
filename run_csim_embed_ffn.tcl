open_project embed_ffn_proj
set_top embed_ffn_top
add_files embed_ffn_source/embed_ffn.cpp 
add_files embed_ffn_source/embed_ffn.h 
add_files dnn_block_source/dnn_block.h
add_files attn_block_source/attn_helpers.h
add_files attn_block_source/attn_block_types.h
add_files -tb embed_ffn_source/embed_ffn_tb.cpp
add_files -tb test_benches/tb_helpers.h 
add_files -tb cnpy/cnpy.cpp
add_files -tb cnpy/cnpy.h
open_solution "solution1"   
set_part xcvc1902-vsva2197-2MP-e-S
csim_design -ldflags "-lz" 