# Does binding the linear-layer multiplies to LUT fabric free enough DSPs to
# make four-way head parallelism affordable?
open_project -reset csynth_fabmul_proj
set_top attn_block_obj_top
open_solution -reset "sol"
set_part xcvc1902-vsva2197-2MP-e-S
create_clock -period 12.5 -name default
add_files attn_block_pl/attn_block_obj_top.cpp -cflags "-DLN_MODE=5 -DOBJ_DATAFLOW $::env(EXTRA)"
csynth_design
exit
