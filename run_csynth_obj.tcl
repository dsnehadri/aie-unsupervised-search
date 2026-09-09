# Object attention block on its own: latency and resource estimate at 80 MHz.
# Sweep the linear-layer k-partitioning with -DLIN_PARTITION.
open_project -reset csynth_obj_proj
set_top attn_block_obj_top
open_solution -reset "sol"
set_part xcvc1902-vsva2197-2MP-e-S
create_clock -period 12.5 -name default
add_files attn_block_pl/attn_block_obj_top.cpp -cflags "-DLN_MODE=$::env(LNM)"
csynth_design
exit
