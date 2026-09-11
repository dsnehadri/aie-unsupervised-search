# Latency experiments on the object block. Env: PROJ, CLK (ns), LNM, EXTRA.
open_project -reset $::env(PROJ)
set_top attn_block_obj_top
open_solution -reset "sol"
set_part xcvc1902-vsva2197-2MP-e-S
create_clock -period $::env(CLK) -name default
add_files attn_block_pl/attn_block_obj_top.cpp -cflags "-DLN_MODE=$::env(LNM) -DOBJ_DATAFLOW $::env(EXTRA)"
csynth_design
exit
