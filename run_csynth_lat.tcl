# Latency experiments on the object block. Env: PROJ (project dir), CLK (ns),
# EXTRA (extra -D flags). Same source and base flags as run_csynth_fabmul.tcl.
open_project -reset $::env(PROJ)
set_top attn_block_obj_top
open_solution -reset "sol"
set_part xcvc1902-vsva2197-2MP-e-S
create_clock -period $::env(CLK) -name default
add_files attn_block_pl/attn_block_obj_top.cpp -cflags "-DLN_MODE=5 -DOBJ_DATAFLOW $::env(EXTRA)"
csynth_design
exit
