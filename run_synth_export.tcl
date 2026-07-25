# HLS C-synthesis + export .xo for pl_stream_top (rebuild with retrained weights)
# usage: vitis_hls -f run_synth_export.tcl
open_project pl_stream_rebuild
open_solution solution1
set_part {xcvc1902-vsva2197-2MP-e-S}
create_clock -period 5 -name default
set_top pl_stream_top
add_files -cflags "-I src/pl_stream" src/pl_stream/pl_stream_top.cpp
config_export -format xo
csynth_design
export_design -format xo -output pl_stream_top.xo
exit
