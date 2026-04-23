open_project pl_stream
open_solution solution1
set_part {xcvc1902-vsva2197-2MP-e-S}
set_top pl_stream_top
add_files src/pl_stream/pl_stream_top.cpp
add_files -tb test_benches/pl_stream_tb.cpp
add_files -tb test_benches/tb_helpers.h
add_files -tb cnpy/cnpy.cpp
add_files -tb cnpy/cnpy.h

config_export -format xo
csynth_design
export_design -format xo -output pl_stream_top.xo