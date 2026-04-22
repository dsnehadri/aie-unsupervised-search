open_project passwd_stream
open_solution solution1
set_part {xcvc1902-vsva2197-2MP-e-S}
set_top passwd_stream_top
add_files src/passwd_stream/passwd_stream_top.cpp
add_files -tb test_benches/passwd_stream_tb.cpp
add_files -tb test_benches/tb_helpers.h
add_files -tb cnpy/cnpy.cpp
add_files -tb cnpy/cnpy.h

config_export -format xo
csynth_design
export_design -format xo -output passwd_stream_top.xo