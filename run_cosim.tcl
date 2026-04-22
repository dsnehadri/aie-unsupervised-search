open_project passwd_stream_proj
set_top passwd_stream_top
open_solution "solution1"
set_part {xcvc1902-vsva2197-2MP-e-S}
create_clock -period 5 -name default

add_files passwd_stream_source/passwd_stream_top.cpp

add_files -tb test_benches/passwd_stream_tb.cpp
add_files -tb cnpy/cnpy.cpp
add_files -tb cnpy/cnpy.h

csim_design -ldflags "-lz" -clean
csynth_design
cosim_design -ldflags "-lz"
exit