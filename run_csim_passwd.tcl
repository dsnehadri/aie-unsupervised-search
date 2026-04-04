open_project passwd_top_proj
set_top passwd_top
open_solution "solution1"
set_part xcvc1902-vsva2197-2MP-e-S
create_clock -period 5 -name default

add_files passwd_source/passwd.cpp

add_files -tb test_benches/passwd_tb.cpp
add_files -tb test_benches/tb_helpers.h
add_files -tb cnpy/cnpy.cpp
add_files -tb cnpy/cnpy.h

csim_design -ldflags "-lz" -clean

exit