open_project autoencoder_proj
set_top dual_autoencoder_top
open_solution "solution1"
set_part xcvc1902-vsva2197-2MP-e-S
create_clock -period 5 -name default

add_files -tb autoencoder_source/autoencoder.cpp
add_files -tb test_benches/autoencoder_tb.cpp
add_files -tb cnpy/cnpy.cpp
add_files -tb test_benches/tb_helpers.h

csim_design -ldflags "-lz" -clean
exit