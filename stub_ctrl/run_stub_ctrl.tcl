open_project stub_ctrl
open_solution solution1
set_part {xcvc1902-vsva2197-2MP-e-S}
create_clock -period 5 -name default
set_top stub_ctrl
add_files stub_ctrl.cpp
config_export -format xo
csynth_design
export_design -format xo -output stub_ctrl.xo