open_project stub
open_solution solution1
set_part {xcvc1902-vsva2197-2MP-e-S}
create_clock -period 5 -name default
set_top stub_top
add_files src/stub/stub_top.cpp
config_export -format xo
csynth_design
export_design -format xo -output stub_top.xo
 
