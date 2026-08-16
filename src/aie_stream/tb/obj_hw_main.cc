// Obj-attention-only AIE graph for hardware. Instantiates ONLY ObjAttnGraphL<0>
// for minimal first hardware bringup. main() needed for the PS partition wrapper.

#include <adf.h>
#include "../aie/aie_graph.h"

ObjAttnGraphL<0> obj_graph;

int main(void) {
    obj_graph.init();
    obj_graph.run(1);
    obj_graph.end();
    return 0;
}
