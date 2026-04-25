// top level aie graph entry point for aiecompiler

#include "aie_graph.h"

PasswdFullGraph aie_graph;

#if defined(__AIESIM__) || defined(__X86SIM__)
int main(void) [
    aie_graph.init();
    aie_graph.run(1);
    aie_graph.end();
    return 0;
]

#endif