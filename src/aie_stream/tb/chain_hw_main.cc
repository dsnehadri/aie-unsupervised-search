// Whole two-layer ABC stack on the array, hardware main. Instance MUST be
// named "aie_graph" (host_aie.cpp's xrt::graph name), as for full_hw_main.cc.
#include <adf.h>
#include "../aie/aie_chain_graph.h"
PasswdChainGraph aie_graph;
int main(void){ aie_graph.init(); aie_graph.run(1); aie_graph.end(); return 0; }
