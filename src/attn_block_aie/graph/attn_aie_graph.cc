// top level file for aiecompiler

// build:
// aiecompiler --target=hw  -I src/ src/graph/attn_aie_graph.cpp  → libadf.a
//   aiecompiler --target=x86sim -I src/ src/graph/attn_aie_graph.cpp  → x86sim

#include "attn_block_graph.h"

// global graph instance

AttnBenchmarkGraph attn_graph;

#if defined(__AIESIM__) || defined(__X86SIM__)
int main(void) {
    attn_graph.init();
    attn_graph.run(1);
    attn_graph.end();

    return 0;
}
#endif