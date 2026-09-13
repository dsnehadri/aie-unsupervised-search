// Whole two-layer ABC stack on the array: x86sim / aiesim vehicle.
#include <adf.h>
#include "../aie/aie_chain_graph.h"
PasswdChainGraph g;
#ifndef AIE_NUM_EVENTS
#define AIE_NUM_EVENTS 20
#endif
int main(void) { g.init(); g.run(AIE_NUM_EVENTS); g.end(); return 0; }
