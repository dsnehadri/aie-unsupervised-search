// Whole-chain graph for aiesimulator --profile: several events, so the
// per-kernel numbers are steady-state rather than a single cold pass.
// chain_hw_main.cc runs one iteration, which is what the hardware host drives.
#include <adf.h>
#include "../aie/aie_chain_graph.h"
#ifndef AIE_NUM_EVENTS
#define AIE_NUM_EVENTS 8
#endif
PasswdChainGraph aie_graph;
int main(void) { aie_graph.init(); aie_graph.run(AIE_NUM_EVENTS); aie_graph.end(); return 0; }
