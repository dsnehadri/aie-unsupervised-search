// Two on-array halves, x86sim / aiesim vehicle. The layer-0 -> layer-1
// loopback that the PL does on hardware is a file here: data/chain_x1_in.txt
// is PyTorch's layer-1 input (x after cross L0), so each half is checked
// against its own golden.
#include <adf.h>
#include "../aie/aie_chain_graph.h"
PasswdChainHalvesGraph g;
#ifndef AIE_NUM_EVENTS
#define AIE_NUM_EVENTS 20
#endif
int main(void) { g.init(); g.run(AIE_NUM_EVENTS); g.end(); return 0; }
