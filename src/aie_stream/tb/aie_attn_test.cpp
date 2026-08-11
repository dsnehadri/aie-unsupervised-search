#include <adf.h>
#include "../aie/aie_graph.h"

using namespace adf;

class AttnTestGraph : public graph {
public:
    ObjAttnGraphL<0> obj0;
    CandAttnGraphL<0> cand0;
    CrossAttnGraphL<0> cross0;
    // Full L1 coverage: all six subgraphs, so the stepped hybrid golden
    // (hybrid_golden_steps) can chain PL stages with x86sim per block.
    ObjAttnGraphL<1> obj1;
    CandAttnGraphL<1> cand1;
    CrossAttnGraphL<1> cross1;
};

AttnTestGraph g;

#ifndef AIE_NUM_EVENTS
#define AIE_NUM_EVENTS 100
#endif

#if defined(__AIESIM__) || defined(__X86SIM__)
int main(int /*argc*/, char** /*argv*/) {
    g.init();
    // one iteration = one event. plio text files contain
    // AIE_NUM_EVENTS events' data, concatenated by gen_attn_inputs.py.
    g.run(AIE_NUM_EVENTS);
    g.end();
    return 0;
}
#endif
