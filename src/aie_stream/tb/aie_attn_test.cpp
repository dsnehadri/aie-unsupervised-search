#include <adf.h>
#include "../aie/aie_graph.h"

using namespace adf;

class AttnTestGraph : public graph {
public:
    ObjAttnGraphL<0> obj0;
    CandAttnGraphL<0> cand0;
    CrossAttnGraphL<0> cross0;
    // L1 obj included to cover the no-wij variant (fed a copy of the L0
    // input; its output is not golden-checked, it exercises compile+run)
    ObjAttnGraphL<1> obj1;
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
