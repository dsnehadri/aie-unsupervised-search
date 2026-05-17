#include <adf.h>
#include "../aie/aie_graph.h"

using namespace adf;

class AttnTestGraph : public graph {
public:
    ObjAttnGraphL<0> obj0;
    CandAttnGraphL<0> cand0;
    CrossAttnGraphL<0> cross0;
};

AttnTestGraph g;

#if defined(__AIESIM__) || defined(__X86SIM__)
int main(int /*argc*/, char** /*argv*/) {
    g.init();
    // one iteration = one event. plio text files contain
    // one events data. gen_attn_inputs.py picks event 0
    g.run(1);
    g.end();
    return 0;
}
#endif
