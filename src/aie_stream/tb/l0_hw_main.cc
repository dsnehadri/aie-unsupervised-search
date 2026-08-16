// Layer-0-only isolation graph: obj0 + cand0 + cross0 (AttnTestGraph shape).
// Instance named "aie_graph" so host_aie's xrt::graph(...,"aie_graph") matches.
#include <adf.h>
#include "../aie/aie_graph.h"
class L0Graph : public adf::graph {
public:
    ObjAttnGraphL<0>   obj0;
    CandAttnGraphL<0>  cand0;
    CrossAttnGraphL<0> cross0;
};
L0Graph aie_graph;
int main(void){ aie_graph.init(); aie_graph.run(1); aie_graph.end(); return 0; }
