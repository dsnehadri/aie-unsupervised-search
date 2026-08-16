#include <adf.h>
#include "../aie/aie_graph.h"
using namespace adf;
class OCGraph : public graph {
public:
    ObjAttnGraphL<0>  obj0;
    CandAttnGraphL<0> cand0;
    OCGraph() {
        // PLACEMENT SEPARATION EXPERIMENT: force obj0 to cols 0-2, cand0 to cols 47-49
        // (opposite ends of the array) to test the co-placement/NoC-proximity deadlock hypothesis.
        // obj0 -> cols 0-2
        location<kernel>(obj0.k_pre[0])=tile(0,1); location<kernel>(obj0.k_pre[1])=tile(0,2);
        location<kernel>(obj0.k_pre[2])=tile(0,3); location<kernel>(obj0.k_pre[3])=tile(0,4);
        location<kernel>(obj0.k_post_h[0])=tile(1,1); location<kernel>(obj0.k_post_h[1])=tile(1,2);
        location<kernel>(obj0.k_post_h[2])=tile(1,3); location<kernel>(obj0.k_post_h[3])=tile(1,4);
        location<kernel>(obj0.k_post_ac)=tile(2,1); location<kernel>(obj0.k_post_ap)=tile(2,2);
        location<kernel>(obj0.k_post_b1)=tile(2,3); location<kernel>(obj0.k_post_b2)=tile(2,4);
        location<kernel>(obj0.k_post_c)=tile(2,5);
        // cand0 -> cols 47-49
        location<kernel>(cand0.k_pre[0])=tile(47,1); location<kernel>(cand0.k_pre[1])=tile(47,2);
        location<kernel>(cand0.k_pre[2])=tile(47,3); location<kernel>(cand0.k_pre[3])=tile(47,4);
        location<kernel>(cand0.k_post_h[0])=tile(48,1); location<kernel>(cand0.k_post_h[1])=tile(48,2);
        location<kernel>(cand0.k_post_h[2])=tile(48,3); location<kernel>(cand0.k_post_h[3])=tile(48,4);
        location<kernel>(cand0.k_post_ac)=tile(49,1); location<kernel>(cand0.k_post_ap)=tile(49,2);
        location<kernel>(cand0.k_post_b1)=tile(49,3); location<kernel>(cand0.k_post_b2)=tile(49,4);
        location<kernel>(cand0.k_post_c)=tile(49,5);
    }
};
OCGraph aie_graph;
int main(void){ aie_graph.init(); aie_graph.run(1); aie_graph.end(); return 0; }
