#include <adf.h>
#include "../aie/aie_graph.h"
// One instance of each attention block type, for per-block batch sweeps.
class Blocks3 : public adf::graph { public:
  ObjAttnGraphL<0,0> obj0;
  CandAttnGraphL<0>  cand0;
  CrossAttnGraphL<0> cross0;
};
Blocks3 aie_graph;
int main(void){ aie_graph.init(); aie_graph.run(1); aie_graph.end(); return 0; }
