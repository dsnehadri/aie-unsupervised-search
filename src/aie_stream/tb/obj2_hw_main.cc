#include <adf.h>
#include "../aie/aie_graph.h"
class Obj2Graph : public adf::graph { public:
  ObjAttnGraphL<0,0> a;   // PLIO suffix _L0
  ObjAttnGraphL<0,1> b;   // PLIO suffix _L0_i1
};
Obj2Graph aie_graph;
int main(void){ aie_graph.init(); aie_graph.run(1); aie_graph.end(); return 0; }
