#include <adf.h>
#include "../aie/aie_graph.h"
class CrossGraph : public adf::graph { public: CrossAttnGraphL<0> cross0; };
CrossGraph aie_graph;
int main(void){ aie_graph.init(); aie_graph.run(1); aie_graph.end(); return 0; }
