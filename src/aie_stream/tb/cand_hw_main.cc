#include <adf.h>
#include "../aie/aie_graph.h"
class CandGraph : public adf::graph { public: CandAttnGraphL<0> cand0; };
CandGraph aie_graph;
int main(void){ aie_graph.init(); aie_graph.run(1); aie_graph.end(); return 0; }
