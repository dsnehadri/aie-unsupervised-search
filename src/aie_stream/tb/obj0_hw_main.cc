#include <adf.h>
#include "../aie/aie_graph.h"
class Obj0Graph : public adf::graph { public: ObjAttnGraphL<0> obj0; };
Obj0Graph aie_graph;
int main(void){ aie_graph.init(); aie_graph.run(1); aie_graph.end(); return 0; }
