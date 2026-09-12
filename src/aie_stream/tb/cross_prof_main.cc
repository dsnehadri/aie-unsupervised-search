// Cross-attention-only AIE graph for the aiesimulator cycle profile
// (the counterpart of obj_prof_main.cc).
#include <adf.h>
#include "../aie/aie_graph.h"

CrossAttnGraphL<0> cross_graph;

int main(void) {
    cross_graph.init();
    cross_graph.run(4);
    cross_graph.end();
    return 0;
}
