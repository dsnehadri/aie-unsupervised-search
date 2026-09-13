// Candidate-attention-only AIE graph for the aiesimulator cycle profile
// (the counterpart of obj_prof_main.cc).
#include <adf.h>
#include "../aie/aie_graph.h"

CandAttnGraphL<0> cand_graph;

int main(void) {
    cand_graph.init();
    cand_graph.run(4);
    cand_graph.end();
    return 0;
}
