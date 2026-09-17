// x86sim / aiesim vehicle for the pairwise bias MLP on the array: raw jets in,
// the 12 x 16 bias rows out, to compare with data/obj_wij_h0_L0.txt (the
// fabric's padded slice, from the PyTorch tensors).
#include <adf.h>
#include "../aie/aie_chain_graph.h"
class PairTest : public adf::graph {
public:
    input_plio jets; output_plio out; PairwiseGraph pair;
    PairTest() {
        jets = input_plio::create("embed_jets_in", plio_64_bits, "data/embed_jets_in.txt");
        out  = output_plio::create("pair_out", plio_64_bits, "data/pair_out.txt");
        connect<window<EMBED_IN_WORDS * sizeof(int16)>>(jets.out[0], pair.jets_in);
        connect<stream>(pair.k_m4.out[0], out.in[0]);
    }
};
PairTest g;
int main(void) { g.init(); g.run(AIE_NUM_EVENTS); g.end(); return 0; }
