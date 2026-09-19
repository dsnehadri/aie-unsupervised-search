// Array micro-benchmark graph: one 13-row x window into four kernels.
#include <adf.h>
#include "../../attn_block_aie/kernels/attn_aie_types.h"
using namespace adf;
void bench_ln(input_window_int16*, output_window_int16*);
void bench_ln_il(input_window_int16*, output_window_int16*);
void bench_sm_vec(input_window_int16*, output_window_int16*);
void bench_sm_lut(input_window_int16*, output_window_int16*);
class Bench : public graph {
public:
    input_plio in; output_plio o[4]; kernel k[4];
    Bench() {
        in = input_plio::create("obj_x_in_L0", plio_64_bits, "data/obj_x_in_L0.txt");
        const char* names[4] = {"bench_ln", "bench_ln_il", "bench_sm_vec", "bench_sm_lut"};
        k[0] = kernel::create(bench_ln); k[1] = kernel::create(bench_ln_il);
        k[2] = kernel::create(bench_sm_vec); k[3] = kernel::create(bench_sm_lut);
        for (int i = 0; i < 4; i++) {
            source(k[i]) = (std::string("kernels/") + names[i] + ".cc").c_str();
            runtime<ratio>(k[i]) = 0.9;
            o[i] = output_plio::create(names[i], plio_64_bits, (std::string("data/") + names[i] + ".txt").c_str());
            connect<window<(N_MAX + 1) * E_DIM * sizeof(int16)>>(in.out[0], k[i].in[0]);
            connect<window<N_MAX * E_DIM * sizeof(int16)>>(k[i].out[0], o[i].in[0]);
        }
    }
};
Bench g;
int main(void) { g.init(); g.run(4); g.end(); return 0; }
