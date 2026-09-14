// One-layer slice of the on-array stack (embed -> obj0 -> post-obj -> cand0 ->
// cross0), for a faster aiesimulator check of the per-consumer window fan-out.
#include <adf.h>
#include "../aie/aie_chain_graph.h"
class Chain1Graph : public graph {
public:
    input_plio  plio_jets_in, plio_mask_in, plio_wij_h0, plio_wij_h1, plio_wij_h2, plio_wij_h3;
    output_plio plio_x_out, plio_c_out;
    kernel k_embed, k_asm0, k_pobj0, k_fan0;
    ObjChainL<0> obj0;  CandChainL<0> cand0;  CrossChainL<0> cross0;
    Chain1Graph() {
        plio_jets_in = input_plio::create("embed_jets_in", plio_64_bits, "data/embed_jets_in.txt");
        plio_mask_in = input_plio::create("mask_in",       plio_64_bits, "data/mask_in.txt");
        plio_wij_h0  = input_plio::create("obj_wij_h0_L0", plio_64_bits, "data/obj_wij_h0_L0.txt");
        plio_wij_h1  = input_plio::create("obj_wij_h1_L0", plio_64_bits, "data/obj_wij_h1_L0.txt");
        plio_wij_h2  = input_plio::create("obj_wij_h2_L0", plio_64_bits, "data/obj_wij_h2_L0.txt");
        plio_wij_h3  = input_plio::create("obj_wij_h3_L0", plio_64_bits, "data/obj_wij_h3_L0.txt");
        plio_x_out   = output_plio::create("chain1_x_out", plio_64_bits, "data/chain1_x_out.txt");
        plio_c_out   = output_plio::create("chain1_c_out", plio_64_bits, "data/chain1_c_out.txt");
        k_embed = kernel::create(embed_mlp);           source(k_embed) = "kernels/embed_kernel.cc";
        k_asm0  = kernel::create(chain_assemble_zero); source(k_asm0)  = "kernels/chain_kernels.cc";
        k_pobj0 = kernel::create(chain_post_obj);      source(k_pobj0) = "kernels/chain_kernels.cc";
        k_fan0  = kernel::create(chain_fanout_c);      source(k_fan0)  = "kernels/chain_kernels.cc";
        for (kernel* k : {&k_embed, &k_asm0, &k_pobj0, &k_fan0}) runtime<ratio>(*k) = 0.9;
        constexpr int jets_sz = EMBED_IN_WORDS * sizeof(int16), mask_sz = E_DIM * sizeof(int16);
        constexpr int x_sz = N_MAX * E_DIM * sizeof(int16), xm_sz = (N_MAX + 1) * E_DIM * sizeof(int16);
        constexpr int c_sz = T_DIM * E_DIM * sizeof(int16), wij_sz = N_MAX * N_KV * sizeof(int16);
        connect<window<jets_sz>>(plio_jets_in.out[0], k_embed.in[0]);
        connect<window<x_sz>>(k_embed.out[0], k_asm0.in[0]);
        connect<window<mask_sz>>(plio_mask_in.out[0], k_asm0.in[1]);
        for (int h = 0; h < N_HEADS; h++) connect<window<xm_sz>>(k_asm0.out[h], obj0.x_in_h[h]);
        connect<window<xm_sz>>(k_asm0.out[N_HEADS], obj0.x_in_res);
        connect<window<wij_sz>>(plio_wij_h0.out[0], obj0.wij_h0);
        connect<window<wij_sz>>(plio_wij_h1.out[0], obj0.wij_h1);
        connect<window<wij_sz>>(plio_wij_h2.out[0], obj0.wij_h2);
        connect<window<wij_sz>>(plio_wij_h3.out[0], obj0.wij_h3);
        connect<window<x_sz>>(obj0.x_out, k_pobj0.in[0]);
        connect<window<mask_sz>>(plio_mask_in.out[0], k_pobj0.in[1]);
        for (int h = 0; h < N_HEADS; h++) connect<window<x_sz>>(k_pobj0.out[h], cross0.x_in_h[h]);
        connect<window<x_sz>>(k_pobj0.out[N_HEADS], cross0.x_in_res);
        for (int h = 0; h < N_HEADS; h++) connect<window<c_sz>>(k_pobj0.out[5 + h], cand0.c_in_h[h]);
        connect<window<c_sz>>(k_pobj0.out[5 + N_HEADS], cand0.c_in_res);
        connect<window<c_sz>>(cand0.c_out, k_fan0.in[0]);
        for (int h = 0; h < N_HEADS; h++) connect<window<c_sz>>(k_fan0.out[h], cross0.c_in_h[h]);
        connect<window<c_sz>>(k_fan0.out[N_HEADS], plio_c_out.in[0]);
        connect<window<x_sz>>(cross0.x_out, plio_x_out.in[0]);
    }
};
Chain1Graph g;
int main(void) { g.init(); g.run(4); g.end(); return 0; }
