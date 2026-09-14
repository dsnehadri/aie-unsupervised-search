// The whole two-layer ABC stack on the array (2026-09-13). The six attention
// subgraphs are the ones in aie_graph.h with graph ports instead of PLIOs; the
// tile kernels of chain_kernels.h do what the fabric did between blocks and
// emit STREAMS, which the switch multicasts to the 4-5 window inputs of a
// block (connect<stream, window<>>), exactly as a PLIO feeds a block. PLIOs
// left: raw jets and the mask in, the four wij slices in (layer 0), x after
// cross L1 and c after cand L1 out for the lorentz / autoencoder stages on the
// fabric. 8 PLIOs instead of 20.
#ifndef AIE_CHAIN_GRAPH_H
#define AIE_CHAIN_GRAPH_H
// The interface tiles run at the fabric clock, so the PLIO rate has to be
// declared with it: -DPLIO_FREQ_MHZ=120 alongside a 120 MHz link.
#ifndef PLIO_FREQ_MHZ
#define PLIO_FREQ_MHZ 100
#endif
#include "aie_graph.h"
#include "../../attn_block_aie/kernels/chain_kernels.h"
template <int LAYER, int INST = 0>
class ObjChainL : public graph {
public:
    port<input> wij_h0, wij_h1, wij_h2, wij_h3;
    // wij PLIOs exist only for layer 0 (layer 1 has no wij bias; the old
    // graph streamed 624 zeros/event through 4 dummy PLIOs)
public:
    kernel k_pre[N_HEADS];
    kernel k_post_h[N_HEADS];
#ifdef POST_MERGED
    kernel k_post_ap, k_post_bc;      // b1 + b2 + c in one kernel
#else
    kernel k_post_ap, k_post_b1, k_post_b2, k_post_c;
#endif
public:
    ObjChainL() {
        const std::string suffix = "_L" + std::to_string(LAYER) +
            (INST > 0 ? ("_i" + std::to_string(INST)) : std::string(""));

        if constexpr (LAYER == 0) {
            k_pre[0] = kernel::create(obj_attn_head_pre_h0_L0);
            k_pre[1] = kernel::create(obj_attn_head_pre_h1_L0);
            k_pre[2] = kernel::create(obj_attn_head_pre_h2_L0);
            k_pre[3] = kernel::create(obj_attn_head_pre_h3_L0);
            k_post_h[0] = kernel::create(obj_attn_head_post_h0_L0);
            k_post_h[1] = kernel::create(obj_attn_head_post_h1_L0);
            k_post_h[2] = kernel::create(obj_attn_head_post_h2_L0);
            k_post_h[3] = kernel::create(obj_attn_head_post_h3_L0);
        } else {
            k_pre[0] = kernel::create(obj_attn_head_pre_h0_L1);
            k_pre[1] = kernel::create(obj_attn_head_pre_h1_L1);
            k_pre[2] = kernel::create(obj_attn_head_pre_h2_L1);
            k_pre[3] = kernel::create(obj_attn_head_pre_h3_L1);
            k_post_h[0] = kernel::create(obj_attn_head_post_h0_L1);
            k_post_h[1] = kernel::create(obj_attn_head_post_h1_L1);
            k_post_h[2] = kernel::create(obj_attn_head_post_h2_L1);
            k_post_h[3] = kernel::create(obj_attn_head_post_h3_L1);
        }
        for (int h = 0; h < N_HEADS; h++) {
            source(k_pre[h]) = ("kernels/obj_head" + std::to_string(h) +
                                "_pre_L" + std::to_string(LAYER) + ".cc").c_str();
            runtime<ratio>(k_pre[h]) = 0.9;
            source(k_post_h[h]) = ("kernels/obj_head" + std::to_string(h) +
                                "_post_L" + std::to_string(LAYER) + ".cc").c_str();
            runtime<ratio>(k_post_h[h]) = 0.9;
        }

        if constexpr (LAYER == 0) {
            k_post_ap = kernel::create(obj_post_a_proj_L0);
#ifndef POST_MERGED
            k_post_b1 = kernel::create(obj_post_b1_L0);
            k_post_b2 = kernel::create(obj_post_b2_L0);
            k_post_c  = kernel::create(obj_post_c_L0);
#else
            k_post_bc = kernel::create(obj_post_bc_L0);
#endif
        } else {
            k_post_ap = kernel::create(obj_post_a_proj_L1);
#ifndef POST_MERGED
            k_post_b1 = kernel::create(obj_post_b1_L1);
            k_post_b2 = kernel::create(obj_post_b2_L1);
            k_post_c  = kernel::create(obj_post_c_L1);
#else
            k_post_bc = kernel::create(obj_post_bc_L1);
#endif
        }
        source(k_post_ap) = ("kernels/obj_post_ap_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_ap) = 0.9;
#ifndef POST_MERGED
        source(k_post_b1) = ("kernels/obj_post_b1_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_b1) = 0.9;
        source(k_post_b2) = ("kernels/obj_post_b2_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_b2) = 0.9;
        source(k_post_c) = ("kernels/obj_post_c_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_c) = 0.9;
#else
        source(k_post_bc) = ("kernels/obj_post_bc_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_bc) = 0.9;
#endif

        // window sizes. obj x INPUT carries N_MAX+1 rows: row N_MAX is the
        // padding mask (nonzero = padded), giving both layers true key
        // masking. The output stays N_MAX rows.
        constexpr int x_sz       = (N_MAX + 1) * E_DIM * sizeof(aiedt);
        constexpr int x_out_sz   = N_MAX * E_DIM * sizeof(aiedt);
        constexpr int wij_sz     = N_MAX * N_KV * sizeof(aiedt);
#ifdef TRANSPOSED
        // S^T 16 keys x 16 lanes, V^T 4 x 16, O^T 4 x 16, proj^T 16 x 16
        constexpr int scores_sz  = 16 * 16 * sizeof(aiedt);
        constexpr int v_sz       = D_HEAD * 16 * sizeof(aiedt);
        constexpr int hout       = D_HEAD * 16 * sizeof(aiedt);
        constexpr int concat_sz  = 16 * 16 * sizeof(aiedt);
        constexpr int proj_sz    = 16 * 16 * sizeof(aiedt);
#else
        constexpr int scores_sz  = N_MAX * N_KV_PAD * sizeof(aiedt);
        constexpr int v_sz       = N_KV_PAD * D_HEAD * sizeof(aiedt);
        constexpr int hout       = N_MAX * D_HEAD * sizeof(aiedt);
        constexpr int concat_sz  = N_MAX * E_DIM * sizeof(aiedt);
        constexpr int proj_sz    = N_MAX * E_DIM * sizeof(aiedt);
#endif

        // plio -> pre (X for all 4 heads)
        for (int h = 0; h < N_HEADS; h++) {
        }

        // pre -> post_h: scores + V
        for (int h = 0; h < N_HEADS; h++) {
            connect<window<scores_sz>>(k_pre[h].out[0], k_post_h[h].in[0]);
            connect<window<v_sz>>     (k_pre[h].out[1], k_post_h[h].in[1]);
        }

        // wij PLIOs -> post_h (layer 0 only; L1 kernels have no wij port)
        if constexpr (LAYER == 0) {
            connect<window<wij_sz>>(wij_h0, k_post_h[0].in[2]);
            connect<window<wij_sz>>(wij_h1, k_post_h[1].in[2]);
            connect<window<wij_sz>>(wij_h2, k_post_h[2].in[2]);
            connect<window<wij_sz>>(wij_h3, k_post_h[3].in[2]);
        }

        // head_post -> post_a_proj directly (the concat tile is gone),
        // residual X -> post_a_proj
        for (int h = 0; h < N_HEADS; h++) {
            connect<window<hout>>(k_post_h[h].out[0], k_post_ap.in[h]);
        }

        // post_a_proj -> post_b1 (ffn0) and post_a_proj -> post_c (FFN-residual broadcast)
#if defined(POST_STREAM)
        // rows stream a_proj -> b1 -> b2 -> c; the block output is a stream
        connect<stream>(k_post_ap.out[0], k_post_b1.in[0]);
        connect<stream>(k_post_ap.out[0], k_post_c.in[1]);
        connect<stream>(k_post_b1.out[0], k_post_b2.in[0]);
        connect<stream>(k_post_b2.out[0], k_post_c.in[0]);
        // block output: k_post_c.out[0] (stream), connected by the top graph
#elif !defined(POST_MERGED)
        connect<window<proj_sz>>(k_post_ap.out[0], k_post_b1.in[0]);
        connect<window<proj_sz>>(k_post_ap.out[0], k_post_c.in[1]);

        // post_b1 -> post_b2 -> post_c -> PLIO
        connect<window<proj_sz>>(k_post_b1.out[0], k_post_b2.in[0]);
        connect<window<proj_sz>>(k_post_b2.out[0], k_post_c.in[0]);
#else
        connect<window<proj_sz>>(k_post_ap.out[0], k_post_bc.in[0]);
#endif
    }
};
template <int LAYER, int INST = 0>
class CandChainL : public graph {
public:
public:
    kernel k_pre[N_HEADS];
    kernel k_post_h[N_HEADS];
#ifdef POST_MERGED
    kernel k_post_ap, k_post_bc;      // b1 + b2 + c in one kernel
#else
    kernel k_post_ap, k_post_b1, k_post_b2, k_post_c;
#endif
public:
    CandChainL() {
        const std::string suffix = "_L" + std::to_string(LAYER);

        if constexpr (LAYER == 0) {
            k_pre[0] = kernel::create(cand_attn_head_pre_h0_L0);
            k_pre[1] = kernel::create(cand_attn_head_pre_h1_L0);
            k_pre[2] = kernel::create(cand_attn_head_pre_h2_L0);
            k_pre[3] = kernel::create(cand_attn_head_pre_h3_L0);
            k_post_h[0] = kernel::create(cand_attn_head_post_h0_L0);
            k_post_h[1] = kernel::create(cand_attn_head_post_h1_L0);
            k_post_h[2] = kernel::create(cand_attn_head_post_h2_L0);
            k_post_h[3] = kernel::create(cand_attn_head_post_h3_L0);
        } else {
            k_pre[0] = kernel::create(cand_attn_head_pre_h0_L1);
            k_pre[1] = kernel::create(cand_attn_head_pre_h1_L1);
            k_pre[2] = kernel::create(cand_attn_head_pre_h2_L1);
            k_pre[3] = kernel::create(cand_attn_head_pre_h3_L1);
            k_post_h[0] = kernel::create(cand_attn_head_post_h0_L1);
            k_post_h[1] = kernel::create(cand_attn_head_post_h1_L1);
            k_post_h[2] = kernel::create(cand_attn_head_post_h2_L1);
            k_post_h[3] = kernel::create(cand_attn_head_post_h3_L1);
        }
        for (int h = 0; h < N_HEADS; h++) {
            source(k_pre[h]) = ("kernels/cand_head" + std::to_string(h) +
                                "_pre_L" + std::to_string(LAYER) + ".cc").c_str();
            runtime<ratio>(k_pre[h]) = 0.9;
            source(k_post_h[h]) = ("kernels/cand_head" + std::to_string(h) +
                                "_post_L" + std::to_string(LAYER) + ".cc").c_str();
            runtime<ratio>(k_post_h[h]) = 0.9;
        }

        if constexpr (LAYER == 0) {
            k_post_ap = kernel::create(cand_post_a_proj_L0);
#ifndef POST_MERGED
            k_post_b1 = kernel::create(cand_post_b1_L0);
            k_post_b2 = kernel::create(cand_post_b2_L0);
            k_post_c  = kernel::create(cand_post_c_L0);
#else
            k_post_bc = kernel::create(cand_post_bc_L0);
#endif
        } else {
            k_post_ap = kernel::create(cand_post_a_proj_L1);
#ifndef POST_MERGED
            k_post_b1 = kernel::create(cand_post_b1_L1);
            k_post_b2 = kernel::create(cand_post_b2_L1);
            k_post_c  = kernel::create(cand_post_c_L1);
#else
            k_post_bc = kernel::create(cand_post_bc_L1);
#endif
        }
        source(k_post_ap) = ("kernels/cand_post_ap_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_ap) = 0.9;
#ifndef POST_MERGED
        source(k_post_b1) = ("kernels/cand_post_b1_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_b1) = 0.9;
        source(k_post_b2) = ("kernels/cand_post_b2_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_b2) = 0.9;
        source(k_post_c) = ("kernels/cand_post_c_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_c) = 0.9;
#else
        source(k_post_bc) = ("kernels/cand_post_bc_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_bc) = 0.9;
#endif

        constexpr int c_sz      = T_DIM * E_DIM * sizeof(aiedt);
        // (the candidate block keeps the vector-I/O kernels under TRANSPOSED:
        // three candidates do not fill 16 lanes)
        constexpr int scores_sz = 4 * T_KV * sizeof(aiedt);
        constexpr int v_sz      = T_KV * D_HEAD * sizeof(aiedt);
        constexpr int hout      = T_DIM * D_HEAD * sizeof(aiedt);
        constexpr int concat_sz = T_DIM * E_DIM * sizeof(aiedt);
        constexpr int proj_sz   = T_DIM * E_DIM * sizeof(aiedt);

        for (int h = 0; h < N_HEADS; h++) {
            connect<window<scores_sz>>(k_pre[h].out[0], k_post_h[h].in[0]);
            connect<window<v_sz>>     (k_pre[h].out[1], k_post_h[h].in[1]);
            connect<window<hout>>(k_post_h[h].out[0], k_post_ap.in[h]);
        }

#if defined(POST_STREAM)
        // rows stream a_proj -> b1 -> b2 -> c; the block output is a stream
        connect<stream>(k_post_ap.out[0], k_post_b1.in[0]);
        connect<stream>(k_post_ap.out[0], k_post_c.in[1]);
        connect<stream>(k_post_b1.out[0], k_post_b2.in[0]);
        connect<stream>(k_post_b2.out[0], k_post_c.in[0]);
        // block output: k_post_c.out[0] (stream), connected by the top graph
#elif !defined(POST_MERGED)
        connect<window<proj_sz>>(k_post_ap.out[0], k_post_b1.in[0]);
        connect<window<proj_sz>>(k_post_ap.out[0], k_post_c.in[1]);
        connect<window<proj_sz>>(k_post_b1.out[0], k_post_b2.in[0]);
        connect<window<proj_sz>>(k_post_b2.out[0], k_post_c.in[0]);
#else
        connect<window<proj_sz>>(k_post_ap.out[0], k_post_bc.in[0]);
#endif
    }
};
template <int LAYER, int INST = 0>
class CrossChainL : public graph {
public:
public:
    kernel k_pre[N_HEADS];
    kernel k_post_h[N_HEADS];
#ifdef POST_MERGED
    kernel k_post_ap, k_post_bc;      // b1 + b2 + c in one kernel
#else
    kernel k_post_ap, k_post_b1, k_post_b2, k_post_c;
#endif
public:
    CrossChainL() {
        const std::string suffix = "_L" + std::to_string(LAYER);

        if constexpr (LAYER == 0) {
            k_pre[0] = kernel::create(cross_attn_head_pre_h0_L0);
            k_pre[1] = kernel::create(cross_attn_head_pre_h1_L0);
            k_pre[2] = kernel::create(cross_attn_head_pre_h2_L0);
            k_pre[3] = kernel::create(cross_attn_head_pre_h3_L0);
            k_post_h[0] = kernel::create(cross_attn_head_post_h0_L0);
            k_post_h[1] = kernel::create(cross_attn_head_post_h1_L0);
            k_post_h[2] = kernel::create(cross_attn_head_post_h2_L0);
            k_post_h[3] = kernel::create(cross_attn_head_post_h3_L0);
        } else {
            k_pre[0] = kernel::create(cross_attn_head_pre_h0_L1);
            k_pre[1] = kernel::create(cross_attn_head_pre_h1_L1);
            k_pre[2] = kernel::create(cross_attn_head_pre_h2_L1);
            k_pre[3] = kernel::create(cross_attn_head_pre_h3_L1);
            k_post_h[0] = kernel::create(cross_attn_head_post_h0_L1);
            k_post_h[1] = kernel::create(cross_attn_head_post_h1_L1);
            k_post_h[2] = kernel::create(cross_attn_head_post_h2_L1);
            k_post_h[3] = kernel::create(cross_attn_head_post_h3_L1);
        }
        for (int h = 0; h < N_HEADS; h++) {
            source(k_pre[h]) = ("kernels/cross_head" + std::to_string(h) +
                                "_pre_L" + std::to_string(LAYER) + ".cc").c_str();
            runtime<ratio>(k_pre[h]) = 0.9;
            source(k_post_h[h]) = ("kernels/cross_head" + std::to_string(h) +
                                "_post_L" + std::to_string(LAYER) + ".cc").c_str();
            runtime<ratio>(k_post_h[h]) = 0.9;
        }

        if constexpr (LAYER == 0) {
            k_post_ap = kernel::create(cross_post_a_proj_L0);
#ifndef POST_MERGED
            k_post_b1 = kernel::create(cross_post_b1_L0);
            k_post_b2 = kernel::create(cross_post_b2_L0);
            k_post_c  = kernel::create(cross_post_c_L0);
#else
            k_post_bc = kernel::create(cross_post_bc_L0);
#endif
        } else {
            k_post_ap = kernel::create(cross_post_a_proj_L1);
#ifndef POST_MERGED
            k_post_b1 = kernel::create(cross_post_b1_L1);
            k_post_b2 = kernel::create(cross_post_b2_L1);
            k_post_c  = kernel::create(cross_post_c_L1);
#else
            k_post_bc = kernel::create(cross_post_bc_L1);
#endif
        }
        source(k_post_ap) = ("kernels/cross_post_ap_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_ap) = 0.9;
#ifndef POST_MERGED
        source(k_post_b1) = ("kernels/cross_post_b1_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_b1) = 0.9;
        source(k_post_b2) = ("kernels/cross_post_b2_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_b2) = 0.9;
        source(k_post_c) = ("kernels/cross_post_c_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_c) = 0.9;
#else
        source(k_post_bc) = ("kernels/cross_post_bc_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_bc) = 0.9;
#endif

        constexpr int x_sz      = N_MAX * E_DIM * sizeof(aiedt);
        constexpr int c_sz      = T_DIM * E_DIM * sizeof(aiedt);
#ifdef TRANSPOSED
        constexpr int scores_sz = T_KV * 16 * sizeof(aiedt);
        constexpr int v_sz      = D_HEAD * T_KV * sizeof(aiedt);
        constexpr int hout      = D_HEAD * 16 * sizeof(aiedt);
        constexpr int concat_sz = 16 * 16 * sizeof(aiedt);
        constexpr int proj_sz   = 16 * 16 * sizeof(aiedt);
#else
        constexpr int scores_sz = N_MAX * T_KV * sizeof(aiedt);
        constexpr int v_sz      = T_KV * D_HEAD * sizeof(aiedt);
        constexpr int hout      = N_MAX * D_HEAD * sizeof(aiedt);
        constexpr int concat_sz = N_MAX * E_DIM * sizeof(aiedt);
        constexpr int proj_sz   = N_MAX * E_DIM * sizeof(aiedt);
#endif

        for (int h = 0; h < N_HEADS; h++) {
            connect<window<scores_sz>>(k_pre[h].out[0], k_post_h[h].in[0]);
            connect<window<v_sz>>     (k_pre[h].out[1], k_post_h[h].in[1]);
            connect<window<hout>>(k_post_h[h].out[0], k_post_ap.in[h]);
        }

#if defined(POST_STREAM)
        // rows stream a_proj -> b1 -> b2 -> c; the block output is a stream
        connect<stream>(k_post_ap.out[0], k_post_b1.in[0]);
        connect<stream>(k_post_ap.out[0], k_post_c.in[1]);
        connect<stream>(k_post_b1.out[0], k_post_b2.in[0]);
        connect<stream>(k_post_b2.out[0], k_post_c.in[0]);
        // block output: k_post_c.out[0] (stream), connected by the top graph
#elif !defined(POST_MERGED)
        connect<window<proj_sz>>(k_post_ap.out[0], k_post_b1.in[0]);
        connect<window<proj_sz>>(k_post_ap.out[0], k_post_c.in[1]);
        connect<window<proj_sz>>(k_post_b1.out[0], k_post_b2.in[0]);
        connect<window<proj_sz>>(k_post_b2.out[0], k_post_c.in[0]);
#else
        connect<window<proj_sz>>(k_post_ap.out[0], k_post_bc.in[0]);
#endif
    }
};

class PasswdChainGraph : public graph {
public:
    input_plio  plio_jets_in, plio_mask_in;
    input_plio  plio_wij_h0;
#if !defined(WIJ_ONE_PORT)
    input_plio  plio_wij_h1, plio_wij_h2, plio_wij_h3;
#endif
    output_plio plio_x_out, plio_c_out;
    kernel k_embed, k_asm0, k_pobj0, k_asm1, k_pobj1;
    ObjChainL<0> obj0;  CandChainL<0> cand0;  CrossChainL<0> cross0;
    ObjChainL<1> obj1;  CandChainL<1> cand1;  CrossChainL<1> cross1;
    PasswdChainGraph() {
        plio_jets_in = input_plio::create("embed_jets_in", plio_64_bits, "data/embed_jets_in.txt", PLIO_FREQ_MHZ);
        plio_mask_in = input_plio::create("mask_in",       plio_64_bits, "data/mask_in.txt", PLIO_FREQ_MHZ);
        plio_wij_h0  = input_plio::create("obj_wij_h0_L0", plio_64_bits, "data/obj_wij_h0_L0.txt", PLIO_FREQ_MHZ);
#if !defined(WIJ_ONE_PORT)
        plio_wij_h1  = input_plio::create("obj_wij_h1_L0", plio_64_bits, "data/obj_wij_h1_L0.txt", PLIO_FREQ_MHZ);
        plio_wij_h2  = input_plio::create("obj_wij_h2_L0", plio_64_bits, "data/obj_wij_h2_L0.txt", PLIO_FREQ_MHZ);
        plio_wij_h3  = input_plio::create("obj_wij_h3_L0", plio_64_bits, "data/obj_wij_h3_L0.txt", PLIO_FREQ_MHZ);
#endif
        plio_x_out   = output_plio::create("chain_x_out", plio_64_bits, "data/chain_x_out.txt", PLIO_FREQ_MHZ);
        plio_c_out   = output_plio::create("chain_c_out", plio_64_bits, "data/chain_c_out.txt", PLIO_FREQ_MHZ);

        k_embed = kernel::create(embed_mlp);           source(k_embed) = "kernels/embed_kernel.cc";
        k_asm0  = kernel::create(chain_assemble_zero); source(k_asm0)  = "kernels/chain_kernels.cc";
        k_pobj0 = kernel::create(chain_post_obj);      source(k_pobj0) = "kernels/chain_kernels.cc";
        k_asm1  = kernel::create(chain_assemble);      source(k_asm1)  = "kernels/chain_kernels.cc";
        k_pobj1 = kernel::create(chain_post_obj);      source(k_pobj1) = "kernels/chain_kernels.cc";
        for (kernel* k : {&k_embed, &k_asm0, &k_pobj0, &k_asm1, &k_pobj1}) runtime<ratio>(*k) = 0.9;

        constexpr int jets_sz = EMBED_IN_WORDS * sizeof(int16);       // 128 B
        constexpr int mask_sz = E_DIM * sizeof(int16);                //  32 B
        constexpr int x_sz    = N_MAX * E_DIM * sizeof(int16);        // 384 B
        constexpr int xm_sz   = (N_MAX + 1) * E_DIM * sizeof(int16);  // 416 B, with the mask row
        constexpr int c_sz    = T_DIM * E_DIM * sizeof(int16);        //  96 B
        constexpr int wij_sz  = N_MAX * N_KV * sizeof(int16);         // 312 B

        // embedding -> (zero padded rows, + mask row) -> stream -> object L0 (4 heads + residual)
        connect<window<jets_sz>>(plio_jets_in.out[0], k_embed.in[0]);
#if defined(CHAIN_STREAM)
        connect<stream>(k_embed.out[0], k_asm0.in[0]);
#else
        connect<window<x_sz>>(k_embed.out[0], k_asm0.in[0]);
#endif
        connect<window<mask_sz>>(plio_mask_in.out[0], k_asm0.in[1]);
#if defined(PRE_STREAM)
        for (int h = 0; h < N_HEADS; h++) connect<stream>(k_asm0.out[0], obj0.k_pre[h].in[0]);
#else
        for (int h = 0; h < N_HEADS; h++) connect<stream, window<xm_sz>>(k_asm0.out[0], obj0.k_pre[h].in[0]);
#endif
        connect<stream, window<xm_sz>>(k_asm0.out[0], obj0.k_post_ap.in[N_HEADS]);
#if defined(WIJ_ONE_PORT)
        // The fabric used to send the SAME wij slice four times, once per head.
        // One PLIO feeds all four head-post kernels instead: a PLIO already
        // multicasts to five kernels elsewhere in this graph.
        connect<window<wij_sz>>(plio_wij_h0.out[0], obj0.wij_h0);
        connect<window<wij_sz>>(plio_wij_h0.out[0], obj0.wij_h1);
        connect<window<wij_sz>>(plio_wij_h0.out[0], obj0.wij_h2);
        connect<window<wij_sz>>(plio_wij_h0.out[0], obj0.wij_h3);
#else
        connect<window<wij_sz>>(plio_wij_h0.out[0], obj0.wij_h0);
        connect<window<wij_sz>>(plio_wij_h1.out[0], obj0.wij_h1);
        connect<window<wij_sz>>(plio_wij_h2.out[0], obj0.wij_h2);
        connect<window<wij_sz>>(plio_wij_h3.out[0], obj0.wij_h3);
#endif
        // object L0 (stream out) -> remask + candidate build -> streams -> cross L0 (x), candidate L0 (c)
#if defined(CHAIN_STREAM)
        connect<stream>(obj0.k_post_c.out[0], k_pobj0.in[0]);
#else
        connect<stream, window<x_sz>>(obj0.k_post_c.out[0], k_pobj0.in[0]);
#endif
        connect<window<mask_sz>>(plio_mask_in.out[0], k_pobj0.in[1]);
#if defined(PRE_STREAM)
        for (int h = 0; h < N_HEADS; h++) connect<stream>(k_pobj0.out[0], cross0.k_pre[h].in[0]);
#else
        for (int h = 0; h < N_HEADS; h++) connect<stream, window<x_sz>>(k_pobj0.out[0], cross0.k_pre[h].in[0]);
#endif
        connect<stream, window<x_sz>>(k_pobj0.out[0], cross0.k_post_ap.in[N_HEADS]);
        for (int h = 0; h < N_HEADS; h++) connect<stream, window<c_sz>>(k_pobj0.out[1], cand0.k_pre[h].in[0]);
        connect<stream, window<c_sz>>(k_pobj0.out[1], cand0.k_post_ap.in[N_HEADS]);
        // candidate L0 (stream out) -> cross L0 heads
        for (int h = 0; h < N_HEADS; h++) connect<stream, window<c_sz>>(cand0.k_post_c.out[0], cross0.k_pre[h].in[1]);
        // layer 1
#if defined(CHAIN_STREAM)
        connect<stream>(cross0.k_post_c.out[0], k_asm1.in[0]);
#else
        connect<stream, window<x_sz>>(cross0.k_post_c.out[0], k_asm1.in[0]);
#endif
        connect<window<mask_sz>>(plio_mask_in.out[0], k_asm1.in[1]);
#if defined(PRE_STREAM)
        for (int h = 0; h < N_HEADS; h++) connect<stream>(k_asm1.out[0], obj1.k_pre[h].in[0]);
#else
        for (int h = 0; h < N_HEADS; h++) connect<stream, window<xm_sz>>(k_asm1.out[0], obj1.k_pre[h].in[0]);
#endif
        connect<stream, window<xm_sz>>(k_asm1.out[0], obj1.k_post_ap.in[N_HEADS]);
#if defined(CHAIN_STREAM)
        connect<stream>(obj1.k_post_c.out[0], k_pobj1.in[0]);
#else
        connect<stream, window<x_sz>>(obj1.k_post_c.out[0], k_pobj1.in[0]);
#endif
        connect<window<mask_sz>>(plio_mask_in.out[0], k_pobj1.in[1]);
#if defined(PRE_STREAM)
        for (int h = 0; h < N_HEADS; h++) connect<stream>(k_pobj1.out[0], cross1.k_pre[h].in[0]);
#else
        for (int h = 0; h < N_HEADS; h++) connect<stream, window<x_sz>>(k_pobj1.out[0], cross1.k_pre[h].in[0]);
#endif
        connect<stream, window<x_sz>>(k_pobj1.out[0], cross1.k_post_ap.in[N_HEADS]);
        for (int h = 0; h < N_HEADS; h++) connect<stream, window<c_sz>>(k_pobj1.out[1], cand1.k_pre[h].in[0]);
        connect<stream, window<c_sz>>(k_pobj1.out[1], cand1.k_post_ap.in[N_HEADS]);
        for (int h = 0; h < N_HEADS; h++) connect<stream, window<c_sz>>(cand1.k_post_c.out[0], cross1.k_pre[h].in[1]);
        // out: x after cross L1, c after candidate L1 (streams to the PL)
        connect<stream>(cross1.k_post_c.out[0], plio_x_out.in[0]);
        connect<stream>(cand1.k_post_c.out[0], plio_c_out.in[0]);
    }
};
#endif // AIE_CHAIN_GRAPH_H
