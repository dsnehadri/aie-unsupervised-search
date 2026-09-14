// The whole two-layer ABC stack on the array (2026-09-13). The six attention
// subgraphs are the ones in aie_graph.h with graph ports instead of PLIOs, so
// the blocks hand their windows to each other on the array; small tile kernels
// (chain_kernels.h) do what the fabric did between blocks (mask row, remask,
// candidate build); windows with more than two consumers go through trees of
// 2-output copy kernels (a tile has two MM2S DMA channels). PLIOs left: raw jets and the
// mask in, the four wij slices in (layer 0), x and c out for the lorentz /
// autoencoder stages on the fabric. 8 PLIOs instead of 20.
#ifndef AIE_CHAIN_GRAPH_H
#define AIE_CHAIN_GRAPH_H
#include "aie_graph.h"
#include "../../attn_block_aie/kernels/chain_kernels.h"
template <int LAYER, int INST = 0>
class ObjChainL : public graph {
public:
    port<input> x_in_h[N_HEADS], x_in_res;
    port<input> wij_h0, wij_h1, wij_h2, wij_h3;
    port<output> x_out;
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
            connect<window<x_sz>>(x_in_h[h], k_pre[h].in[0]);
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
        connect<window<x_sz>>(x_in_res, k_post_ap.in[N_HEADS]);

        // post_a_proj -> post_b1 (ffn0) and post_a_proj -> post_c (FFN-residual broadcast)
#ifndef POST_MERGED
        connect<window<proj_sz>>(k_post_ap.out[0], k_post_b1.in[0]);
        connect<window<proj_sz>>(k_post_ap.out[0], k_post_c.in[1]);

        // post_b1 -> post_b2 -> post_c -> PLIO
        connect<window<proj_sz>>(k_post_b1.out[0], k_post_b2.in[0]);
        connect<window<proj_sz>>(k_post_b2.out[0], k_post_c.in[0]);
        connect<window<x_out_sz>>(k_post_c.out[0], x_out);
#else
        connect<window<proj_sz>>(k_post_ap.out[0], k_post_bc.in[0]);
        connect<window<x_out_sz>>(k_post_bc.out[0], x_out);
#endif
    }
};
template <int LAYER, int INST = 0>
class CandChainL : public graph {
public:
    port<input> c_in_h[N_HEADS], c_in_res;
    port<output> c_out;
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
            connect<window<c_sz>>(c_in_h[h], k_pre[h].in[0]);
            connect<window<scores_sz>>(k_pre[h].out[0], k_post_h[h].in[0]);
            connect<window<v_sz>>     (k_pre[h].out[1], k_post_h[h].in[1]);
            connect<window<hout>>(k_post_h[h].out[0], k_post_ap.in[h]);
        }
        connect<window<c_sz>>(c_in_res, k_post_ap.in[N_HEADS]);

#ifndef POST_MERGED
        connect<window<proj_sz>>(k_post_ap.out[0], k_post_b1.in[0]);
        connect<window<proj_sz>>(k_post_ap.out[0], k_post_c.in[1]);
        connect<window<proj_sz>>(k_post_b1.out[0], k_post_b2.in[0]);
        connect<window<proj_sz>>(k_post_b2.out[0], k_post_c.in[0]);
        connect<window<c_sz>>(k_post_c.out[0], c_out);
#else
        connect<window<proj_sz>>(k_post_ap.out[0], k_post_bc.in[0]);
        connect<window<c_sz>>(k_post_bc.out[0], c_out);
#endif
    }
};
template <int LAYER, int INST = 0>
class CrossChainL : public graph {
public:
    port<input> x_in_h[N_HEADS], x_in_res;
    port<input> c_in_h[N_HEADS];
    port<output> x_out;
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
            connect<window<x_sz>>(x_in_h[h], k_pre[h].in[0]);
            connect<window<c_sz>>(c_in_h[h], k_pre[h].in[1]);
            connect<window<scores_sz>>(k_pre[h].out[0], k_post_h[h].in[0]);
            connect<window<v_sz>>     (k_pre[h].out[1], k_post_h[h].in[1]);
            connect<window<hout>>(k_post_h[h].out[0], k_post_ap.in[h]);
        }
        connect<window<x_sz>>(x_in_res, k_post_ap.in[N_HEADS]);

#ifndef POST_MERGED
        connect<window<proj_sz>>(k_post_ap.out[0], k_post_b1.in[0]);
        connect<window<proj_sz>>(k_post_ap.out[0], k_post_c.in[1]);
        connect<window<proj_sz>>(k_post_b1.out[0], k_post_b2.in[0]);
        connect<window<proj_sz>>(k_post_b2.out[0], k_post_c.in[0]);
        connect<window<x_sz>>(k_post_c.out[0], x_out);
#else
        connect<window<proj_sz>>(k_post_ap.out[0], k_post_bc.in[0]);
        connect<window<x_sz>>(k_post_bc.out[0], x_out);
#endif
    }
};

// Fan-out trees of 2-output copy kernels (each tile: 1 window in, 2 out).
//   fan4: R -> A -> (d0, d1), R -> B -> (d2, d3)            3 kernels, 2 hops
//   fan5: R -> A -> (d0, d1), R -> B -> (d2, C -> (d3, d4)) 4 kernels, 2-3 hops
class PasswdChainGraph : public graph {
public:
    input_plio  plio_jets_in, plio_mask_in;
    input_plio  plio_wij_h0, plio_wij_h1, plio_wij_h2, plio_wij_h3;
    output_plio plio_x_out, plio_c_out;
    kernel k_embed, k_asm0, k_pobj0, k_asm1, k_pobj1;
    static constexpr int NDUP = 40;
    kernel dup[NDUP]; int ndup = 0;
    ObjChainL<0> obj0;  CandChainL<0> cand0;  CrossChainL<0> cross0;
    ObjChainL<1> obj1;  CandChainL<1> cand1;  CrossChainL<1> cross1;

    static constexpr int jets_sz = EMBED_IN_WORDS * sizeof(int16);       // 128 B
    static constexpr int mask_sz = E_DIM * sizeof(int16);                //  32 B
    static constexpr int x_sz    = N_MAX * E_DIM * sizeof(int16);        // 384 B
    static constexpr int xm_sz   = (N_MAX + 1) * E_DIM * sizeof(int16);  // 416 B, with the mask row
    static constexpr int c_sz    = T_DIM * E_DIM * sizeof(int16);        //  96 B
    static constexpr int wij_sz  = N_MAX * N_KV * sizeof(int16);         // 312 B

    // the graph front-end needs the kernel function named literally at each
    // kernel::create, so the copy kernel is chosen by size with plain ifs
    kernel& mkdup(int sz) {
        kernel& k = dup[ndup++];
        if (sz == xm_sz) {
            k = kernel::create(chain_dup2_208);
        } else if (sz == x_sz) {
            k = kernel::create(chain_dup2_192);
        } else {
            k = kernel::create(chain_dup2_48);
        }
        source(k) = "kernels/chain_kernels.cc"; runtime<ratio>(k) = 0.9;
        return k;
    }
    template <int SZ, typename SRC>
    void fan4(SRC& src, port<input>& d0, port<input>& d1, port<input>& d2, port<input>& d3) {
        kernel& R = mkdup(SZ); kernel& A = mkdup(SZ); kernel& B = mkdup(SZ);
        connect<window<SZ>>(src, R.in[0]);
        connect<window<SZ>>(R.out[0], A.in[0]); connect<window<SZ>>(R.out[1], B.in[0]);
        connect<window<SZ>>(A.out[0], d0); connect<window<SZ>>(A.out[1], d1);
        connect<window<SZ>>(B.out[0], d2); connect<window<SZ>>(B.out[1], d3);
    }
    template <int SZ, typename SRC, typename D4>
    void fan5(SRC& src, port<input>& d0, port<input>& d1, port<input>& d2, port<input>& d3, D4& d4) {
        kernel& R = mkdup(SZ); kernel& A = mkdup(SZ); kernel& B = mkdup(SZ); kernel& C = mkdup(SZ);
        connect<window<SZ>>(src, R.in[0]);
        connect<window<SZ>>(R.out[0], A.in[0]); connect<window<SZ>>(R.out[1], B.in[0]);
        connect<window<SZ>>(A.out[0], d0); connect<window<SZ>>(A.out[1], d1);
        connect<window<SZ>>(B.out[0], d2); connect<window<SZ>>(B.out[1], C.in[0]);
        connect<window<SZ>>(C.out[0], d3); connect<window<SZ>>(C.out[1], d4);
    }

    PasswdChainGraph() {
        plio_jets_in = input_plio::create("embed_jets_in", plio_64_bits, "data/embed_jets_in.txt");
        plio_mask_in = input_plio::create("mask_in",       plio_64_bits, "data/mask_in.txt");
        plio_wij_h0  = input_plio::create("obj_wij_h0_L0", plio_64_bits, "data/obj_wij_h0_L0.txt");
        plio_wij_h1  = input_plio::create("obj_wij_h1_L0", plio_64_bits, "data/obj_wij_h1_L0.txt");
        plio_wij_h2  = input_plio::create("obj_wij_h2_L0", plio_64_bits, "data/obj_wij_h2_L0.txt");
        plio_wij_h3  = input_plio::create("obj_wij_h3_L0", plio_64_bits, "data/obj_wij_h3_L0.txt");
        plio_x_out   = output_plio::create("chain_x_out", plio_64_bits, "data/chain_x_out.txt");
        plio_c_out   = output_plio::create("chain_c_out", plio_64_bits, "data/chain_c_out.txt");

        k_embed = kernel::create(embed_mlp);           source(k_embed) = "kernels/embed_kernel.cc";
        k_asm0  = kernel::create(chain_assemble_zero); source(k_asm0)  = "kernels/chain_kernels.cc";
        k_pobj0 = kernel::create(chain_post_obj);      source(k_pobj0) = "kernels/chain_kernels.cc";
        k_asm1  = kernel::create(chain_assemble);      source(k_asm1)  = "kernels/chain_kernels.cc";
        k_pobj1 = kernel::create(chain_post_obj);      source(k_pobj1) = "kernels/chain_kernels.cc";
        for (kernel* k : {&k_embed, &k_asm0, &k_pobj0, &k_asm1, &k_pobj1}) runtime<ratio>(*k) = 0.9;

        // embedding -> (zero padded rows, + mask row) -> object L0 (5 consumers)
        connect<window<jets_sz>>(plio_jets_in.out[0], k_embed.in[0]);
        connect<window<x_sz>>(k_embed.out[0], k_asm0.in[0]);
        connect<window<mask_sz>>(plio_mask_in.out[0], k_asm0.in[1]);
        fan5<xm_sz>(k_asm0.out[0], obj0.x_in_h[0], obj0.x_in_h[1], obj0.x_in_h[2], obj0.x_in_h[3], obj0.x_in_res);
        connect<window<wij_sz>>(plio_wij_h0.out[0], obj0.wij_h0);
        connect<window<wij_sz>>(plio_wij_h1.out[0], obj0.wij_h1);
        connect<window<wij_sz>>(plio_wij_h2.out[0], obj0.wij_h2);
        connect<window<wij_sz>>(plio_wij_h3.out[0], obj0.wij_h3);
        // object L0 -> remask + candidate build -> cross L0 (x, 5), candidate L0 (c, 5)
        connect<window<x_sz>>(obj0.x_out, k_pobj0.in[0]);
        connect<window<mask_sz>>(plio_mask_in.out[0], k_pobj0.in[1]);
        fan5<x_sz>(k_pobj0.out[0], cross0.x_in_h[0], cross0.x_in_h[1], cross0.x_in_h[2], cross0.x_in_h[3], cross0.x_in_res);
        fan5<c_sz>(k_pobj0.out[1], cand0.c_in_h[0], cand0.c_in_h[1], cand0.c_in_h[2], cand0.c_in_h[3], cand0.c_in_res);
        // candidate L0 -> c (4) -> cross L0
        fan4<c_sz>(cand0.c_out, cross0.c_in_h[0], cross0.c_in_h[1], cross0.c_in_h[2], cross0.c_in_h[3]);
        // layer 1
        connect<window<x_sz>>(cross0.x_out, k_asm1.in[0]);
        connect<window<mask_sz>>(plio_mask_in.out[0], k_asm1.in[1]);
        fan5<xm_sz>(k_asm1.out[0], obj1.x_in_h[0], obj1.x_in_h[1], obj1.x_in_h[2], obj1.x_in_h[3], obj1.x_in_res);
        connect<window<x_sz>>(obj1.x_out, k_pobj1.in[0]);
        connect<window<mask_sz>>(plio_mask_in.out[0], k_pobj1.in[1]);
        fan5<x_sz>(k_pobj1.out[0], cross1.x_in_h[0], cross1.x_in_h[1], cross1.x_in_h[2], cross1.x_in_h[3], cross1.x_in_res);
        fan5<c_sz>(k_pobj1.out[1], cand1.c_in_h[0], cand1.c_in_h[1], cand1.c_in_h[2], cand1.c_in_h[3], cand1.c_in_res);
        // candidate L1 -> c (4 cross heads + the c PLIO out)
        fan5<c_sz>(cand1.c_out, cross1.c_in_h[0], cross1.c_in_h[1], cross1.c_in_h[2], cross1.c_in_h[3], plio_c_out.in[0]);
        // out: x after cross L1
        connect<window<x_sz>>(cross1.x_out, plio_x_out.in[0]);
    }
};
#endif // AIE_CHAIN_GRAPH_H
