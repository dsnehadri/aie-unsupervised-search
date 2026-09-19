// instantiates 6 attention subgraphs; each head split across pre/post tiles

#ifndef PASSWD_FULL_GRAPH_H
#define PASSWD_FULL_GRAPH_H

#include <adf.h>

using namespace adf;

#include "../../attn_block_aie/kernels/attn_aie_types.h"
#include "../../attn_block_aie/kernels/attn_head_kernel.h"
#include "../../attn_block_aie/kernels/attn_post_kernel.h"
#include "../../attn_block_aie/kernels/embed_kernel.h"

// Residual port of the projection kernel: with HEAD_STREAM the object and cross
// projections read two merged head streams, so the residual is input 2. The
// candidate projection keeps its four head windows and stays at N_HEADS.
#if defined(HEAD_STREAM_OBJ)
#define AP_RESID_IN_OBJ 2
#else
#define AP_RESID_IN_OBJ N_HEADS
#endif
#if defined(HEAD_STREAM_CROSS)
#define AP_RESID_IN_CROSS 2
#else
#define AP_RESID_IN_CROSS N_HEADS
#endif


// Heads need 4 distinct kernel functions per layer (the aiecompiler dedups
// wrappers by function symbol identity). The aiecompiler can't see through a
// typedef'd function-pointer variable for signature introspection, so we
// dispatch with `if constexpr` over LAYER and a switch over h.

// obj attn subgraph

template <int LAYER, int INST = 0>
class ObjAttnGraphL : public graph {
public:
    input_plio  plio_x_in;
    // wij PLIOs exist only for layer 0 (layer 1 has no wij bias; the old
    // graph streamed 624 zeros/event through 4 dummy PLIOs)
    input_plio  plio_wij_h0, plio_wij_h1, plio_wij_h2, plio_wij_h3;
    output_plio plio_x_out;
public:
    kernel k_pre[N_HEADS];
    kernel k_post_h[N_HEADS];
#if defined(HEAD_STREAM_OBJ)
    kernel k_merge[2];                // pair the four head streams for the projection
#endif
#ifdef POST_MERGED
    kernel k_post_ap, k_post_bc;      // b1 + b2 + c in one kernel
#else
    kernel k_post_ap, k_post_b1, k_post_b2, k_post_c;
#if defined(POST_SPLIT_C)
    kernel k_post_c1;                 // FFN layer 2 + norm + ReLU; post_c keeps the residual add + norm
#if defined(ROW_SPLIT_OBJ)
    kernel k_hb[4];                   // second row chain (odd rows): b1, b2, c1, c
    kernel k_rowmerge;                // puts the two chains' rows back in order
#endif
#endif
#endif
public:
    ObjAttnGraphL() {
        const std::string suffix = "_L" + std::to_string(LAYER) +
            (INST > 0 ? ("_i" + std::to_string(INST)) : std::string(""));
        plio_x_in = input_plio::create("obj_x_in" + suffix, plio_64_bits,
                                        "data/obj_x_in" + suffix + ".txt");
        if constexpr (LAYER == 0) {
            plio_wij_h0 = input_plio::create("obj_wij_h0" + suffix, plio_64_bits,
                                            "data/obj_wij_h0" + suffix + ".txt");
            plio_wij_h1 = input_plio::create("obj_wij_h1" + suffix, plio_64_bits,
                                            "data/obj_wij_h1" + suffix + ".txt");
            plio_wij_h2 = input_plio::create("obj_wij_h2" + suffix, plio_64_bits,
                                            "data/obj_wij_h2" + suffix + ".txt");
            plio_wij_h3 = input_plio::create("obj_wij_h3" + suffix, plio_64_bits,
                                            "data/obj_wij_h3" + suffix + ".txt");
        }
        plio_x_out = output_plio::create("obj_x_out" + suffix, plio_64_bits,
                                        "data/obj_x_out" + suffix + ".txt");

        if constexpr (LAYER == 0) {
            k_pre[0] = kernel::create(obj_attn_head_pre_h0_L0);
            k_pre[1] = kernel::create(obj_attn_head_pre_h1_L0);
            k_pre[2] = kernel::create(obj_attn_head_pre_h2_L0);
            k_pre[3] = kernel::create(obj_attn_head_pre_h3_L0);
            k_post_h[0] = kernel::create(obj_attn_head_post_h0_L0);
            k_post_h[1] = kernel::create(obj_attn_head_post_h1_L0);
            k_post_h[2] = kernel::create(obj_attn_head_post_h2_L0);
            k_post_h[3] = kernel::create(obj_attn_head_post_h3_L0);
#if defined(HEAD_STREAM_OBJ)
            k_merge[0] = kernel::create(obj_head_merge0_L0);
            k_merge[1] = kernel::create(obj_head_merge1_L0);
#endif
        } else {
            k_pre[0] = kernel::create(obj_attn_head_pre_h0_L1);
            k_pre[1] = kernel::create(obj_attn_head_pre_h1_L1);
            k_pre[2] = kernel::create(obj_attn_head_pre_h2_L1);
            k_pre[3] = kernel::create(obj_attn_head_pre_h3_L1);
            k_post_h[0] = kernel::create(obj_attn_head_post_h0_L1);
            k_post_h[1] = kernel::create(obj_attn_head_post_h1_L1);
            k_post_h[2] = kernel::create(obj_attn_head_post_h2_L1);
            k_post_h[3] = kernel::create(obj_attn_head_post_h3_L1);
#if defined(HEAD_STREAM_OBJ)
            k_merge[0] = kernel::create(obj_head_merge0_L1);
            k_merge[1] = kernel::create(obj_head_merge1_L1);
#endif
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
#if defined(POST_SPLIT_C)
            k_post_c1 = kernel::create(obj_post_c1_L0);
#endif
#else
            k_post_bc = kernel::create(obj_post_bc_L0);
#endif
        } else {
            k_post_ap = kernel::create(obj_post_a_proj_L1);
#ifndef POST_MERGED
            k_post_b1 = kernel::create(obj_post_b1_L1);
            k_post_b2 = kernel::create(obj_post_b2_L1);
            k_post_c  = kernel::create(obj_post_c_L1);
#if defined(POST_SPLIT_C)
            k_post_c1 = kernel::create(obj_post_c1_L1);
#endif
#else
            k_post_bc = kernel::create(obj_post_bc_L1);
#endif
        }
        source(k_post_ap) = ("kernels/obj_post_ap_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_ap) = 0.9;
#if defined(HEAD_STREAM_OBJ)
        for (int m = 0; m < 2; m++) {
            source(k_merge[m]) = ("kernels/obj_head_merge" + std::to_string(m) + "_L" + std::to_string(LAYER) + ".cc").c_str();
            runtime<ratio>(k_merge[m]) = 0.9;
        }
#endif
#ifndef POST_MERGED
        source(k_post_b1) = ("kernels/obj_post_b1_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_b1) = 0.9;
        source(k_post_b2) = ("kernels/obj_post_b2_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_b2) = 0.9;
        source(k_post_c) = ("kernels/obj_post_c_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_c) = 0.9;
#if defined(POST_SPLIT_C)
        source(k_post_c1) = ("kernels/obj_post_c1_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_c1) = 0.9;
#endif
#else
        source(k_post_bc) = ("kernels/obj_post_bc_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_bc) = 0.9;
#endif

        // window sizes. obj x INPUT carries N_MAX+1 rows: row N_MAX is the
        // padding mask (nonzero = padded), giving both layers true key
        // masking. The output stays N_MAX rows.
        constexpr int x_sz       = (N_MAX + 1) * E_DIM * sizeof(aiedt);
        constexpr int x_out_sz   = N_MAX * E_DIM * sizeof(aiedt);
#if defined(WIJ_PAD16)
        constexpr int wij_sz     = N_MAX * 16 * sizeof(aiedt);   // padded to the score rows' 16 lanes
#else
        constexpr int wij_sz     = N_MAX * N_KV * sizeof(aiedt);
#endif
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

        // plio -> pre (X for all 4 heads). PRE_STREAM makes the pre kernels
        // read x as a row stream; in the chain graph that stream comes from
        // the previous block's glue kernel, here from the PLIO.
        for (int h = 0; h < N_HEADS; h++) {
#if defined(PRE_STREAM)
            connect<stream>(plio_x_in.out[0], k_pre[h].in[0]);
#else
            connect<window<x_sz>>(plio_x_in.out[0], k_pre[h].in[0]);
#endif
        }

        // pre -> post_h: scores + V
        for (int h = 0; h < N_HEADS; h++) {
#if defined(SCORE_STREAM)
            connect<stream>(k_pre[h].out[0], k_post_h[h].in[0]);        // V, then scores four rows at a time
#else
            connect<window<scores_sz>>(k_pre[h].out[0], k_post_h[h].in[0]);
            connect<window<v_sz>>     (k_pre[h].out[1], k_post_h[h].in[1]);
#endif
        }

        // wij PLIOs -> post_h (layer 0 only; L1 kernels have no wij port)
        if constexpr (LAYER == 0) {
#if defined(SCORE_STREAM)
            connect<window<wij_sz>>(plio_wij_h0.out[0], k_post_h[0].in[1]);
#else
            connect<window<wij_sz>>(plio_wij_h0.out[0], k_post_h[0].in[2]);
#endif
#if defined(SCORE_STREAM)
            connect<window<wij_sz>>(plio_wij_h1.out[0], k_post_h[1].in[1]);
#else
            connect<window<wij_sz>>(plio_wij_h1.out[0], k_post_h[1].in[2]);
#endif
#if defined(SCORE_STREAM)
            connect<window<wij_sz>>(plio_wij_h2.out[0], k_post_h[2].in[1]);
#else
            connect<window<wij_sz>>(plio_wij_h2.out[0], k_post_h[2].in[2]);
#endif
#if defined(SCORE_STREAM)
            connect<window<wij_sz>>(plio_wij_h3.out[0], k_post_h[3].in[1]);
#else
            connect<window<wij_sz>>(plio_wij_h3.out[0], k_post_h[3].in[2]);
#endif
        }

        // head_post -> post_a_proj directly (the concat tile is gone),
        // residual X -> post_a_proj
#if defined(HEAD_STREAM_OBJ)
        connect<stream>(k_post_h[0].out[0], k_merge[0].in[0]);
        connect<stream>(k_post_h[1].out[0], k_merge[0].in[1]);
        connect<stream>(k_post_h[2].out[0], k_merge[1].in[0]);
        connect<stream>(k_post_h[3].out[0], k_merge[1].in[1]);
        connect<stream>(k_merge[0].out[0], k_post_ap.in[0]);
        connect<stream>(k_merge[1].out[0], k_post_ap.in[1]);
#else
        for (int h = 0; h < N_HEADS; h++) {
            connect<window<hout>>(k_post_h[h].out[0], k_post_ap.in[h]);
        }
#endif
        connect<window<x_sz>>(plio_x_in.out[0], k_post_ap.in[AP_RESID_IN_OBJ]);

        // post_a_proj -> post_b1 (ffn0) and post_a_proj -> post_c (FFN-residual broadcast)
#if defined(POST_STREAM)
        // rows stream a_proj -> b1 -> b2 -> c; the block output is a stream
        connect<stream>(k_post_ap.out[0], k_post_b1.in[0]);
        connect<stream>(k_post_ap.out[0], k_post_c.in[1]);
        connect<stream>(k_post_b1.out[0], k_post_b2.in[0]);
#if defined(POST_SPLIT_C)
        connect<stream>(k_post_b2.out[0], k_post_c1.in[0]);
        connect<stream>(k_post_c1.out[0], k_post_c.in[0]);
#else
        connect<stream>(k_post_b2.out[0], k_post_c.in[0]);
#endif
#if defined(ROW_SPLIT_OBJ) && defined(POST_SPLIT_C)
        // ROW_SPLIT: a_proj's second output carries the odd rows through a copy of the chain
        if constexpr (LAYER == 0) {
            k_hb[0] = kernel::create(obj_post_b1_hb_L0); k_hb[1] = kernel::create(obj_post_b2_hb_L0);
            k_hb[2] = kernel::create(obj_post_c1_hb_L0); k_hb[3] = kernel::create(obj_post_c_hb_L0);
            k_rowmerge = kernel::create(obj_post_rowmerge_L0);
        } else {
            k_hb[0] = kernel::create(obj_post_b1_hb_L1); k_hb[1] = kernel::create(obj_post_b2_hb_L1);
            k_hb[2] = kernel::create(obj_post_c1_hb_L1); k_hb[3] = kernel::create(obj_post_c_hb_L1);
            k_rowmerge = kernel::create(obj_post_rowmerge_L1);
        }
        {
            const char* st[4] = {"b1", "b2", "c1", "c"};
            for (int i = 0; i < 4; i++) {
                source(k_hb[i]) = ("kernels/obj_post_" + std::string(st[i]) + "_hb_L" + std::to_string(LAYER) + ".cc").c_str();
                runtime<ratio>(k_hb[i]) = 0.9;
            }
            source(k_rowmerge) = ("kernels/obj_post_rowmerge_L" + std::to_string(LAYER) + ".cc").c_str();
            runtime<ratio>(k_rowmerge) = 0.9;
        }
        connect<stream>(k_post_ap.out[1], k_hb[0].in[0]);
        connect<stream>(k_post_ap.out[1], k_hb[3].in[1]);
        connect<stream>(k_hb[0].out[0], k_hb[1].in[0]);
        connect<stream>(k_hb[1].out[0], k_hb[2].in[0]);
        connect<stream>(k_hb[2].out[0], k_hb[3].in[0]);
        connect<stream>(k_post_c.out[0], k_rowmerge.in[0]);
        connect<stream>(k_hb[3].out[0], k_rowmerge.in[1]);
        connect<stream>(k_rowmerge.out[0], plio_x_out.in[0]);
#else
        connect<stream>(k_post_c.out[0], plio_x_out.in[0]);
#endif
#elif !defined(POST_MERGED)
        connect<window<proj_sz>>(k_post_ap.out[0], k_post_b1.in[0]);
        connect<window<proj_sz>>(k_post_ap.out[0], k_post_c.in[1]);

        // post_b1 -> post_b2 -> post_c -> PLIO
        connect<window<proj_sz>>(k_post_b1.out[0], k_post_b2.in[0]);
        connect<window<proj_sz>>(k_post_b2.out[0], k_post_c.in[0]);
        connect<window<x_out_sz>>(k_post_c.out[0], plio_x_out.in[0]);
#else
        connect<window<proj_sz>>(k_post_ap.out[0], k_post_bc.in[0]);
        connect<window<x_out_sz>>(k_post_bc.out[0], plio_x_out.in[0]);
#endif
    }
};

// cand attn subgraph

template <int LAYER>
class CandAttnGraphL : public graph {
public:
    input_plio  plio_c_in;
    output_plio plio_c_out;
public:
    kernel k_pre[N_HEADS];
    kernel k_post_h[N_HEADS];
#ifdef POST_MERGED
    kernel k_post_ap, k_post_bc;      // b1 + b2 + c in one kernel
#else
    kernel k_post_ap, k_post_b1, k_post_b2, k_post_c;
#if defined(POST_SPLIT_C)
    kernel k_post_c1;                 // FFN layer 2 + norm + ReLU; post_c keeps the residual add + norm
#endif
#endif
public:
    CandAttnGraphL() {
        const std::string suffix = "_L" + std::to_string(LAYER);
        plio_c_in = input_plio::create("cand_c_in" + suffix, plio_64_bits,
                                        "data/cand_c_in" + suffix + ".txt");
        plio_c_out = output_plio::create("cand_c_out" + suffix, plio_64_bits,
                                        "data/cand_c_out" + suffix + ".txt");

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
#if defined(POST_SPLIT_C)
            k_post_c1 = kernel::create(cand_post_c1_L0);
#endif
#else
            k_post_bc = kernel::create(cand_post_bc_L0);
#endif
        } else {
            k_post_ap = kernel::create(cand_post_a_proj_L1);
#ifndef POST_MERGED
            k_post_b1 = kernel::create(cand_post_b1_L1);
            k_post_b2 = kernel::create(cand_post_b2_L1);
            k_post_c  = kernel::create(cand_post_c_L1);
#if defined(POST_SPLIT_C)
            k_post_c1 = kernel::create(cand_post_c1_L1);
#endif
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
#if defined(POST_SPLIT_C)
        source(k_post_c1) = ("kernels/cand_post_c1_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_c1) = 0.9;
#endif
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
            connect<window<c_sz>>(plio_c_in.out[0], k_pre[h].in[0]);
            connect<window<scores_sz>>(k_pre[h].out[0], k_post_h[h].in[0]);
            connect<window<v_sz>>     (k_pre[h].out[1], k_post_h[h].in[1]);
            connect<window<hout>>(k_post_h[h].out[0], k_post_ap.in[h]);
        }
        connect<window<c_sz>>(plio_c_in.out[0], k_post_ap.in[N_HEADS]);

#if defined(POST_STREAM)
        // rows stream a_proj -> b1 -> b2 -> c; the block output is a stream
        connect<stream>(k_post_ap.out[0], k_post_b1.in[0]);
        connect<stream>(k_post_ap.out[0], k_post_c.in[1]);
        connect<stream>(k_post_b1.out[0], k_post_b2.in[0]);
#if defined(POST_SPLIT_C)
        connect<stream>(k_post_b2.out[0], k_post_c1.in[0]);
        connect<stream>(k_post_c1.out[0], k_post_c.in[0]);
#else
        connect<stream>(k_post_b2.out[0], k_post_c.in[0]);
#endif
        connect<stream>(k_post_c.out[0], plio_c_out.in[0]);
#elif !defined(POST_MERGED)
        connect<window<proj_sz>>(k_post_ap.out[0], k_post_b1.in[0]);
        connect<window<proj_sz>>(k_post_ap.out[0], k_post_c.in[1]);
        connect<window<proj_sz>>(k_post_b1.out[0], k_post_b2.in[0]);
        connect<window<proj_sz>>(k_post_b2.out[0], k_post_c.in[0]);
        connect<window<c_sz>>(k_post_c.out[0], plio_c_out.in[0]);
#else
        connect<window<proj_sz>>(k_post_ap.out[0], k_post_bc.in[0]);
        connect<window<c_sz>>(k_post_bc.out[0], plio_c_out.in[0]);
#endif
    }
};

// cross attn subgraph

template <int LAYER>
class CrossAttnGraphL : public graph {
public:
    input_plio  plio_x_in;
    input_plio  plio_c_in;
    output_plio plio_x_out;
public:
    kernel k_pre[N_HEADS];
    kernel k_post_h[N_HEADS];
#if defined(HEAD_STREAM_CROSS)
    kernel k_merge[2];                // pair the four head streams for the projection
#endif
#ifdef POST_MERGED
    kernel k_post_ap, k_post_bc;      // b1 + b2 + c in one kernel
#else
    kernel k_post_ap, k_post_b1, k_post_b2, k_post_c;
#if defined(POST_SPLIT_C)
    kernel k_post_c1;                 // FFN layer 2 + norm + ReLU; post_c keeps the residual add + norm
#if defined(ROW_SPLIT_CROSS)
    kernel k_hb[4];                   // second row chain (odd rows): b1, b2, c1, c
    kernel k_rowmerge;                // puts the two chains' rows back in order
#endif
#endif
#endif
public:
    CrossAttnGraphL() {
        const std::string suffix = "_L" + std::to_string(LAYER);
        plio_x_in = input_plio::create("cross_x_in" + suffix, plio_64_bits,
                                        "data/cross_x_in" + suffix + ".txt");
        plio_c_in = input_plio::create("cross_c_in" + suffix, plio_64_bits,
                                        "data/cross_c_in" + suffix + ".txt");
        plio_x_out = output_plio::create("cross_x_out" + suffix, plio_64_bits,
                                        "data/cross_x_out" + suffix + ".txt");

        if constexpr (LAYER == 0) {
            k_pre[0] = kernel::create(cross_attn_head_pre_h0_L0);
            k_pre[1] = kernel::create(cross_attn_head_pre_h1_L0);
            k_pre[2] = kernel::create(cross_attn_head_pre_h2_L0);
            k_pre[3] = kernel::create(cross_attn_head_pre_h3_L0);
            k_post_h[0] = kernel::create(cross_attn_head_post_h0_L0);
            k_post_h[1] = kernel::create(cross_attn_head_post_h1_L0);
            k_post_h[2] = kernel::create(cross_attn_head_post_h2_L0);
            k_post_h[3] = kernel::create(cross_attn_head_post_h3_L0);
#if defined(HEAD_STREAM_CROSS)
            k_merge[0] = kernel::create(cross_head_merge0_L0);
            k_merge[1] = kernel::create(cross_head_merge1_L0);
#endif
        } else {
            k_pre[0] = kernel::create(cross_attn_head_pre_h0_L1);
            k_pre[1] = kernel::create(cross_attn_head_pre_h1_L1);
            k_pre[2] = kernel::create(cross_attn_head_pre_h2_L1);
            k_pre[3] = kernel::create(cross_attn_head_pre_h3_L1);
            k_post_h[0] = kernel::create(cross_attn_head_post_h0_L1);
            k_post_h[1] = kernel::create(cross_attn_head_post_h1_L1);
            k_post_h[2] = kernel::create(cross_attn_head_post_h2_L1);
            k_post_h[3] = kernel::create(cross_attn_head_post_h3_L1);
#if defined(HEAD_STREAM_CROSS)
            k_merge[0] = kernel::create(cross_head_merge0_L1);
            k_merge[1] = kernel::create(cross_head_merge1_L1);
#endif
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
#if defined(POST_SPLIT_C)
            k_post_c1 = kernel::create(cross_post_c1_L0);
#endif
#else
            k_post_bc = kernel::create(cross_post_bc_L0);
#endif
        } else {
            k_post_ap = kernel::create(cross_post_a_proj_L1);
#ifndef POST_MERGED
            k_post_b1 = kernel::create(cross_post_b1_L1);
            k_post_b2 = kernel::create(cross_post_b2_L1);
            k_post_c  = kernel::create(cross_post_c_L1);
#if defined(POST_SPLIT_C)
            k_post_c1 = kernel::create(cross_post_c1_L1);
#endif
#else
            k_post_bc = kernel::create(cross_post_bc_L1);
#endif
        }
        source(k_post_ap) = ("kernels/cross_post_ap_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_ap) = 0.9;
#if defined(HEAD_STREAM_CROSS)
        for (int m = 0; m < 2; m++) {
            source(k_merge[m]) = ("kernels/cross_head_merge" + std::to_string(m) + "_L" + std::to_string(LAYER) + ".cc").c_str();
            runtime<ratio>(k_merge[m]) = 0.9;
        }
#endif
#ifndef POST_MERGED
        source(k_post_b1) = ("kernels/cross_post_b1_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_b1) = 0.9;
        source(k_post_b2) = ("kernels/cross_post_b2_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_b2) = 0.9;
        source(k_post_c) = ("kernels/cross_post_c_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_c) = 0.9;
#if defined(POST_SPLIT_C)
        source(k_post_c1) = ("kernels/cross_post_c1_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_c1) = 0.9;
#endif
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
#if defined(PRE_STREAM) && defined(PRE_STREAM_CROSS)
            connect<stream>(plio_x_in.out[0], k_pre[h].in[0]);
            connect<stream>(plio_c_in.out[0], k_pre[h].in[1]);
#else
            connect<window<x_sz>>(plio_x_in.out[0], k_pre[h].in[0]);
            connect<window<c_sz>>(plio_c_in.out[0], k_pre[h].in[1]);
#endif
#if defined(SCORE_STREAM)
            connect<stream>(k_pre[h].out[0], k_post_h[h].in[0]);        // V, then scores four rows at a time
#else
            connect<window<scores_sz>>(k_pre[h].out[0], k_post_h[h].in[0]);
            connect<window<v_sz>>     (k_pre[h].out[1], k_post_h[h].in[1]);
#endif
#if !defined(HEAD_STREAM_CROSS)
            connect<window<hout>>(k_post_h[h].out[0], k_post_ap.in[h]);
#endif
        }
#if defined(HEAD_STREAM_CROSS)
        connect<stream>(k_post_h[0].out[0], k_merge[0].in[0]);
        connect<stream>(k_post_h[1].out[0], k_merge[0].in[1]);
        connect<stream>(k_post_h[2].out[0], k_merge[1].in[0]);
        connect<stream>(k_post_h[3].out[0], k_merge[1].in[1]);
        connect<stream>(k_merge[0].out[0], k_post_ap.in[0]);
        connect<stream>(k_merge[1].out[0], k_post_ap.in[1]);
        connect<window<x_sz>>(plio_x_in.out[0], k_post_ap.in[2]);
#else
        connect<window<x_sz>>(plio_x_in.out[0], k_post_ap.in[AP_RESID_IN_CROSS]);
#endif

#if defined(POST_STREAM)
        // rows stream a_proj -> b1 -> b2 -> c; the block output is a stream
        connect<stream>(k_post_ap.out[0], k_post_b1.in[0]);
        connect<stream>(k_post_ap.out[0], k_post_c.in[1]);
        connect<stream>(k_post_b1.out[0], k_post_b2.in[0]);
#if defined(POST_SPLIT_C)
        connect<stream>(k_post_b2.out[0], k_post_c1.in[0]);
        connect<stream>(k_post_c1.out[0], k_post_c.in[0]);
#else
        connect<stream>(k_post_b2.out[0], k_post_c.in[0]);
#endif
#if defined(ROW_SPLIT_CROSS) && defined(POST_SPLIT_C)
        // ROW_SPLIT: a_proj's second output carries the odd rows through a copy of the chain
        if constexpr (LAYER == 0) {
            k_hb[0] = kernel::create(cross_post_b1_hb_L0); k_hb[1] = kernel::create(cross_post_b2_hb_L0);
            k_hb[2] = kernel::create(cross_post_c1_hb_L0); k_hb[3] = kernel::create(cross_post_c_hb_L0);
            k_rowmerge = kernel::create(cross_post_rowmerge_L0);
        } else {
            k_hb[0] = kernel::create(cross_post_b1_hb_L1); k_hb[1] = kernel::create(cross_post_b2_hb_L1);
            k_hb[2] = kernel::create(cross_post_c1_hb_L1); k_hb[3] = kernel::create(cross_post_c_hb_L1);
            k_rowmerge = kernel::create(cross_post_rowmerge_L1);
        }
        {
            const char* st[4] = {"b1", "b2", "c1", "c"};
            for (int i = 0; i < 4; i++) {
                source(k_hb[i]) = ("kernels/cross_post_" + std::string(st[i]) + "_hb_L" + std::to_string(LAYER) + ".cc").c_str();
                runtime<ratio>(k_hb[i]) = 0.9;
            }
            source(k_rowmerge) = ("kernels/cross_post_rowmerge_L" + std::to_string(LAYER) + ".cc").c_str();
            runtime<ratio>(k_rowmerge) = 0.9;
        }
        connect<stream>(k_post_ap.out[1], k_hb[0].in[0]);
        connect<stream>(k_post_ap.out[1], k_hb[3].in[1]);
        connect<stream>(k_hb[0].out[0], k_hb[1].in[0]);
        connect<stream>(k_hb[1].out[0], k_hb[2].in[0]);
        connect<stream>(k_hb[2].out[0], k_hb[3].in[0]);
        connect<stream>(k_post_c.out[0], k_rowmerge.in[0]);
        connect<stream>(k_hb[3].out[0], k_rowmerge.in[1]);
        connect<stream>(k_rowmerge.out[0], plio_x_out.in[0]);
#else
        connect<stream>(k_post_c.out[0], plio_x_out.in[0]);
#endif
#elif !defined(POST_MERGED)
        connect<window<proj_sz>>(k_post_ap.out[0], k_post_b1.in[0]);
        connect<window<proj_sz>>(k_post_ap.out[0], k_post_c.in[1]);
        connect<window<proj_sz>>(k_post_b1.out[0], k_post_b2.in[0]);
        connect<window<proj_sz>>(k_post_b2.out[0], k_post_c.in[0]);
        connect<window<x_sz>>(k_post_c.out[0], plio_x_out.in[0]);
#else
        connect<window<proj_sz>>(k_post_ap.out[0], k_post_bc.in[0]);
        connect<window<x_sz>>(k_post_bc.out[0], plio_x_out.in[0]);
#endif
    }
};

// The jet embedding MLP on one tile. On the fabric this stage took 59 us/event
// and set the whole pipeline's rate; here it is a single kernel fed by the PL.
class EmbedGraphL : public graph {
public:
    input_plio  plio_jets_in;
    output_plio plio_x_out;
private:
    kernel k_embed;
public:
    EmbedGraphL() {
        plio_jets_in = input_plio::create("embed_jets_in", plio_64_bits, "data/embed_jets_in.txt");
        plio_x_out   = output_plio::create("embed_x_out",  plio_64_bits, "data/embed_x_out.txt");
        k_embed = kernel::create(embed_mlp);
        source(k_embed) = "kernels/embed_kernel.cc";
        runtime<ratio>(k_embed) = 0.9;
        constexpr int in_sz  = EMBED_IN_WORDS * sizeof(int16);   // 128, 32B-aligned
        constexpr int out_sz = EMBED_ROWS * E_DIM * sizeof(int16);
        connect<window<in_sz>>(plio_jets_in.out[0], k_embed.in[0]);
        connect<window<out_sz>>(k_embed.out[0], plio_x_out.in[0]);
    }
};

class PasswdFullGraph : public graph {
public:
    ObjAttnGraphL<0> obj0;
    CandAttnGraphL<0> cand0;
    CrossAttnGraphL<0> cross0;
    ObjAttnGraphL<1> obj1;
    CandAttnGraphL<1> cand1;
    CrossAttnGraphL<1> cross1;
#ifdef EMBED_ON_AIE
    EmbedGraphL embed;
#endif
};

#endif
