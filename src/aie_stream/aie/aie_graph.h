// instantiates 6 attention subgraphs; each head split across pre/post tiles

#ifndef PASSWD_FULL_GRAPH_H
#define PASSWD_FULL_GRAPH_H

#include <adf.h>

using namespace adf;

#include "../../attn_block_aie/kernels/attn_aie_types.h"
#include "../../attn_block_aie/kernels/attn_head_kernel.h"
#include "../../attn_block_aie/kernels/attn_post_kernel.h"

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
    kernel k_post_ap, k_post_b1, k_post_b2, k_post_c;
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
            k_post_b1 = kernel::create(obj_post_b1_L0);
            k_post_b2 = kernel::create(obj_post_b2_L0);
            k_post_c  = kernel::create(obj_post_c_L0);
        } else {
            k_post_ap = kernel::create(obj_post_a_proj_L1);
            k_post_b1 = kernel::create(obj_post_b1_L1);
            k_post_b2 = kernel::create(obj_post_b2_L1);
            k_post_c  = kernel::create(obj_post_c_L1);
        }
        source(k_post_ap) = ("kernels/obj_post_ap_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_ap) = 0.9;
        source(k_post_b1) = ("kernels/obj_post_b1_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_b1) = 0.9;
        source(k_post_b2) = ("kernels/obj_post_b2_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_b2) = 0.9;
        source(k_post_c) = ("kernels/obj_post_c_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_c) = 0.9;

        // window sizes. obj x INPUT carries N_MAX+1 rows: row N_MAX is the
        // padding mask (nonzero = padded), giving both layers true key
        // masking. The output stays N_MAX rows.
        constexpr int x_sz       = (N_MAX + 1) * E_DIM * sizeof(int16);
        constexpr int x_out_sz   = N_MAX * E_DIM * sizeof(int16);
        constexpr int wij_sz     = N_MAX * N_KV * sizeof(int16);
        constexpr int scores_sz  = N_MAX * N_KV_PAD * sizeof(int16);
        constexpr int v_sz       = N_KV_PAD * D_HEAD * sizeof(int16);
        constexpr int hout       = N_MAX * D_HEAD * sizeof(int16);
        constexpr int concat_sz  = N_MAX * E_DIM * sizeof(int16);
        constexpr int proj_sz    = N_MAX * E_DIM * sizeof(int16);

        // plio -> pre (X for all 4 heads)
        for (int h = 0; h < N_HEADS; h++) {
            connect<window<x_sz>>(plio_x_in.out[0], k_pre[h].in[0]);
        }

        // pre -> post_h: scores + V
        for (int h = 0; h < N_HEADS; h++) {
            connect<window<scores_sz>>(k_pre[h].out[0], k_post_h[h].in[0]);
            connect<window<v_sz>>     (k_pre[h].out[1], k_post_h[h].in[1]);
        }

        // wij PLIOs -> post_h (layer 0 only; L1 kernels have no wij port)
        if constexpr (LAYER == 0) {
            connect<window<wij_sz>>(plio_wij_h0.out[0], k_post_h[0].in[2]);
            connect<window<wij_sz>>(plio_wij_h1.out[0], k_post_h[1].in[2]);
            connect<window<wij_sz>>(plio_wij_h2.out[0], k_post_h[2].in[2]);
            connect<window<wij_sz>>(plio_wij_h3.out[0], k_post_h[3].in[2]);
        }

        // head_post -> post_a_proj directly (the concat tile is gone),
        // residual X -> post_a_proj
        for (int h = 0; h < N_HEADS; h++) {
            connect<window<hout>>(k_post_h[h].out[0], k_post_ap.in[h]);
        }
        connect<window<x_sz>>(plio_x_in.out[0], k_post_ap.in[N_HEADS]);

        // post_a_proj -> post_b1 (ffn0) and post_a_proj -> post_c (FFN-residual broadcast)
        connect<window<proj_sz>>(k_post_ap.out[0], k_post_b1.in[0]);
        connect<window<proj_sz>>(k_post_ap.out[0], k_post_c.in[1]);

        // post_b1 -> post_b2 -> post_c -> PLIO
        connect<window<proj_sz>>(k_post_b1.out[0], k_post_b2.in[0]);
        connect<window<proj_sz>>(k_post_b2.out[0], k_post_c.in[0]);
        connect<window<x_out_sz>>(k_post_c.out[0], plio_x_out.in[0]);
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
    kernel k_post_ap, k_post_b1, k_post_b2, k_post_c;
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
            k_post_b1 = kernel::create(cand_post_b1_L0);
            k_post_b2 = kernel::create(cand_post_b2_L0);
            k_post_c  = kernel::create(cand_post_c_L0);
        } else {
            k_post_ap = kernel::create(cand_post_a_proj_L1);
            k_post_b1 = kernel::create(cand_post_b1_L1);
            k_post_b2 = kernel::create(cand_post_b2_L1);
            k_post_c  = kernel::create(cand_post_c_L1);
        }
        source(k_post_ap) = ("kernels/cand_post_ap_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_ap) = 0.9;
        source(k_post_b1) = ("kernels/cand_post_b1_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_b1) = 0.9;
        source(k_post_b2) = ("kernels/cand_post_b2_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_b2) = 0.9;
        source(k_post_c) = ("kernels/cand_post_c_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_c) = 0.9;

        constexpr int c_sz      = T_DIM * E_DIM * sizeof(int16);
        constexpr int scores_sz = 4 * T_KV * sizeof(int16);
        constexpr int v_sz      = T_KV * D_HEAD * sizeof(int16);
        constexpr int hout      = T_DIM * D_HEAD * sizeof(int16);
        constexpr int concat_sz = T_DIM * E_DIM * sizeof(int16);
        constexpr int proj_sz   = T_DIM * E_DIM * sizeof(int16);

        for (int h = 0; h < N_HEADS; h++) {
            connect<window<c_sz>>(plio_c_in.out[0], k_pre[h].in[0]);
            connect<window<scores_sz>>(k_pre[h].out[0], k_post_h[h].in[0]);
            connect<window<v_sz>>     (k_pre[h].out[1], k_post_h[h].in[1]);
            connect<window<hout>>(k_post_h[h].out[0], k_post_ap.in[h]);
        }
        connect<window<c_sz>>(plio_c_in.out[0], k_post_ap.in[N_HEADS]);

        connect<window<proj_sz>>(k_post_ap.out[0], k_post_b1.in[0]);
        connect<window<proj_sz>>(k_post_ap.out[0], k_post_c.in[1]);
        connect<window<proj_sz>>(k_post_b1.out[0], k_post_b2.in[0]);
        connect<window<proj_sz>>(k_post_b2.out[0], k_post_c.in[0]);
        connect<window<c_sz>>(k_post_c.out[0], plio_c_out.in[0]);
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
    kernel k_post_ap, k_post_b1, k_post_b2, k_post_c;
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
            k_post_b1 = kernel::create(cross_post_b1_L0);
            k_post_b2 = kernel::create(cross_post_b2_L0);
            k_post_c  = kernel::create(cross_post_c_L0);
        } else {
            k_post_ap = kernel::create(cross_post_a_proj_L1);
            k_post_b1 = kernel::create(cross_post_b1_L1);
            k_post_b2 = kernel::create(cross_post_b2_L1);
            k_post_c  = kernel::create(cross_post_c_L1);
        }
        source(k_post_ap) = ("kernels/cross_post_ap_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_ap) = 0.9;
        source(k_post_b1) = ("kernels/cross_post_b1_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_b1) = 0.9;
        source(k_post_b2) = ("kernels/cross_post_b2_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_b2) = 0.9;
        source(k_post_c) = ("kernels/cross_post_c_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post_c) = 0.9;

        constexpr int x_sz      = N_MAX * E_DIM * sizeof(int16);
        constexpr int c_sz      = T_DIM * E_DIM * sizeof(int16);
        constexpr int scores_sz = N_MAX * T_KV * sizeof(int16);
        constexpr int v_sz      = T_KV * D_HEAD * sizeof(int16);
        constexpr int hout      = N_MAX * D_HEAD * sizeof(int16);
        constexpr int concat_sz = N_MAX * E_DIM * sizeof(int16);
        constexpr int proj_sz   = N_MAX * E_DIM * sizeof(int16);

        for (int h = 0; h < N_HEADS; h++) {
            connect<window<x_sz>>(plio_x_in.out[0], k_pre[h].in[0]);
            connect<window<c_sz>>(plio_c_in.out[0], k_pre[h].in[1]);
            connect<window<scores_sz>>(k_pre[h].out[0], k_post_h[h].in[0]);
            connect<window<v_sz>>     (k_pre[h].out[1], k_post_h[h].in[1]);
            connect<window<hout>>(k_post_h[h].out[0], k_post_ap.in[h]);
        }
        connect<window<x_sz>>(plio_x_in.out[0], k_post_ap.in[N_HEADS]);

        connect<window<proj_sz>>(k_post_ap.out[0], k_post_b1.in[0]);
        connect<window<proj_sz>>(k_post_ap.out[0], k_post_c.in[1]);
        connect<window<proj_sz>>(k_post_b1.out[0], k_post_b2.in[0]);
        connect<window<proj_sz>>(k_post_b2.out[0], k_post_c.in[0]);
        connect<window<x_sz>>(k_post_c.out[0], plio_x_out.in[0]);
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

};

#endif
