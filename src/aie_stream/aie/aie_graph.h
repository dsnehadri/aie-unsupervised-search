// instantiates 6 attention subgraphs   

#ifndef PASSWD_FULL_GRAPH_H
#define PASSWD_FULL_GRAPH_H

#include <adf.h>
#include "../../attn_block_aie/kernels/attn_aie_types.h"
#include "../../attn_block_aie/kernels/attn_head_kernel.h"
#include "../../attn_block_aie/kernels/attn_post_kernel.h"

using namespace adf;

// obj attn subgraph

// template parameter to distinguish layer instances
template <int LAYER>    
class ObjAttnGraphL : public graph {
public:
    input_plio  plio_x_in;
    input_plio  plio_wij_h0, plio_wij_h1, plio_wij_h2, plio_wij_h3;
    output_plio plio_x_out;
private:
    kernel k_head[N_HEADS];
    kernel k_post;
public:
    ObjAttnGraphL() {
        // layer unique plio names
        const std::string suffix = "_L" + std::to_string(LAYER);
        plio_x_in = input_plio::create("obj_x_in" + suffix, plio_64_bits,
                                        "data/obj_x_in" + suffix + ".txt");
        plio_wij_h0 = input_plio::create("obj_wij_h0" + suffix, plio_64_bits,
                                        "data/obj_wij_h0" + suffix + ".txt");
        plio_wij_h1 = input_plio::create("obj_wij_h1" + suffix, plio_64_bits,
                                        "data/obj_wij_h1" + suffix + ".txt");
        plio_wij_h2 = input_plio::create("obj_wij_h2" + suffix, plio_64_bits,
                                        "data/obj_wij_h2" + suffix + ".txt");
        plio_wij_h3 = input_plio::create("obj_wij_h3" + suffix, plio_64_bits,
                                        "data/obj_wij_h3" + suffix + ".txt");
        plio_x_out = output_plio::create("obj_x_out" + suffix, plio_64_bits,
                                        "data/obj_x_out" + suffix + ".txt");

        // layer specific kernel source files
        for (int h = 0; h < N_HEADS; h++) {
            k_head[h] = kernel::create(obj_attn_head);
            source(k_head[h]) = ("kernels/obj_head" + std::to_string(h) +
                                "_L" + std::to_string(LAYER) + ".cc").c_str();
            runtime<ratio>(k_head[h]) = 0.9;
        }

        k_post = kernel::create(attn_post);
        source(k_post) = ("kernels/obj_post_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post) = 0.9;

        // window sizes

        constexpr int x_sz = N_MAX * E_DIM * sizeof(int16);
        constexpr int wij_sz = N_MAX * N_KV * sizeof(int16);
        constexpr int hout = N_MAX * D_HEAD * sizeof(int16);

        // plio -> heads

        connect<window<x_sz>> (plio_x_in.out[0], k_head[0].in[0]);
        connect<window<wij_sz>> (plio_wij_h0.out[0], k_head[0].in[1]);
        connect<window<x_sz>> (plio_x_in.out[0], k_head[1].in[0]);
        connect<window<wij_sz>> (plio_wij_h1.out[0], k_head[1].in[1]);
        connect<window<x_sz>> (plio_x_in.out[0], k_head[2].in[0]);
        connect<window<wij_sz>> (plio_wij_h2.out[0], k_head[2].in[1]);
        connect<window<x_sz>> (plio_x_in.out[0], k_head[3].in[0]);
        connect<window<wij_sz>> (plio_wij_h3.out[0], k_head[3].in[1]);

        // heads -> post

        connect<window<hout>> (k_head[0].out[0], k_post.in[0]);
        connect<window<hout>> (k_head[1].out[0], k_post.in[1]);
        connect<window<hout>> (k_head[2].out[0], k_post.in[2]);
        connect<window<hout>> (k_head[3].out[0], k_post.in[3]);
        connect<window<x_sz>> (plio_x_in.out[0], k_post.in[4]);

        // post -> PLIO
        connect<window<x_sz>> (k_post.out[0], plio_x_out.in[0]);
    }
};

// cand attn subgraph

template <int LAYER>    
class CandAttnGraphL : public graph {
public:
    input_plio  plio_c_in;
    output_plio plio_c_out;
private:
    kernel k_head[N_HEADS];
    kernel k_post;
public:
    CandAttnGraphL() {
        // layer unique plio names
        const std::string suffix = "_L" + std::to_string(LAYER);
        plio_c_in = input_plio::create("cand_c_in" + suffix, plio_64_bits,
                                        "data/cand_c_in" + suffix + ".txt");
        plio_c_out = output_plio::create("cand_c_out" + suffix, plio_64_bits,
                                        "data/cand_c_out" + suffix + ".txt");

        // layer specific kernel source files
        for (int h = 0; h < N_HEADS; h++) {
            k_head[h] = kernel::create(cand_attn_head);
            source(k_head[h]) = ("kernels/cand_head" + std::to_string(h) +
                                "_L" + std::to_string(LAYER) + ".cc").c_str();
            runtime<ratio>(k_head[h]) = 0.9;
        }

        k_post = kernel::create(attn_post);
        source(k_post) = ("kernels/cand_post_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post) = 0.9;

        // window sizes

        constexpr int c_sz = T_DIM * E_DIM * sizeof(int16);
        constexpr int hout = T_DIM * D_HEAD * sizeof(int16);

        for (int h = 0; h < N_HEADS; h++) {
            connect<window<c_sz>>(plio_c_in.out[0], k_head[h].in[0]);
        }

        // heads -> post

        connect<window<hout>> (k_head[0].out[0], k_post.in[0]);
        connect<window<hout>> (k_head[1].out[0], k_post.in[1]);
        connect<window<hout>> (k_head[2].out[0], k_post.in[2]);
        connect<window<hout>> (k_head[3].out[0], k_post.in[3]);
        connect<window<c_sz>> (plio_c_in.out[0], k_post.in[4]);

        // post -> PLIO
        connect<window<c_sz>> (k_post.out[0], plio_c_out.in[0]);
    }
};

// cross attn subgrapg

template <int LAYER>    
class CrossAttnGraphL : public graph {
public:
    input_plio  plio_x_in;
    input_plio  plio_c_in;
    output_plio plio_x_out;
private:
    kernel k_head[N_HEADS];
    kernel k_post;
public:
    CrossAttnGraphL() {
        // layer unique plio names
        const std::string suffix = "_L" + std::to_string(LAYER);
        plio_x_in = input_plio::create("cross_x_in" + suffix, plio_64_bits,
                                        "data/cross_x_in" + suffix + ".txt");
        plio_c_in = input_plio::create("cross_c_in" + suffix, plio_64_bits,
                                        "data/cross_c_in" + suffix + ".txt");
        plio_x_out = output_plio::create("cross_x_out" + suffix, plio_64_bits,
                                        "data/cross_x_out" + suffix + ".txt");

        // layer specific kernel source files
        for (int h = 0; h < N_HEADS; h++) {
            k_head[h] = kernel::create(cross_attn_head);
            source(k_head[h]) = ("kernels/cross_head" + std::to_string(h) +
                                "_L" + std::to_string(LAYER) + ".cc").c_str();
            runtime<ratio>(k_head[h]) = 0.9;
        }

        k_post = kernel::create(attn_post);
        source(k_post) = ("kernels/cross_post_L" + std::to_string(LAYER) + ".cc").c_str();
        runtime<ratio>(k_post) = 0.9;

        // window sizes

        constexpr int x_sz = N_MAX * E_DIM * sizeof(int16);
        constexpr int c_sz = T_DIM * E_DIM * sizeof(int16);
        constexpr int hout = N_MAX * D_HEAD * sizeof(int16);

        for (int h = 0; h < N_HEADS; h++) {
            connect<window<x_sz>>(plio_x_in.out[0], k_head[h].in[0]);
            connect<window<c_sz>>(plio_c_in.out[0], k_head[h].in[1]);
        }

        // heads -> post

        connect<window<hout>> (k_head[0].out[0], k_post.in[0]);
        connect<window<hout>> (k_head[1].out[0], k_post.in[1]);
        connect<window<hout>> (k_head[2].out[0], k_post.in[2]);
        connect<window<hout>> (k_head[3].out[0], k_post.in[3]);
        connect<window<x_sz>> (plio_x_in.out[0], k_post.in[4]);

        // post -> PLIO
        connect<window<x_sz>> (k_post.out[0], plio_x_out.in[0]);
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