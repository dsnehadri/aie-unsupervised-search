// aie graph for one complete attention block layer

// PLIO connections: 
//  IN: x_data - embedded jet features [N_MAX x E_DIM]
//      wij_h0..3 - pairwise bias per head [N_MAX x N_KV]

#ifndef ATTN_BLOCK_GRAPH_H
#define ATTN_BLOCK_GRAPH_H

#include <adf.h>
#include "../kernels/attn_aie_types.h"
#include "../kernels/attn_head_kernel.h"
#include "../kernels/attn_post_kernel.h"

using namespace adf;

class ObjAttnGraph : public graph {
public:
    //external plio ports
    input_plio  plio_x_in; //X[N_MAX][E_DIM]
    input_plio  plio_wij_h0;
    input_plio  plio_wij_h1;
    input_plio  plio_wij_h2;
    input_plio  plio_wij_h3;
    output_plio  plio_x_out;

private:
    kernel k_head[N_HEADS];
    kernel k_post;

public:
    ObjAttnGraph() {

        // create plio ports 
        // 64 bit plio at aie clock (1GHz) -> 8GB/s per port
        plio_x_in = input_plio::create("obj_x_in", plio_64_bits, "data/obj_x_in.txt");
        plio_wij_h0 = input_plio::create("obj_wij_h0", plio_64_bits, "data/obj_wij_h0.txt");
        plio_wij_h1 = input_plio::create("obj_wij_h1", plio_64_bits, "data/obj_wij_h1.txt");
        plio_wij_h2 = input_plio::create("obj_wij_h2", plio_64_bits, "data/obj_wij_h2.txt");
        plio_wij_h3 = input_plio::create("obj_wij_h3", plio_64_bits, "data/obj_wij_h3.txt");
        plio_x_out = output_plio::create("obj_x_out", plio_64_bits, "data/obj_x_out.txt");

        // create head kernels (4 tiles)
        for (int h = 0; h < N_HEADS; h++) {
            k_head[h] = kernel::create(obj_attn_head);
            source(k_head[h]) = "kernels/attn_head_kernel.cc";
            runtime<ratio>(k_head[h]) = 0.9;
        }

        // create post-attention kernel (1 tile)

        k_post = kernel::create(attn_post);
        source(k_post) = "kernels/attn_post_kernel.cc";
        runtime<ratio>(k_post) = 0.9;

        // window sizes (in bytes)

        constexpr int x_window_bytes = N_MAX * E_DIM * sizeof(int16); // 384
        constexpr int wij_window_bytes = N_MAX * N_KV * sizeof(int16); //312
        constexpr int head_out_bytes = N_MAX * D_HEAD * sizeof(int16); // 96

        // connect plio -> head kernels
        // each head gets the full x input (broadcast from PL)
        // and its own Wij slice

        connect<window<x_window_bytes>>(plio_x_in.out[0], k_head[0].in[0]);
        connect<window<wij_window_bytes>>(plio_wij_h0.out[0], k_head[0].in[1]);

        connect<window<x_window_bytes>>(plio_x_in.out[0], k_head[1].in[0]);
        connect<window<wij_window_bytes>>(plio_wij_h1.out[0], k_head[1].in[1]);

        connect<window<x_window_bytes>>(plio_x_in.out[0], k_head[2].in[0]);
        connect<window<wij_window_bytes>>(plio_wij_h2.out[0], k_head[2].in[1]);

        connect<window<x_window_bytes>>(plio_x_in.out[0], k_head[3].in[0]);
        connect<window<wij_window_bytes>>(plio_wij_h3.out[0], k_head[3].in[1]);

        // connect head outputs -> post kernel

        connect<window<head_out_bytes>>(k_head[0].out[0], k_post.in[0]);
        connect<window<head_out_bytes>>(k_head[1].out[0], k_post.in[1]);
        connect<window<head_out_bytes>>(k_head[2].out[0], k_post.in[2]);
        connect<window<head_out_bytes>>(k_head[3].out[0], k_post.in[3]);

        // residual input for skip connection (same X data)
        connect<window<x_window_bytes>>(plio_x_in.out[0], k_post.in[4]);

        // connect post kernel -> output plio
        connect<window<x_window_bytes>>(k_post.out[0], plio_x_out.in[0]);
    }
};

class CandAttnGraph : public graph {
public:
    //external plio ports
    input_plio  plio_c_in; //C[T_DIM][E_DIM]
    output_plio  plio_c_out; // output [T_DIM][E_DIM]

private:
    kernel k_head[N_HEADS];
    kernel k_post;

public:
    CandAttnGraph() {

        plio_c_in = input_plio::create("cand_c_in", plio_64_bits, "data/cand_c_in.txt");
        plio_c_out = output_plio::create("cand_c_out", plio_64_bits, "data/cand_c_out.txt");

        // create head kernels (4 tiles)
        for (int h = 0; h < N_HEADS; h++) {
            k_head[h] = kernel::create(cand_attn_head);
            source(k_head[h]) = "kernels/attn_head_kernel.cc";
            runtime<ratio>(k_head[h]) = 0.9;
        }

        // create post-attention kernel (1 tile)

        k_post = kernel::create(attn_post);
        source(k_post) = "kernels/attn_post_kernel.cc";
        runtime<ratio>(k_post) = 0.9;

        // window sizes (in bytes)

        constexpr int c_window_bytes = T_DIM * E_DIM * sizeof(int16); // 96
        constexpr int head_out_bytes = T_DIM * D_HEAD * sizeof(int16); // 24

        // connect plio -> head kernels
        for (int h = 0; h < N_HEADS; h++) {
            connect<window<c_window_bytes>>(plio_c_in.out[0], k_head[h].in[0]);
        }

        // connect head outputs -> post kernel

        connect<window<head_out_bytes>>(k_head[0].out[0], k_post.in[0]);
        connect<window<head_out_bytes>>(k_head[1].out[0], k_post.in[1]);
        connect<window<head_out_bytes>>(k_head[2].out[0], k_post.in[2]);
        connect<window<head_out_bytes>>(k_head[3].out[0], k_post.in[3]);

        // residual input for skip connection (same X data)
        connect<window<c_window_bytes>>(plio_c_in.out[0], k_post.in[4]);

        // connect post kernel -> output plio
        connect<window<c_window_bytes>>(k_post.out[0], plio_c_out.in[0]);
    }
};

class CrossAttnGraph : public graph {
public:
    //external plio ports
    input_plio  plio_x_in; //X[N_MAX][E_DIM]
    input_plio  plio_c_in; //C[T_DIM][E_DIM]
    output_plio  plio_x_out; // output [N_MAX][E_DIM]

private:
    kernel k_head[N_HEADS];
    kernel k_post;

public:
    CrossAttnGraph() {
        plio_x_in = input_plio::create("cross_x_in", plio_64_bits, "data/cross_x_in.txt");
        plio_c_in = input_plio::create("cross_c_in", plio_64_bits, "data/cross_c_in.txt");
        plio_x_out = output_plio::create("cross_out", plio_64_bits, "data/cross_x_out.txt");

        // create head kernels (4 tiles)
        for (int h = 0; h < N_HEADS; h++) {
            k_head[h] = kernel::create(cross_attn_head);
            source(k_head[h]) = "kernels/attn_head_kernel.cc";
            runtime<ratio>(k_head[h]) = 0.9;
        }

        // create post-attention kernel (1 tile)

        k_post = kernel::create(attn_post);

        source(k_post) = "kernels/attn_post_kernel.cc";
        runtime<ratio>(k_post) = 0.9;

        // window sizes (in bytes)
        constexpr int x_window_bytes = N_MAX * E_DIM * sizeof(int16); // 384
        constexpr int c_window_bytes = T_DIM * E_DIM * sizeof(int16); // 96
        constexpr int head_out_bytes = N_MAX * D_HEAD * sizeof(int16); // 96

        // connect plio -> head kernels
        for (int h = 0; h < N_HEADS; h++) {
            connect<window<x_window_bytes>>(plio_x_in.out[0], k_head[h].in[0]);
            connect<window<c_window_bytes>>(plio_c_in.out[0], k_head[h].in[1]);
        }

        // connect head outputs -> post kernel

        connect<window<head_out_bytes>>(k_head[0].out[0], k_post.in[0]);
        connect<window<head_out_bytes>>(k_head[1].out[0], k_post.in[1]);
        connect<window<head_out_bytes>>(k_head[2].out[0], k_post.in[2]);
        connect<window<head_out_bytes>>(k_head[3].out[0], k_post.in[3]);

        connect<window<x_window_bytes>>(plio_x_in.out[0], k_post.in[4]);

        // connect post kernel -> output plio
        connect<window<x_window_bytes>>(k_post.out[0], plio_x_out.in[0]);
    }
};


// top level graph, full abc layer

// in full system, flow is:

// PL: embed -> pairwise_mlp
// AIE: obj_attn
// PL: build candidates
// AIE: cand_attn
// AIE: cross_attn
// PL: masking, next layer, or candidate building

class AttnBenchmarkGraph : public graph {
public:
    ObjAttnGraph obj_attn;
    CandAttnGraph cand_attn;
    CrossAttnGraph cross_attn;

    AttnBenchmarkGraph() {

    }
};

#endif