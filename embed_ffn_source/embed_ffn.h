#ifndef EMBED_FFN_H
#define EMBED_FFN_H

#include "../attn_block_source/attn_block_types.h"
#include "../attn_block_source/attn_helpers.h"

// embedding specific constant
static const int IN_DIM = 5; // raw jet features, log pT, eta, cos phi, sin phi, log E

//linear 5 -> 16 (first layer only; batchnorm fused into these weights)
// applies same weights to each of N_ROWS jets

template <int N_ROWS>
void linear_first(
    const data_t x[N_ROWS][IN_DIM],
    const weight_t W[E_DIM][IN_DIM],
    const weight_t b[E_DIM],
    data_t out[N_ROWS][E_DIM]
) {
    for (int i = 0; i < N_ROWS; i++) {
        for (int o = 0; o < E_DIM; o++) {
            #pragma HLS PIPELINE II=1
            acc_t sum = (acc_t)b[o];
            for (int k = 0; k < IN_DIM; k++) {
                sum += (acc_t)W[o][k] * (acc_t)x[i][k];
            }
            out[i][o] = (data_t)sum;
        }
    }
}

template <int N_ROWS>
void relu_2d(data_t x[N_ROWS][E_DIM]) {
    for (int i = 0; i < N_ROWS; i++) {
        for (int j = 0; j < E_DIM; j++) {
            #pragma HLS PIPELINE II=1
            if (x[i][j] < (data_t)0) x[i][j] = (data_t)0;
        }
    }
}

// top level embedding for one event
// inference path after bn fusion
// linear (5->16) -> layernorm -> relu
// linear (16->16) -> layernorm -> relu
// linear (16->16)
// zero mask padded jets

inline void embed_ffn(
    const data_t jets[N_MAX][IN_DIM],
    const bool mask[N_MAX],

    // layer 0 linear (5->16) with bn fused

    const weight_t lin0_w[E_DIM][IN_DIM],
    const weight_t lin0_b[E_DIM],
    const ln_param_t ln0_g[E_DIM],
    const ln_param_t ln0_b[E_DIM],

    // layer 1 linear (16->16) with bn fused

    const weight_t lin1_w[E_DIM][E_DIM],
    const weight_t lin1_b[E_DIM],
    const ln_param_t ln1_g[E_DIM],
    const ln_param_t ln1_b[E_DIM],

    // layer 2 linear (16->16) no norm

    const weight_t lin2_w[E_DIM][E_DIM],
    const weight_t lin2_b[E_DIM],

    // output 
    data_t embed[N_MAX][E_DIM]
) {
    data_t buf[N_MAX][E_DIM];

    // layer 0 linear + layernorm + relu

    linear_first<N_MAX>(jets, lin0_w, lin0_b, buf);
    layernorm<N_MAX>(buf, ln0_g, ln0_b);
    relu_2d<N_MAX>(buf);

    // layer 1 linear + layernorm + relu

    linear<N_MAX>(buf, lin1_w, lin1_b, embed);
    layernorm<N_MAX>(embed, ln1_g, ln1_b);
    relu_2d<N_MAX>(embed);

    // layer 2 linear

    linear<N_MAX>(embed, lin2_w, lin2_b, buf);

    // copy to output and zero masked jets

    for (int j = 0; j < N_MAX; j++) {
        for (int e = 0; e < E_DIM; e++) {
            #pragma HLS_PIPELINE II=1
            embed[j][e] = mask[j] ? (data_t)0 : buf[j][e];
        }
    }
}

#endif