#ifndef ATTN_HELPERS_H
#define ATTN_HELPERS_H

#include "attn_block_types.h"

static const int T_DIM = 3; // number of categories
static const int T_KV = T_DIM + 1; // to account for bias_kv token

// computes y = x @ W^T + bias

// template allows for compile-time sizing

template <int N_ROWS>

static void linear(
    const data_t in[N_MAX][E_DIM],
    const weight_t W[E_DIM][E_DIM],
    const weight_t bias[E_DIM],
    data_t out[N_MAX][E_DIM],
) {
    LIN_I:
    for (int i = 0; i < N_ROWS; i++) {
        LIN_J:
        for (int j=0; j < E_DIM; j++) {
            #pragma HLS PIPELINE II=1
            acc_t sum = (acc_t) bias[j]
            LIN_K:
            for (int k=0; j < E_DIM; k++) {
                #pragma HLS UNROLL
                sum = sum += (acc_t)in[j][k] * (acc_t)W[j][k]
            }
            out[i][j] = (data_t)sum;
        }
    }
}


#endif