#ifndef DNN_BLOCK_H
#define DNN_BLOCK_H

#include "../attn_block_source/attn_block_types.h"
#include "../attn_block_source/attn_helpers.h"

// general dnn block 
// pattern:
// linear(IN_DIM -> HIDDEN) + LN + ReLU
// linear(HIDDEN -> HIDDEN) + LN + ReLU 
// linear(HIDDEN -> OUT_DIM)

// template params
// N_ROWS = batch dimension
// IN_DIM = input feature dimension
// HIDDEN = hidden layer width
// OUTDIM = output feature dimension
// N_MID = number of middle hidden -> hidden layers

template <int N_ROWS, int IN_DIM, int HIDDEN, int OUT_DIM, int N_MID> 
inline void dnn_block(
    const data_t input[N_ROWS][IN_DIM],

    // layer 0 linear (5->16) with bn fused

    const weight_t first_w[HIDDEN][IN_DIM],
    const weight_t first_b[HIDDEN],
    const ln_param_t first_ln_g[HIDDEN],
    const ln_param_t first_ln_b[HIDDEN],

    // layer 1 linear (16->16) with bn fused

    const weight_t mid_w[N_MID][HIDDEN][HIDDEN],
    const weight_t mid_b[N_MID][HIDDEN],
    const ln_param_t mid_ln_g[N_MID][HIDDEN],
    const ln_param_t mid_ln_b[N_MID][HIDDEN],

    // layer 2 linear (16->16) no norm

    const weight_t last_w[OUT_DIM][HIDDEN],
    const weight_t last_b[OUT_DIM],

    // output 
    data_t output[N_ROWS][OUT_DIM]
) {
    data_t buf_a[N_ROWS][HIDDEN];
    data_t buf_b[N_ROWS][HIDDEN];

    // layer 0 linear + layernorm + relu

    linear<N_ROWS, HIDDEN, IN_DIM>(input, first_w, first_b, buf_a);

    layernorm<N_ROWS>(buf_a, first_ln_g, first_ln_b);
    relu_2d<N_ROWS>(buf_a);

    for (int l = 0; l < N_MID; l++) {

        linear<N_ROWS, HIDDEN, HIDDEN>(buf_a, mid_w[l], mid_b[l], buf_b);
        layernorm<N_ROWS, HIDDEN>(buf_b, mid_ln_g[l], mid_ln_b[l]);
        relu_2d<N_ROWS>(buf_b);

        // swap buffers

        for (int i = 0; i < N_ROWS; i++) {
            for (int j = 0; j < HIDDEN; j++) {
                buf_a[i][j] = buf_b[i][j];
            }
        }
    }
    
    // layer 2 linear

    linear<N_ROWS, OUT_DIM, HIDDEN>(buf_a, last_w, last_b, output);

}

template <int IN_DIM, int HIDDEN, int OUT_DIM, int N_MID>
struct DNNBlockWeights {
    weight_t first_w[HIDDEN][IN_DIM];
    weight_t first_b[HIDDEN];
    ln_param_t first_ln_g[HIDDEN]; 
    ln_param_t first_ln_b[HIDDEN];

    weight_t mid_w[N_MID][HIDDEN][HIDDEN];
    weight_t mid_b[N_MID][HIDDEN];

    ln_param_t mid_ln_g[N_MID][HIDDEN];
    ln_param_t mid_ln_b[N_MID][HIDDEN];

    weight_t last_w[OUT_DIM][HIDDEN];
    weight_t last_b[OUT_DIM];

};

#endif