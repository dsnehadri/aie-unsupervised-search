#ifndef EMBED_FFN_H
#define EMBED_FFN_H

#include "../dnn_block_source/dnn_block.h"

// embedding specific constant
static const int EMBED_IN = 5; // raw jet features, log pT, eta, cos phi, sin phi, log E
static const int EMBED_HIDDEN = 16; // E_DIM
static const int EMBED_OUT = 16; //E_DIM
static const int EMBED_N_MID = 1;

typedef DNNBlockWeights<EMBED_IN, EMBED_HIDDEN, EMBED_OUT, EMBED_N_MID> EmbedWeights;

inline void embed_ffn(
    const data_t jets[N_MAX][EMBED_IN],
    const bool mask[N_MAX],
    const EmbedWeights &weights,
    data_t embed[N_MAX][EMBED_OUT]
) {
    dnn_block<N_MAX, EMBED_IN, EMBED_HIDDEN, EMBED_OUT, EMBED_N_MID>(
        jets,
        weights.first_w, weights.first_b, weights.first_ln_g, weights.first_ln_b,
        weights.mid_w, weights.mid_b, weights.mid_ln_g, weights.mid_ln_b,
        weights.last_w, weights.last_b,
        embed
    );

    // zero mask padded jets
    for (int j = 0; j < N_MAX; j++) {
        for (int e = 0; e < EMBED_OUT; e++) {
            #pragma HLS PIPELINE II=1
            if (mask[j]) embed[j][e] = (data_t)0;
        }
    }
}

#endif