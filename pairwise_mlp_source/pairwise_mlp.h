#ifndef PAIRWISE_MLP_H
#define PAIRWISE_MLP_H

#include "../dnn_block_source/dnn_block.h"
#include <string>

static const int MLP_IN = 3; // (delta_eta, cos_dphi, sin_dphi)
static const int MLP_HIDDEN = 16;
static const int MLP_OUT = 1;
static const int MLP_N_MID = 2;

typedef DNNBlockWeights<MLP_IN, MLP_HIDDEN, MLP_OUT, MLP_N_MID> MLPWeights;

// (eta, cos_phi, sin_phi) -> (delta_eta, cos_dphi, sin_dphi)
static void compute_pairwise(
    const data_t w[N_MAX][3],
    data_t wij_raw[N_MAX][N_MAX][3]
) {
    for (int i = 0; i < N_MAX; i++) {
        data_t eta_i = w[i][0];
        data_t cphi_i = w[i][1];
        data_t sphi_i = w[i][2];

        for (int j = 0; j < N_MAX; j++) {
            #pragma HLS PIPELINE II = 1
            data_t eta_j = w[j][0];
            data_t cphi_j = w[j][1];
            data_t sphi_j = w[j][2];

            wij_raw[i][j][0] = (data_t)((acc_t)eta_i - (acc_t)eta_j);
            wij_raw[i][j][1] = (data_t)((acc_t)cphi_i * (acc_t)cphi_j + (acc_t)sphi_i * (acc_t)sphi_j);
            wij_raw[i][j][2] = (data_t)((acc_t)sphi_i * (acc_t)cphi_j - (acc_t)cphi_i * (acc_t)sphi_j);
        }
    }
}

// pairwise + mlp; compute angular features and then run MLP on each pair

inline void pairwise_mlp(
    const data_t w[N_MAX][3],
    const MLPWeights &weights,
    data_t wij[N_MAX][N_MAX]
) {
    // compute pairwise features

    data_t wij_raw[N_MAX][N_MAX][3];
    printf("before compute_pairwise\n"); fflush(stdout);
    compute_pairwise(w, wij_raw);
    printf("after compute_pairwise\n"); fflush(stdout);

    // run mlp on each pair

    PAIR_I:
    for (int i = 0; i < N_MAX; i++) {
        PAIR_J:
        for (int j = 0; j < N_MAX; j++) {

            fflush(stdout);
            data_t pair_in[1][MLP_IN];
            data_t pair_out[1][MLP_OUT];

            pair_in[0][0] = wij_raw[i][j][0];
            pair_in[0][1] = wij_raw[i][j][1];
            pair_in[0][2] = wij_raw[i][j][2];

            dnn_block<1, MLP_IN, MLP_HIDDEN, MLP_OUT, MLP_N_MID>(
                pair_in,
                weights.first_w, weights.first_b, weights.first_ln_g, weights.first_ln_b,
                weights.mid_w, weights.mid_b, weights.mid_ln_g, weights.mid_ln_b,
                weights.last_w, weights.last_b,
                pair_out
            );

            wij[i][j] = pair_out[0][0];
        }
    }
}

#endif