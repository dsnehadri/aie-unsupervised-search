#ifndef PAIRWISE_MLP_H
#define PAIRWISE_MLP_H

#include "../dnn_block/dnn_block.h"
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
    // Inline the whole MLP call tree (dnn_block -> linear/layernorm) so that
    // unrolling PAIR_J below actually REPLICATES the datapath into parallel
    // MLP units. Without this, HLS keeps one shared dnn_block instance and
    // serializes the unrolled calls (the UNROLL alone had no effect).
    #pragma HLS INLINE recursive
    // Partition the (small) MLP weights so the 4 unrolled MLP units can read
    // them concurrently. Without this they contend on single memory ports and
    // serialize -> the UNROLL replicated hardware but gave NO speedup.
    #pragma HLS ARRAY_PARTITION variable=weights.first_w   complete
    #pragma HLS ARRAY_PARTITION variable=weights.first_b   complete
    #pragma HLS ARRAY_PARTITION variable=weights.first_ln_g complete
    #pragma HLS ARRAY_PARTITION variable=weights.first_ln_b complete
    #pragma HLS ARRAY_PARTITION variable=weights.mid_w     complete
    #pragma HLS ARRAY_PARTITION variable=weights.mid_b     complete
    #pragma HLS ARRAY_PARTITION variable=weights.mid_ln_g  complete
    #pragma HLS ARRAY_PARTITION variable=weights.mid_ln_b  complete
    #pragma HLS ARRAY_PARTITION variable=weights.last_w    complete
    #pragma HLS ARRAY_PARTITION variable=weights.last_b    complete
    // Output wij partitioned on j (the unrolled dim) so 4 writes/reads overlap.
    #pragma HLS ARRAY_PARTITION variable=wij dim=2 cyclic factor=4
#ifdef PAIRWISE_PL_LOWDSP
    // all-PL is DSP-constrained (attention already ~990 DSP). Pipelining the
    // MLP fully-spatializes it to a ~1030-DSP floor (II-independent) -> 2020
    // total, over budget. Cap the multipliers so HLS SHARES them: fewer DSP,
    // higher II, but pairwise stays far below the ~841us PL-attention wall.
    #pragma HLS ALLOCATION operation instances=mul limit=384
#endif
    // compute pairwise features

    data_t wij_raw[N_MAX][N_MAX][3];
    #pragma HLS ARRAY_PARTITION variable=wij_raw dim=2 cyclic factor=4
    #pragma HLS ARRAY_PARTITION variable=wij_raw dim=3 complete
    compute_pairwise(w, wij_raw);

    // run mlp on each pair

    PAIR_I:
    for (int i = 0; i < N_MAX; i++) {
        PAIR_J:
        for (int j = 0; j < N_MAX; j++) {
            // PIPELINE (not unroll) the pair loop: overlap the 144 independent
            // MLP evals on one datapath. UNROLL alone replicated hardware but
            // ran the copies serially (no overlap) -> no speedup. Pipelining
            // starts a new pair every II cycles -> ~99.7k -> ~O(pairs) cycles,
            // bounded resource. INLINE recursive + partitioned weights above
            // let HLS unroll the MLP internals and hit a low II.
            // II is config-dependent: the hybrid frees PL DSP (attention on AIE)
            // so it can afford II=1 (~1.4k DSP, pairwise ~27us). The all-PL model
            // already spends ~1055 DSP on PL attention, so it uses a higher II
            // (define PAIRWISE_PL_LOWDSP) to fit -- still removes the bottleneck.
#ifdef PAIRWISE_PL_LOWDSP
            #pragma HLS PIPELINE II=16
#else
            #pragma HLS PIPELINE II=1
#endif
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