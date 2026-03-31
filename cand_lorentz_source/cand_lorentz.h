#ifndef CAND_LORENTZ_H
#define CAND_LORENTZ_H

#include "../attn_block_source/attn_block_types.h"
#include <cmath>

// dimensions

static const int P4_DIM = 4; // E, px, py, pz
static const int RAW_DIM = 5; // log_pt, eta, cos_phi, sin_phi, log_e
static const int AE_IN_DIM = E_DIM - T_DIM + 1; // 16 - 3 + 1 = 14

// configurable constants

static const float MASS_SCALE = 100.0f; // from replication_config.json
static const float MASS_EPS = 1e-4f; // for sqrt(m2)

// get_jet_choice - ISR shift and argmax and one-hot

inline void get_jet_choice(
    const data_t attn_x[N_MAX][E_DIM],
    const bool mask[N_MAX],
    int jet_assign[N_MAX], // per-jet, which candidate (0..T_DIM-1)
    float jet_choice[N_MAX][T_DIM] // one-hot assignment matrix
) {
    for (int i = 0; i < N_MAX; i++) {
        #pragma HLS PIPELINE II=1

        // set padded jets to zero
        if (mask[i]) {
            jet_assign[i] =  T_DIM - 1; //padded jets -> ISR
            for (int t =  0; t < T_DIM; t++) {
                jet_choice[i][t] = 0.0f;
            }
            continue;
        }

        // read first T_DIM cols with ISR shift on col 2

        float scores[T_DIM];
        for (int t = 0; t < T_DIM; t++) {
            scores[t] = (float)attn_x[i][t];
        }
        scores[2] -= 1.0f; // ISR shift

        // argmax

        int best = 0;
        float best_val = scores[0];
        for (int t = 1; t < T_DIM; t++) {
            if (scores[t] > best_val) {
                best_val = scores[t];
                best = t;
            }
        }
        jet_assign[i] = best;

        // one-hot encoding

        for (int t = 0; t < T_DIM; t++) {
            jet_choice[i][t] = (t == best) ? 1.0f : 0.0f;
        }
    }
}

inline void x_to_p4_hw(
    const data_t raw_x[N_MAX][RAW_DIM],
    const bool mask[N_MAX],
    float p4[N_MAX][P4_DIM]
) {
    for (int i = 0; i < N_MAX; i++) {
        #pragma HLS PIPELINE II=1

        if (mask[i]) {
            p4[i][0] = 0.0f;
            p4[i][1] = 0.0f;
            p4[i][2] = 0.0f;
            p4[i][3] = 0.0f;
        } else {
            float log_pt = (float)raw_x[i][0];
            float eta = (float)raw_x[i][1];
            float cos_phi = (float)raw_x[i][2];
            float sin_phi = (float)raw_x[i][3];
            float log_e = (float)raw_x[i][4];

            float pt = expf(log_pt);
            if(pt == 1.0f) pt = 0.0f; // for padded jets: log_pt = -inf -> exp = 1 -> force zero
            float e = expf(log_e);
            if(e == 1.0f) e = 0.0f; // same as above

            // sinh_eta = (exp(eta) - exp(-eta))/2
            float exp_pos = expf(eta);
            float exp_neg = expf(-eta);
            float sinh_eta = (exp_pos - exp_neg) * 0.5f;

            p4[i][0] = e;
            p4[i][1] = pt * cos_phi;
            p4[i][2] = pt * sin_phi;
            p4[i][3] = pt * sinh_eta;
        }
    }
}


// build candidates - jet_choice^T @ jp4
// candidates_p4[t][d] = sum_i jet_choice[i][t] * jp4[i][d]
// since jet_choice is one-hot, this is equivalent to summing p4 of all jets assigned to each candidates
inline void build_candidates_p4(
    const float jet_choice[N_MAX][T_DIM],
    const float jp4[N_MAX][P4_DIM],
    float cand_p4[T_DIM][P4_DIM]
) {
    // initialize accumulators
    for (int t = 0; t < T_DIM; t++) {
        #pragma HLS UNROLL
        for (int d = 0; d < P4_DIM; d++) {
            cand_p4[t][d] = 0.0f;
        }
    }

    // accumulate

    for (int i = 0; i < N_MAX; i++) {
        #pragma HLS PIPELINE II=1
        for (int t = 0; t < T_DIM; t++) {
            if (jet_choice[i][t] > 0.5f) {
                for (int d = 0; d < P4_DIM; d++) {
                    cand_p4[t][d] += jp4[i][d];
                }
            }
        }
    }
}


inline void compute_mass(
    const float cand_p4[T_DIM][P4_DIM],
    float cand_mass_scaled[T_DIM]
) {
    for (int t = 0; t < T_DIM; t++) {
        #pragma HLS PIPELINE II=1
        float e = cand_p4[t][0];
        float px = cand_p4[t][1];
        float py = cand_p4[t][2];
        float pz = cand_p4[t][3];

        float e_adj = e * (1.0f + MASS_EPS);
        float m2 = e_adj * e_adj - px * px - py * py - pz * pz + MASS_EPS;

        cand_mass_scaled[t] = sqrtf(m2) / MASS_SCALE;

    }
}


// drop first T cols, append mass
inline void assemble_ae_input(
    const data_t cand_embed[T_DIM][E_DIM],
    const float cand_mass_scaled[T_DIM],
    data_t ae_input[T_DIM][AE_IN_DIM]
) {
    for (int t = 0; t < T_DIM; t++) {
        #pragma HLS PIPELINE II=1
        // copy embedded features, skipping first T_DIM columns
        for (int e = 0; e < E_DIM - T_DIM; e++) {
            ae_input[t][e] = cand_embed[t][e+T_DIM];
        }

        ae_input[t][AE_IN_DIM - 1] = (data_t)cand_mass_scaled[t];
    }
}

// full cand building pipeline

inline void cand_lorentz(
    const data_t raw_x[N_MAX][RAW_DIM], // raw jet features
    const data_t attn_x[N_MAX][E_DIM], // last cross-attn output
    const data_t cand_embed[T_DIM][E_DIM], // last cand_blocks output
    const bool mask[N_MAX], //padding mask

    float jp4[N_MAX][P4_DIM], // jet 4 momenta
    int jet_assign[N_MAX],  // jet -> cand assignment
    float cand_p4[T_DIM][P4_DIM], // candidate 4 momenta
    float cand_mass_scaled[T_DIM], // scaled invariant mass 
    data_t ae_input[T_DIM][AE_IN_DIM] // assembed ae input features
) {
    float jet_choice[N_MAX][T_DIM];
    get_jet_choice(attn_x, mask, jet_assign, jet_choice);
    x_to_p4_hw(raw_x, mask, jp4);
    build_candidates_p4(jet_choice, jp4, cand_p4);
    compute_mass(cand_p4, cand_mass_scaled);
    assemble_ae_input(cand_embed, cand_mass_scaled, ae_input);
}


#endif