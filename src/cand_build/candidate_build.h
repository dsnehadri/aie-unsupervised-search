// candidate building: jet choice = one_hot(argmax(x[:,:,T])) with ISR bias, c = jet_choice^T @ x

#include "../attn_block_pl/attn_block_types.h"

template <int N_ROWS>
void build_candidates(
    data_t x[N_ROWS][E_DIM],
    data_t c[T_DIM][E_DIM],
    int jet_assignment[N_ROWS]
) {
    // isr bias: subtract 1 from feature index 2 (isr category)
    for (int i = 0; i < N_ROWS; i++) {
        #pragma HLS PIPELINE II=1
        x[i][2] = x[i][2] - (data_t)1;
    }

    // argmax over first T_DIM features per jet -> one_hot jet choice
    // jet_choice[i][t] = 1 if t==argmax(x[i][0:T_DIM]), else 0
    // instead of storing full matrix, just store argmax index

    for (int i = 0; i < N_ROWS; i++) {
        #pragma HLS PIPELINE II=1
        data_t best_val = x[i][0];
        int best_idx = 0;
        for (int t = 1; t < T_DIM; t++) {
            if (x[i][t] > best_val) {
                best_val = x[i][t];
                best_idx = t;
            }
        }
        jet_assignment[i] = best_idx;
    }

    // matmul: c[t][e] = sum_i (jet_choice[i][t] * x[i][e])
    // since jet choice is one-hot, this is summing x rows by category

    // zero out candidates

    for (int t = 0; t < T_DIM; t++) {
        #pragma HLS PIPELINE II=1
        for (int e = 0; e< E_DIM; e++) {
            c[t][e] = 0;
        }
    }

    // accumulate 

    for (int i = 0; i < N_ROWS; i++) {
        int t = jet_assignment[i];
        for (int e = 0; e < E_DIM; e++) {
            #pragma HLS PIPELINE II=1
            c[t][e] += x[i][e];
        }
    }
}

