#include "attn_helpers.h"
#include <cstdio>

// object self-attention, Q = K = V = x (12 x 16)
// scores are 12 x 13 per head

void attn_block_obj(
    data_t x[N_MAX][E_DIM], // input embeddings, which are modified in place

    // masks

    const bool padding_mask[N_MAX],
    const score_t wij_bias[N_MAX * N_HEADS][N_KV],
    const bool use_wij,

    // MHA weights

    const weight_t Wq[E_DIM][E_DIM], const weight_t bq[E_DIM],
    const weight_t Wk[E_DIM][E_DIM], const weight_t bk[E_DIM],
    const weight_t Wv[E_DIM][E_DIM], const weight_t bv[E_DIM],
    const weight_t bias_k[E_DIM], const weight_t bias_v[E_DIM],
    const weight_t Wo[E_DIM][E_DIM], const weight_t bo[E_DIM], // output projections

    // post attention layer norm
    const ln_param_t attn_ln_g[E_DIM], const ln_param_t attn_ln_b[E_DIM],

    // ffn weights: n_ffn_layers 
    const weight_t ffn_w[N_FFN_LAYERS][E_DIM][E_DIM],
    const weight_t ffn_b[N_FFN_LAYERS][E_DIM],
    const ln_param_t ffn_ln_g[N_FFN_LAYERS][E_DIM],
    const ln_param_t ffn_ln_b[N_FFN_LAYERS][E_DIM],

    // post ffn layernorm after skip connection

    const ln_param_t post_ffn_g[E_DIM],
    const ln_param_t post_ffn_b[E_DIM]

) {


    // TODO: add interface pragmas when i do system integration
    #pragma HLS INTERFACE ap_memory port = x
    #pragma HLS INTERFACE ap_memory port = Wq

    // save residual for skip connection
    data_t residual[N_MAX][E_DIM];
    for (int i = 0; i < N_MAX; i++) {
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < E_DIM; j++) {
            residual[i][j] = x[i][j];
        }
    }

    // 1. QKV projections: x @ W^T + b -> (N_MAX, E_DIM)
    // reshape to (N_HEADS, N_MAX, D_HEAD) for per-head attention

    data_t Q_full[N_MAX][E_DIM]; data_t K_full[N_MAX][E_DIM]; data_t V_full[N_MAX][E_DIM];
    linear<N_MAX>(x, Wq, bq, Q_full);

    // DEBUG: manually compute Q[0][0] to verify linear()
    acc_t manual_sum = (acc_t)bq[0];
    for (int k = 0; k < E_DIM; k++) {
        manual_sum += (acc_t)x[0][k] * (acc_t)Wq[0][k];
    }
    printf("Manual Q[0][0]: %f\n", (float)manual_sum);
    printf("Linear Q[0][0]: %f\n", (float)Q_full[0][0]);
    printf("x[0][0..7]: %f %f %f %f %f %f %f %f\n",
        (float)x[0][0], (float)x[0][1], (float)x[0][2], (float)x[0][3],
        (float)x[0][4], (float)x[0][5], (float)x[0][6], (float)x[0][7]);
    printf("Wq[0][0..7]: %f %f %f %f %f %f %f %f\n",
        (float)Wq[0][0], (float)Wq[0][1], (float)Wq[0][2], (float)Wq[0][3],
        (float)Wq[0][4], (float)Wq[0][5], (float)Wq[0][6], (float)Wq[0][7]);

    linear<N_MAX>(x, Wk, bk, K_full);
    linear<N_MAX>(x, Wv, bv, V_full);


    // DEBUG: print Q projection output
    printf("Q_full[0][0..3]: %f %f %f %f\n", (float)Q_full[0][0], (float)Q_full[0][1], (float)Q_full[0][2], (float)Q_full[0][3]);

    printf("Full x[0]: ");
    for (int k = 0; k < E_DIM; k++) printf("%.4f ", (float)x[0][k]);
    printf("\n");
    printf("Full Wq[0]: ");
    for (int k = 0; k < E_DIM; k++) printf("%.4f ", (float)Wq[0][k]);
    printf("\n");
    
    // reshape inro heads + bias_kv

    data_t Q_h[N_HEADS][N_MAX][D_HEAD];
    data_t K_h[N_HEADS][N_KV][D_HEAD];
    data_t V_h[N_HEADS][N_KV][D_HEAD];
    reshape_and_append_bias_kv<N_MAX, N_MAX>(
        Q_full, K_full, V_full, bias_k, bias_v, Q_h, K_h, V_h
    );

    // per head attention with masking

    data_t context[N_HEADS][N_MAX][D_HEAD];
    HEAD_LOOP:
    for (int h = 0; h < N_HEADS; h++) {
        // compute raw scores
        score_t scores[N_MAX][N_KV];
        compute_scores<N_MAX,N_KV>(Q_h[h], K_h[h], scores);

         // DEBUG: print head 0 scores before masking
        if (h == 0) {
            printf("Head0 scores[0][0..3] pre-mask: %f %f %f %f\n",
                (float)scores[0][0], (float)scores[0][1],
                (float)scores[0][2], (float)scores[0][3]);
        }

        // add wij bias
        if (use_wij) {
            for (int i = 0; i < N_MAX; i++) {
                #pragma HLS PIPELINE II=1
                for (int j = 0; j<N_MAX; j++) 
                    scores[i][j] += wij_bias[h * N_MAX + i][j];
            }
        } 

        // apply padding mask, since there can be fewer than 12 jets

        for (int i = 0; i < N_MAX; i++) {
            #pragma HLS PIPELINE II=1
            for (int j = 0; j < N_MAX; j++) {
                if (padding_mask[j])
                    scores[i][j] = NEG_INF;
            }
        }

        softmax_and_context<N_MAX, N_KV>(scores, V_h[h], context[h]);
    }

    // concat heads and output projection

    data_t attn_out[N_MAX][E_DIM];
    concat_and_project<N_MAX>(context, Wo, bo, attn_out);

    printf("attn_out[0][0..3]: %f %f %f %f\n",
        (float)attn_out[0][0], (float)attn_out[0][1],
        (float)attn_out[0][2], (float)attn_out[0][3]);

    // Also print residual to check skip connection inputs
    printf("residual[0][0..3]: %f %f %f %f\n",
        (float)residual[0][0], (float)residual[0][1],
        (float)residual[0][2], (float)residual[0][3]);

    // attention skip and layer norm

    for (int i = 0; i < N_MAX; i++) {
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < E_DIM; j++) {
            x[i][j] = attn_out[i][j];
        }
    }
    skip_and_norm<N_MAX>(x, residual, attn_ln_g, attn_ln_b);

    // do ffn

    // DEBUG: print after attention, before FFN
    printf("Post-attn x[0][0..3]: %f %f %f %f\n",
    (float)x[0][0], (float)x[0][1], (float)x[0][2], (float)x[0][3]);

    ffn_block<N_MAX>(x, ffn_w, ffn_b, ffn_ln_g, ffn_ln_b, post_ffn_g, post_ffn_b);


    printf("Post-FFN x[0][0..3]: %f %f %f %f\n",
        (float)x[0][0], (float)x[0][1], (float)x[0][2], (float)x[0][3]);

    // remask padded positions

    for (int i = 0; i < N_MAX; i++) {
        if (padding_mask[i]) {
            #pragma HLS PIPELINE II=1
            for (int j = 0; j < E_DIM; j++) {
                x[i][j] = 0;
            }
        }
    }


}
