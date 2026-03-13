#include "attn_helpers.h"

// candidate self-attention, Q = K = V = c (3 x 16)
// scores are 3 x 4 per head

void attn_block_cand(
    data_t c[T_DIM][E_DIM],

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
    const ln_param_t ffn_ln_b[N_FFN_LAYERS][E_DIM],
    const ln_param_t ffn_ln_g[N_FFN_LAYERS][E_DIM],

    // post ffn layernorm after skip connection

    const ln_param_t post_ffn_g[E_DIM],
    const ln_param_t post_ffn_b[E_DIM]

) {


    // TODO: add interface pragmas when i do system integration
    #pragma HLS INTERFACE ap_memory port = c
    #pragma HLS INTERFACE ap_memory port = Wq

    // save residual for skip connection
    data_t residual[T_DIM][E_DIM];
    for (int i = 0; i < T_DIM; i++) {
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < E_DIM; j++) {
            residual[i][j] = c[i][j];
        }
    }

    // 1. QKV projections: x @ W^T + b -> (N_MAX, E_DIM)
    // reshape to (N_HEADS, N_MAX, D_HEAD) for per-head attention

    data_t Q_full[T_DIM][E_DIM]; data_t K_full[T_DIM][E_DIM]; data_t V_full[T_DIM][E_DIM];
    linear<T_DIM>(c, Wq, bq, Q_full);
    linear<T_DIM>(c, Wk, bk, K_full);
    linear<T_DIM>(c, Wv, bv, V_full);

    // reshape inro heads + bias_kv

    data_t Q_h[N_HEADS][T_DIM][D_HEAD];
    data_t K_h[N_HEADS][T_KV][D_HEAD];
    data_t V_h[N_HEADS][T_KV][D_HEAD];

    reshape_and_append_bias_kv<T_DIM, T_DIM>(
        Q_full, K_full, V_full, bias_k, bias_v, Q_h, K_h, V_h
    );

    // per head attention with masking

    data_t context[N_HEADS][T_DIM][D_HEAD];
    HEAD_LOOP:
    for (int h = 0; h < N_HEADS; h++) {
        // compute raw scores
        score_t scores[T_DIM][T_KV];
        compute_scores<T_DIM,T_KV>(Q_h[h], K_h[h], scores);
        softmax_and_context<T_DIM, T_KV>(scores, V_h[h], context[h]);
    }

    // concat heads and output projection

    data_t attn_out[T_DIM][E_DIM];
    concat_and_project<T_DIM>(context, Wo, bo, attn_out);

    for (int i = 0; i < T_DIM; i++) {
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < E_DIM; j++) {
            c[i][j] = attn_out[i][j];
        }
    }
    skip_and_norm<T_DIM>(c, residual, attn_ln_g, attn_ln_b);

    // do ffn

    ffn_block<T_DIM>(c, ffn_w, ffn_b, ffn_ln_g, ffn_ln_b, post_ffn_g, post_ffn_b);
}
