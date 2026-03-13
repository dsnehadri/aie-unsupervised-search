#include "attn_helpers.h"

// object self-attention

void attn_block_obj(
    data_t x[N_MAX][E_DIM], // input embeddings, which are modified in place

    // masks

    const bool padding_mask[N_MAX],
    const score_t wij_mask[N_MAX * N_HEADS][N_KV],
    const bool use_wig,

    // MHA weights

    const weight_t Wq[E_DIM][E_DIM], const weight_t bq[E_DIM],
    const weight_t Wk[E_DIM][E_DIM], const weight_t bk[E_DIM],
    const weight_t Wv[E_DIM][E_DIM], const weight_t bv[E_DIM],
    const weight_t bias_k[E_DIM], // add_bias_kv learned token
    const weight_t bias_v[E_DIM],
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


    // TODO: add interface pragmas i do system integration
    #pragma HLS INTERFACE ap_memory port = x
    #pragma HLS_INTERFACE ap_memory post = Wq

    // 1. QKV projections: x @ W^T + b -> (N_MAX, E_DIM)
    // reshape to (N_HEADS, N_MAX, D_HEAD) for per-head attention

    data_t Q_full[N_MAX][E_DIM];
    data_t K_full[N_MAX][E_DIM];
    data_t V_full[N_MAX][E_DIM];

    // for self-attention 

}

// reshape QKV into per head arrays and append bias_kv
template <int N_Q, int N_KEY>
void reshape_and_append_bias_kv(
    const data_t Q_full[N_Q][E_DIM],
    const data_t K_full[N_KEY][E_DIM],
    const data_t V_full[N_KEY][E_DIM],
    const weight_t bias_k[E_DIM],
    const weight_t bias_v[E_DIM],
    data_t Q_h[N_HEADS][N_Q][D_HEAD],
    data_t K_h[N_HEADS][N_KEY+1][D_HEAD],
    data_t V_h[N_HEADS][N_KEY+1][D_HEAD]
) {
    RESHAPE:
    for (int h = 0; h <N_HEADS; h++) {
        #pragma HLS UNROLL
        for (int i = 0; i < N_Q; i++) {
            for (int d = 0; d < D_HEAD; d++) {
                #pragma HLS PIPELINE II=1
                Q_h[h][i][d] = Q_full[i][h*D_HEAD+d];
            }
        }
        // separate from above because for cross attention N_KEY != N_Q
        for (int i = 0; i < N_Q, i++) {
            for (int d = 0; d < D_HEAD; d++) {
                #pragma HLS PIPELINE II = 1
                int e = h * D_HEAD + d;
                K_h[h][i][d] = K_full[i][e];
                V_h[h][i][d] = V_full[i][e];
            }
        }
        
        for (int d = 0; d < D_HEAD; d++) {
            #pragma HLS_PIPELINE II=1
            int e = h * D_HEAD + d;
            K_h[h][N_KEY][d] = (data_t)bias_k[e];
            V_h[h][N_KEY][d] = (data_t)bias_v[e];
        }
    }
}