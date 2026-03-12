#include "attn_block_types.h"

// forward declaration of helper functions


// computes y = x @ W^T + bias

static void linear_NxE(
    const data_t in[N_MAX][E_DIM],
    const weight_t W[E_DIM][E_DIM],
    const weight_t bias[E_DIM],
    data_t out[N_MAX][E_DIM],
    int n_rows // actual number of rows, (N_MAX for objects, T for candidates)
);

// normalizes each row to have mean 0 and variance 1

static void layernorm(
    data_t x[N_MAX][E_DIM],
    const ln_param_t gamma[E_DIM],
    const ln_param_t beta[E_DIM],
    int n_rows

);


// converts into probabilities with partition fn

static void softmax_row(
    score_t row[N_KV],
    prob_t out[N_KV],
    int len // actual key length, (N_KV for obj/cross, T for cand)
);

// for exponential table lookup
static exp_t exp_fixed(score_t x);

// object self-attention

// ffn_w, ffn_b are FFN linear layers
// ffn_ln_g, ffn_ln_b are layernorms per layer
// post_ffn_ln_g, post_ffn_ln_b are final layer norms after FFN skip

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
    const weight_t bias_k_param[E_DIM], // add_bias_kv learned token
    const weight_t bias_v_param[E_DIM],

    const weight_t Wo[E_DIM][E_DIM], const weight_t bo[E_DIM], // output projections

    // post attention layer norm
    const ln_param_t attn_ln_g[E_DIM], const ln_param_t attn_ln_b[E_DIM],

    // ffn weights: n_ffn_layers 



)