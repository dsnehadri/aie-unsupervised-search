#ifndef ATTN_HELPERS_H
#define ATTN_HELPERS_H

#include "attn_block_types.h"

static const int T_DIM = 3; // number of categories
static const int T_KV = T_DIM + 1; // to account for bias_kv token

// computes y = x @ W^T + bias

// template allows for compile-time sizing

template <int N_ROWS>
void linear(
    const data_t in[N_ROWS][E_DIM],
    const weight_t W[E_DIM][E_DIM],
    const weight_t bias[E_DIM],
    data_t out[N_ROWS][E_DIM]
) {
    LIN_I:
    for (int i = 0; i < N_ROWS; i++) {
        LIN_J:
        for (int j=0; j < E_DIM; j++) {
            #pragma HLS PIPELINE II=1
            acc_t sum = (acc_t) bias[j];
            LIN_K:
            for (int k=0; k < E_DIM; k++) {
                #pragma HLS UNROLL
                sum += (acc_t)in[i][k] * (acc_t)W[j][k];
            }
            out[i][j] = (data_t)sum;
        }
    }
}

// normalizes each row to have mean 0 and variance 1

template <int N_ROWS>
void layernorm(
    data_t x[N_ROWS][E_DIM],
    const ln_param_t gamma[E_DIM],
    const ln_param_t beta[E_DIM]

) {
    LN_ROW:
    for (int i = 0; i < N_ROWS; i++) {
        // mean
        acc_t sum = 0;
        LN_MEAN:
        for (int j = 0; j < E_DIM; j++) {
            #pragma HLS UNROLL
            sum += (acc_t)x[i][j];
        }

        data_t mean = (data_t)(sum / E_DIM);

        // variance
        acc_t var_sum = 0;
        LN_VAR:
        for (int j = 0; j < E_DIM; j++) {
            #pragma HLS UNROLL
            acc_t diff = (acc_t)x[i][j] - (acc_t)mean;
            var_sum += diff * diff;
        }
        data_t inv_std = (data_t)hls::rsqrt((float)((data_t)(var_sum)/E_DIM) + LN_EPS);

        // normalize and apply affine

        LN_NORM:
        for (int j=0; j < E_DIM; j++) {
            #pragma HLS PIPELINE II=1
            data_t x_norm = (data_t)(((acc_t)x[i][j] - (acc_t)mean)*(acc_t)inv_std);
            x[i][j] = (data_t)((acc_t)gamma[j] * (acc_t)x_norm + (acc_t)beta[j]);
        }
    }
}

// for exponential table lookup
static exp_t exp_lut[EXP_LUT_SIZE + 1];
static bool exp_lut_initialized = false;

static void init_exp_lut() {
    for (int i = 0; i <= EXP_LUT_SIZE; i++) {
        float x_val = EXP_MIN * (1.0f - (float)i / EXP_LUT_SIZE);
        exp_lut[i] = (exp_t)expf(x_val);
    }
    exp_lut_initialized = true;
}

static exp_t exp_fixed(score_t x) {
    #pragma HLS INLINE
    if (x>=(score_t)0) return (exp_t)1.0;
    if (x<=(score_t)EXP_MIN) return (exp_t)0.0;

    float x_f = (float)x;
    float frac = (x_f - EXP_MIN) / (-EXP_MIN);
    int idx = (int)(frac * EXP_LUT_SIZE);
    if (idx < 0) idx = 0;
    if (idx >= EXP_LUT_SIZE) idx = EXP_LUT_SIZE -1;

    return exp_lut[idx];
}

// converts into probabilities with partition fn over a row of length LEN

template <int LEN> // actual key length, (N_KV for obj/cross, T for cand)
void softmax_row(
    score_t row[LEN],
    prob_t out[LEN]
) {
    if (!exp_lut_initialized) init_exp_lut();

    // find max
    score_t max_val = row[0];
    SM_MAX:
    for (int j = 0; j < LEN; j++) {
        #pragma HLS PIPELINE II=1
        if (row[j] > max_val) max_val = row[j];
    }

    // exp and sum

    exp_t exp_vals[LEN];
    exp_t exp_sum = 0;
    SM_EXP:
    for (int j = 0; j < LEN; j++) {
        #pragma HLS PIPELINE II=1
        exp_vals[j] = exp_fixed(row[j] - max_val);
        exp_sum += exp_vals[j];
    }

    // normalize

    exp_t inv_sum = (exp_t)(1.0f / (float)exp_sum);
    SM_NORM:
    for (int j = 0; j < LEN; j++) {
        #pragma HLS PIPELINE II=1
        out[j] = (prob_t)(exp_vals[j] * inv_sum);
    }
}

// FFN layer (linear, layernorm, relu) then skip + layernorm

template <int N_ROWS>
void ffn_block(
    data_t x[N_ROWS][E_DIM],
    const weight_t ffn_w[N_FFN_LAYERS][E_DIM][E_DIM],
    const weight_t ffn_b[N_FFN_LAYERS][E_DIM],
    const ln_param_t ffn_ln_g[N_FFN_LAYERS][E_DIM],
    const ln_param_t ffn_ln_b[N_FFN_LAYERS][E_DIM],
    const ln_param_t post_ffn_ln_g[E_DIM],
    const ln_param_t post_ffn_ln_b[E_DIM]
) {
    // save residuals

    data_t residual[N_ROWS][E_DIM]; 
    for (int i = 0; i < N_ROWS; i++) {
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < E_DIM; j++) {
            residual[i][j] = x[i][j];
        }
    }

    // layers
    for (int l = 0; l < N_FFN_LAYERS; l++) {
        data_t tmp[N_ROWS][E_DIM];
        linear<N_ROWS>(x, ffn_w[l], ffn_b[l], tmp);
        layernorm<N_ROWS>(tmp, ffn_ln_g[l], ffn_ln_b[l]);

        // relu

        for (int i = 0; i < N_ROWS; i++) {
            #pragma HLS PIPELINE II=1
            for (int j = 0; j < E_DIM; j++) {
                x[i][j] = (tmp[i][j] > (data_t)0? tmp[i][j] : (data_t) 0);
            }
        }
    }

    // skip connections and layernorm

    for (int i = 0; i < N_ROWS; i++) {
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < E_DIM; j++) {
            x[i][j] = x[i][j] + residual[i][j];
        }
    }

    layernorm<N_ROWS>(x, post_ffn_ln_g, post_ffn_ln_b);
}

#endif

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