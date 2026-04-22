#ifndef ATTN_HELPERS_H
#define ATTN_HELPERS_H

#include "attn_block_types.h"

// computes y = x @ W^T + bias

// template allows for compile-time sizing

template <int N_ROWS, int FEAT_DIM = E_DIM>
void relu_2d(data_t x[N_ROWS][FEAT_DIM]) {
    for (int i = 0; i < N_ROWS; i++) {
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < FEAT_DIM; j++) {
            if (x[i][j] < (data_t)0) x[i][j] = (data_t)0;
        } 
    }
}

template <int N_ROWS, int OUT_DIM = E_DIM, int IN_DIM = E_DIM>
void linear(
    const data_t in[N_ROWS][IN_DIM],
    const weight_t W[OUT_DIM][IN_DIM],
    const weight_t bias[OUT_DIM],
    data_t out[N_ROWS][OUT_DIM]
) {

    LIN_I:
    for (int i = 0; i < N_ROWS; i++) {
        LIN_J:
        for (int j=0; j < OUT_DIM; j++) {
            #pragma HLS PIPELINE II=1
            acc_t sum = (acc_t) bias[j];
            LIN_K:
            for (int k=0; k < IN_DIM; k++) {
                #pragma HLS UNROLL
                sum += (acc_t)in[i][k] * (acc_t)W[j][k];
            }
            out[i][j] = (data_t)sum;
        }
    }
}

// normalizes each row to have mean 0 and variance 1

template <int N_ROWS, int FEAT_DIM = E_DIM>
void layernorm(
    data_t x[N_ROWS][FEAT_DIM],
    const ln_param_t gamma[FEAT_DIM],
    const ln_param_t beta[FEAT_DIM]

) {
    LN_ROW:
    for (int i = 0; i < N_ROWS; i++) {
        // mean
        acc_t sum = 0;
        LN_MEAN:
        for (int j = 0; j < FEAT_DIM; j++) {
            #pragma HLS UNROLL
            sum += (acc_t)x[i][j];
        }

        data_t mean = (data_t)(sum / FEAT_DIM);

        // variance
        acc_t var_sum = 0;
        LN_VAR:
        for (int j = 0; j < FEAT_DIM; j++) {
            #pragma HLS UNROLL
            acc_t diff = (acc_t)x[i][j] - (acc_t)mean;
            var_sum += diff * diff;
        }
        data_t inv_std = (data_t)hls::rsqrt((float)((data_t)(var_sum/FEAT_DIM)) + (float)LN_EPS);


        // normalize and apply affine

        LN_NORM:
        for (int j=0; j < FEAT_DIM; j++) {
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

// static exp_t exp_fixed(score_t x) {
//     #pragma HLS INLINE
//     if (x>=(score_t)0) return (exp_t)1.0;
//     if (x<=(score_t)EXP_MIN) return (exp_t)0.0;

//     float x_f = (float)x;
//     float frac = (x_f - EXP_MIN) / (-EXP_MIN);
//     int idx = (int)(frac * EXP_LUT_SIZE);
//     if (idx < 0) idx = 0;
//     if (idx >= EXP_LUT_SIZE) idx = EXP_LUT_SIZE -1;

//     return exp_lut[idx];
// }

static exp_t exp_fixed(score_t x) {
    #pragma HLS INLINE
    return (exp_t)expf((float)x);
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
        for (int i = 0; i < N_KEY; i++) {
            for (int d = 0; d < D_HEAD; d++) {
                #pragma HLS PIPELINE II = 1
                int e = h * D_HEAD + d;
                K_h[h][i][d] = K_full[i][e];
                V_h[h][i][d] = V_full[i][e];
            }
        }
        
        for (int d = 0; d < D_HEAD; d++) {
            #pragma HLS PIPELINE II=1
            int e = h * D_HEAD + d;
            K_h[h][N_KEY][d] = (data_t)bias_k[e];
            V_h[h][N_KEY][d] = (data_t)bias_v[e];
        }
    }
}

// scaled dot product attention for one head

template <int N_Q, int N_KEY_TOT>
void compute_scores(
    const data_t Q[N_Q][D_HEAD],
    const data_t K[N_KEY_TOT][D_HEAD],
    score_t scores[N_Q][N_KEY_TOT]
) {
    QK_I:
    for (int i = 0; i < N_Q; i++) {
        QJ_I:
        for (int j = 0; j < N_KEY_TOT; j++) {
            #pragma HLS PIPELINE II=1
            acc_t sum = 0;
            QK_D:
            for (int d = 0; d < D_HEAD; d++) {
                #pragma HLS UNROLL
                sum += (acc_t)Q[i][d] * (acc_t)K[j][d];
            }
            scores[i][j] = (score_t)(sum * (acc_t)SCALE);
        }
    }
}


// softmax scores and then context = attn_weights @ V
template <int N_Q, int N_KEY_TOT>
void softmax_and_context(
    score_t scores[N_Q][N_KEY_TOT],
    const data_t V[N_KEY_TOT][D_HEAD],
    data_t context[N_Q][D_HEAD]
) {
    prob_t attn_w[N_Q][N_KEY_TOT];
    SM_ROWS:
    for (int i = 0; i < N_Q; i++) {
        softmax_row<N_KEY_TOT>(scores[i], attn_w[i]);
    }

    AV_I:
    for (int i = 0; i < N_Q; i++) {
        AV_D:
        for (int d = 0; d < D_HEAD; d++) {
            #pragma HLS PIPELINE II=1
            acc_t sum = 0;
            AV_J:
            for (int j=0; j < N_KEY_TOT; j++) {
                #pragma HLS UNROLL
                sum += (acc_t)attn_w[i][j] * (acc_t)V[j][d];
            }
            context[i][d] = (data_t)sum;
        }
    }
}

// concatenate heads and output projection

template<int N_Q>
void concat_and_project(
    const data_t context[N_HEADS][N_Q][D_HEAD],
    const weight_t Wo[E_DIM][E_DIM],
    const weight_t bo[E_DIM],
    data_t out[N_Q][E_DIM]
) {
    data_t concat_out[N_Q][E_DIM];
    CONCAT:
    for (int i = 0; i < N_Q; i++) {
        #pragma HLS PIPELINE II=1
        for (int h = 0; h < N_HEADS; h++) {
            for (int d = 0; d < D_HEAD; d++) {
                concat_out[i][h * D_HEAD + d] = context[h][i][d];
            }
        }
    }
    linear<N_Q>(concat_out, Wo, bo, out);
}

template<int N_ROWS>
void skip_and_norm(
    data_t x[N_ROWS][E_DIM],
    const data_t residual[N_ROWS][E_DIM],
    const ln_param_t ln_g[E_DIM],
    const ln_param_t ln_b[E_DIM]
) {
    for (int i = 0; i < N_ROWS; i++) {
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < E_DIM; j++)
            x[i][j] = x[i][j] + residual[i][j];
    }
    layernorm<N_ROWS>(x, ln_g, ln_b);
}

// zero out padded jets after attn blocks
inline void remask(data_t x[N_MAX][E_DIM], const bool mask[N_MAX]) {
    REMASK_J: for (int j = 0; j < N_MAX; j++) {
        if (mask[j]) {
            REMASK_E: for (int e = 0; e < E_DIM; e++) {
                #pragma HLS PIPELINE II=1
                x[j][e] = (data_t)0;
            }
        }
    }
}

// expand wij[12x12] to wij_bias[48x13] for multi-head attention
// replicates the same 12x12 wij for each of N_HEADS heads
// column 12 (bias_kv position) stays zero

inline void expand_wij(
    const data_t wij[N_MAX][N_MAX],
    score_t wij_bias[N_HEADS * N_MAX][N_KV]
) {
    //zero-init (column N_MAX) stays zero for the bias_kv token
    EXPAND_ZERO: for (int i = 0; i < N_HEADS * N_MAX; i++) {
        for (int j = 0; j < N_KV; j++) {
            wij_bias[i][j] = (score_t)0;
        }
    }

    EXPAND_COPY: for (int h = 0; h < N_HEADS; h ++) {
        for (int i = 0; i < N_MAX; i++) {
            for (int j = 0; j < N_MAX; j ++) {
                wij_bias[h * N_MAX + i][j] = (score_t)wij[i][j];
            }
        } 
    }
}


struct AttnWeights {
    weight_t Wq[E_DIM][E_DIM], bq[E_DIM];   
    weight_t Wk[E_DIM][E_DIM], bk[E_DIM];   
    weight_t Wv[E_DIM][E_DIM], bv[E_DIM];
    weight_t bias_k[E_DIM], bias_v[E_DIM];
    weight_t Wo[E_DIM][E_DIM], bo[E_DIM];
    ln_param_t attn_ln_g[E_DIM], attn_ln_b[E_DIM];
    weight_t ffn_w[N_FFN_LAYERS][E_DIM][E_DIM];
    weight_t ffn_b[N_FFN_LAYERS][E_DIM];
    ln_param_t ffn_ln_g[N_FFN_LAYERS][E_DIM];
    ln_param_t ffn_ln_b[N_FFN_LAYERS][E_DIM];
    ln_param_t post_ffn_g[E_DIM], post_ffn_b[E_DIM];
};  

#endif