#ifndef ATTN_BLOCK_OBJ_H
#define ATTN_BLOCK_OBJ_H

#include "attn_helpers.h"
#include <cstdio>

// object self-attention, Q = K = V = x (12 x 16)
// scores are 12 x 13 per head


// ---------------------------------------------------------------------------
// Stage bodies for the pipelined (OBJ_DATAFLOW) variant. Each is a separate
// dataflow process, so they work on different events at the same time.
// ---------------------------------------------------------------------------
static void obj_df_qkv(
    const data_t x[N_MAX][E_DIM],
    const weight_t Wq[E_DIM][E_DIM], const weight_t bq[E_DIM],
    const weight_t Wk[E_DIM][E_DIM], const weight_t bk[E_DIM],
    const weight_t Wv[E_DIM][E_DIM], const weight_t bv[E_DIM],
    data_t residual[N_MAX][E_DIM],
    data_t Q_full[N_MAX][E_DIM], data_t K_full[N_MAX][E_DIM], data_t V_full[N_MAX][E_DIM])
{
    for (int i = 0; i < N_MAX; i++) {
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < E_DIM; j++) residual[i][j] = x[i][j];
    }
    linear<N_MAX>(x, Wq, bq, Q_full);
    linear<N_MAX>(x, Wk, bk, K_full);
    linear<N_MAX>(x, Wv, bv, V_full);
}


#ifdef OBJ_HEADS_PARALLEL
static void obj_df_reshape_split(
    const data_t Q_full[N_MAX][E_DIM], const data_t K_full[N_MAX][E_DIM],
    const data_t V_full[N_MAX][E_DIM],
    const weight_t bias_k[E_DIM], const weight_t bias_v[E_DIM],
    data_t A0[3][N_KV][D_HEAD], data_t A1[3][N_KV][D_HEAD],
    data_t A2[3][N_KV][D_HEAD], data_t A3[3][N_KV][D_HEAD])
{
    data_t Q_h[N_HEADS][N_MAX][D_HEAD], K_h[N_HEADS][N_KV][D_HEAD], V_h[N_HEADS][N_KV][D_HEAD];
    reshape_and_append_bias_kv<N_MAX, N_MAX>(Q_full, K_full, V_full, bias_k, bias_v, Q_h, K_h, V_h);
    for (int i = 0; i < N_KV; i++) {
        #pragma HLS PIPELINE II=1
        for (int d = 0; d < D_HEAD; d++) {
            data_t q0 = (i < N_MAX) ? Q_h[0][i][d] : (data_t)0;
            data_t q1 = (i < N_MAX) ? Q_h[1][i][d] : (data_t)0;
            data_t q2 = (i < N_MAX) ? Q_h[2][i][d] : (data_t)0;
            data_t q3 = (i < N_MAX) ? Q_h[3][i][d] : (data_t)0;
            A0[0][i][d] = q0; A0[1][i][d] = K_h[0][i][d]; A0[2][i][d] = V_h[0][i][d];
            A1[0][i][d] = q1; A1[1][i][d] = K_h[1][i][d]; A1[2][i][d] = V_h[1][i][d];
            A2[0][i][d] = q2; A2[1][i][d] = K_h[2][i][d]; A2[2][i][d] = V_h[2][i][d];
            A3[0][i][d] = q3; A3[1][i][d] = K_h[3][i][d]; A3[2][i][d] = V_h[3][i][d];
        }
    }
}

template <int H>
static void obj_df_head(
    const data_t A[3][N_KV][D_HEAD],
    const score_t wij_bias[N_MAX * N_HEADS][N_KV], const bool use_wij,
    const bool padding_mask[N_MAX], data_t ctx[N_MAX][D_HEAD])
{
    data_t Q[N_MAX][D_HEAD], K[N_KV][D_HEAD], V[N_KV][D_HEAD];
    for (int i = 0; i < N_KV; i++) {
        #pragma HLS PIPELINE II=1
        for (int d = 0; d < D_HEAD; d++) {
            if (i < N_MAX) Q[i][d] = A[0][i][d];
            K[i][d] = A[1][i][d]; V[i][d] = A[2][i][d];
        }
    }
    score_t scores[N_MAX][N_KV];
    compute_scores<N_MAX, N_KV>(Q, K, scores);
    if (use_wij) {
        for (int i = 0; i < N_MAX; i++) {
            #pragma HLS PIPELINE II=1
            for (int j = 0; j < N_MAX; j++) scores[i][j] += wij_bias[H * N_MAX + i][j];
        }
    }
    for (int i = 0; i < N_MAX; i++) {
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < N_MAX; j++) if (padding_mask[j]) scores[i][j] = NEG_INF;
    }
    softmax_and_context<N_MAX, N_KV>(scores, V, ctx);
}

static void obj_df_merge(
    const data_t c0[N_MAX][D_HEAD], const data_t c1[N_MAX][D_HEAD],
    const data_t c2[N_MAX][D_HEAD], const data_t c3[N_MAX][D_HEAD],
    data_t context_f[N_MAX][E_DIM])
{
    for (int i = 0; i < N_MAX; i++) {
        #pragma HLS PIPELINE II=1
        for (int d = 0; d < D_HEAD; d++) {
            context_f[i][0 * D_HEAD + d] = c0[i][d];
            context_f[i][1 * D_HEAD + d] = c1[i][d];
            context_f[i][2 * D_HEAD + d] = c2[i][d];
            context_f[i][3 * D_HEAD + d] = c3[i][d];
        }
    }
}
#endif

static void obj_df_reshape(
    const data_t Q_full[N_MAX][E_DIM], const data_t K_full[N_MAX][E_DIM],
    const data_t V_full[N_MAX][E_DIM],
    const weight_t bias_k[E_DIM], const weight_t bias_v[E_DIM],
    data_t QKV_h[3 * N_HEADS][N_KV][D_HEAD])
{
    // Q, K and V packed into one channel so the interface stays a single array.
    data_t Q_h[N_HEADS][N_MAX][D_HEAD], K_h[N_HEADS][N_KV][D_HEAD], V_h[N_HEADS][N_KV][D_HEAD];
    reshape_and_append_bias_kv<N_MAX, N_MAX>(Q_full, K_full, V_full, bias_k, bias_v, Q_h, K_h, V_h);
    for (int h = 0; h < N_HEADS; h++)
        for (int i = 0; i < N_KV; i++) {
            #pragma HLS PIPELINE II=1
            for (int d = 0; d < D_HEAD; d++) {
                QKV_h[h][i][d]               = (i < N_MAX) ? Q_h[h][i][d] : (data_t)0;
                QKV_h[N_HEADS + h][i][d]     = K_h[h][i][d];
                QKV_h[2 * N_HEADS + h][i][d] = V_h[h][i][d];
            }
        }
}

static void obj_df_heads(
    const data_t QKV_h[3 * N_HEADS][N_KV][D_HEAD],
    const score_t wij_bias[N_MAX * N_HEADS][N_KV], const bool use_wij,
    const bool padding_mask[N_MAX],
    data_t context_f[N_MAX][E_DIM])
{
    HEAD_LOOP: for (int h = 0; h < N_HEADS; h++) {
        data_t Q[N_MAX][D_HEAD], K[N_KV][D_HEAD], V[N_KV][D_HEAD];
        for (int i = 0; i < N_KV; i++) {
            #pragma HLS PIPELINE II=1
            for (int d = 0; d < D_HEAD; d++) {
                if (i < N_MAX) Q[i][d] = QKV_h[h][i][d];
                K[i][d] = QKV_h[N_HEADS + h][i][d];
                V[i][d] = QKV_h[2 * N_HEADS + h][i][d];
            }
        }
        score_t scores[N_MAX][N_KV];
        compute_scores<N_MAX, N_KV>(Q, K, scores);
        if (use_wij) {
            for (int i = 0; i < N_MAX; i++) {
                #pragma HLS PIPELINE II=1
                for (int j = 0; j < N_MAX; j++) scores[i][j] += wij_bias[h * N_MAX + i][j];
            }
        }
        for (int i = 0; i < N_MAX; i++) {
            #pragma HLS PIPELINE II=1
            for (int j = 0; j < N_MAX; j++) if (padding_mask[j]) scores[i][j] = NEG_INF;
        }
        data_t ctx[N_MAX][D_HEAD];
        softmax_and_context<N_MAX, N_KV>(scores, V, ctx);
        for (int i = 0; i < N_MAX; i++) {
            #pragma HLS PIPELINE II=1
            for (int d = 0; d < D_HEAD; d++) context_f[i][h * D_HEAD + d] = ctx[i][d];
        }
    }
}

static void obj_df_project(
    const data_t context_f[N_MAX][E_DIM],
    const weight_t Wo[E_DIM][E_DIM], const weight_t bo[E_DIM],
    data_t attn_out[N_MAX][E_DIM])
{
    linear<N_MAX>(context_f, Wo, bo, attn_out);
}

static void obj_df_skipnorm(
    const data_t attn_out[N_MAX][E_DIM], const data_t residual[N_MAX][E_DIM],
    const ln_param_t g[E_DIM], const ln_param_t b[E_DIM], data_t x1[N_MAX][E_DIM])
{
    for (int i = 0; i < N_MAX; i++) {
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < E_DIM; j++) x1[i][j] = attn_out[i][j];
    }
    skip_and_norm<N_MAX>(x1, residual, g, b);
}

static void obj_df_ffn(
    const data_t x1[N_MAX][E_DIM],
    const weight_t ffn_w[N_FFN_LAYERS][E_DIM][E_DIM], const weight_t ffn_b[N_FFN_LAYERS][E_DIM],
    const ln_param_t ffn_ln_g[N_FFN_LAYERS][E_DIM], const ln_param_t ffn_ln_b[N_FFN_LAYERS][E_DIM],
    const ln_param_t post_g[E_DIM], const ln_param_t post_b[E_DIM],
    const bool padding_mask[N_MAX], data_t out[N_MAX][E_DIM])
{
    data_t t[N_MAX][E_DIM];
    for (int i = 0; i < N_MAX; i++) {
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < E_DIM; j++) t[i][j] = x1[i][j];
    }
    ffn_block<N_MAX>(t, ffn_w, ffn_b, ffn_ln_g, ffn_ln_b, post_g, post_b);
    for (int i = 0; i < N_MAX; i++) {
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < E_DIM; j++) out[i][j] = padding_mask[i] ? (data_t)0 : t[i][j];
    }
}

inline void attn_block_obj(
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
#ifdef OBJ_DATAFLOW
    // separate output: with x written by the last stage and read by the first,
    // HLS sees a write-after-read hazard and serializes successive events, so
    // the region synthesizes but never pipelines (interval stays at the sum).
    , data_t x_out[N_MAX][E_DIM]
#endif
) {


#ifdef OBJ_DATAFLOW
    // ---- pipelined variant -------------------------------------------------
    // The block runs as one stage today, so its whole 7762 cycles set the
    // pipeline's rate. Split into dataflow processes the rate follows the
    // LONGEST sub-stage instead of their sum: ffn_block is 2170 cycles, so the
    // block's interval should fall about 3.6x. Every array below is written by
    // one process and read by one process, which is what DATAFLOW requires --
    // the in-place use of x in the sequential version is exactly what blocks it,
    // so x is read once at the front and written once at the back.
    #pragma HLS DATAFLOW
    data_t residual[N_MAX][E_DIM];
    data_t Q_full[N_MAX][E_DIM], K_full[N_MAX][E_DIM], V_full[N_MAX][E_DIM];
    obj_df_qkv(x, Wq, bq, Wk, bk, Wv, bv, residual, Q_full, K_full, V_full);

#ifdef OBJ_HEADS_PARALLEL
    // One process per head. They are independent, so this turns the block's
    // dominant stage (4169 cycles for the serial loop over four heads) into
    // four concurrent ~1040-cycle stages. It costs four times the head
    // multipliers, which is only affordable with LIN_FABRIC_MUL.
    data_t QKV_0[3][N_KV][D_HEAD], QKV_1[3][N_KV][D_HEAD];
    data_t QKV_2[3][N_KV][D_HEAD], QKV_3[3][N_KV][D_HEAD];
    obj_df_reshape_split(Q_full, K_full, V_full, bias_k, bias_v,
                         QKV_0, QKV_1, QKV_2, QKV_3);
    data_t ctx0[N_MAX][D_HEAD], ctx1[N_MAX][D_HEAD];
    data_t ctx2[N_MAX][D_HEAD], ctx3[N_MAX][D_HEAD];
    obj_df_head<0>(QKV_0, wij_bias, use_wij, padding_mask, ctx0);
    obj_df_head<1>(QKV_1, wij_bias, use_wij, padding_mask, ctx1);
    obj_df_head<2>(QKV_2, wij_bias, use_wij, padding_mask, ctx2);
    obj_df_head<3>(QKV_3, wij_bias, use_wij, padding_mask, ctx3);
    data_t context_f[N_MAX][E_DIM];
    obj_df_merge(ctx0, ctx1, ctx2, ctx3, context_f);
#else
    data_t QKV_h[3 * N_HEADS][N_KV][D_HEAD];
    obj_df_reshape(Q_full, K_full, V_full, bias_k, bias_v, QKV_h);

    data_t context_f[N_MAX][E_DIM];
    obj_df_heads(QKV_h, wij_bias, use_wij, padding_mask, context_f);
#endif

    data_t attn_out[N_MAX][E_DIM];
    obj_df_project(context_f, Wo, bo, attn_out);

    data_t x1[N_MAX][E_DIM];
    obj_df_skipnorm(attn_out, residual, attn_ln_g, attn_ln_b, x1);

    obj_df_ffn(x1, ffn_w, ffn_b, ffn_ln_g, ffn_ln_b, post_ffn_g, post_ffn_b, padding_mask, x_out);
#else
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
    linear<N_MAX>(x, Wk, bk, K_full);
    linear<N_MAX>(x, Wv, bv, V_full);
    
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
    
    // attention skip and layer norm

    for (int i = 0; i < N_MAX; i++) {
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < E_DIM; j++) {
            x[i][j] = attn_out[i][j];
        }
    }
    skip_and_norm<N_MAX>(x, residual, attn_ln_g, attn_ln_b);

    // do ffn

    ffn_block<N_MAX>(x, ffn_w, ffn_b, ffn_ln_g, ffn_ln_b, post_ffn_g, post_ffn_b);

    // remask padded positions

    for (int i = 0; i < N_MAX; i++) {
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < E_DIM; j++) {
            if (padding_mask[i]) x[i][j] = 0;
        }
    }


#endif
}

#endif