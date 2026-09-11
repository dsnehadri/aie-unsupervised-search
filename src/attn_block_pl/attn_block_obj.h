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
#ifdef HEADS_BATCHED
    // Write the packed channel directly from the projections in one pass over
    // the 13 key rows; the consumer holds QKV_h fully partitioned. The generic
    // path below builds three intermediate arrays and copies them again, 385
    // cycles for 13 rows of data movement. Same semantics: head h takes
    // columns h*D_HEAD+d, row N_MAX of K and V is the learned bias, row N_MAX
    // of Q is zero.
    #pragma HLS ARRAY_PARTITION variable=Q_full dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=K_full dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=V_full dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=bias_k complete
    #pragma HLS ARRAY_PARTITION variable=bias_v complete
    #pragma HLS ARRAY_PARTITION variable=QKV_h dim=0 complete
    for (int i = 0; i < N_KV; i++) {
        #pragma HLS PIPELINE II=1
        for (int h = 0; h < N_HEADS; h++) {
            #pragma HLS UNROLL
            for (int d = 0; d < D_HEAD; d++) {
                #pragma HLS UNROLL
                const int e = h * D_HEAD + d;
                QKV_h[h][i][d]               = (i < N_MAX) ? Q_full[i][e] : (data_t)0;
                QKV_h[N_HEADS + h][i][d]     = (i < N_MAX) ? K_full[i][e] : (data_t)bias_k[e];
                QKV_h[2 * N_HEADS + h][i][d] = (i < N_MAX) ? V_full[i][e] : (data_t)bias_v[e];
            }
        }
    }
    return;
#endif
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

#ifdef HEADS_BATCHED
// All four heads in ONE set of loops, with the head index as the innermost
// unrolled dimension. HEADS_UNROLL did unroll the outer head loop, but each
// head's body is a sequence of pipelined loops and HLS runs distinct loops one
// after another, so the four copies executed back to back: 1416 cycles, 4 x
// 354, a 2% gain. Here every pipelined iteration computes all four heads, so
// the stage costs one head's chain plus four times the datapath.
// Semantics match obj_df_heads exactly: score = (q.k)*SCALE, plus the w_ij
// bias for j < N_MAX when use_wij, NEG_INF for masked j < N_MAX; column
// N_MAX (the learned bias key/value) is never biased or masked.
static void obj_df_heads_batched(
    const data_t QKV_h[3 * N_HEADS][N_KV][D_HEAD],
    const score_t wij_bias[N_MAX * N_HEADS][N_KV], const bool use_wij,
    const bool padding_mask[N_MAX],
    data_t context_f[N_MAX][E_DIM])
{
    #pragma HLS ARRAY_PARTITION variable=QKV_h dim=0 complete
    #pragma HLS ARRAY_PARTITION variable=context_f dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=wij_bias dim=1 block factor=4
    #pragma HLS ARRAY_PARTITION variable=wij_bias dim=2 complete
    score_t scores[N_HEADS][N_MAX][N_KV];
    #pragma HLS ARRAY_PARTITION variable=scores dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=scores dim=3 complete
    HB_SC_I: for (int i = 0; i < N_MAX; i++) {
        HB_SC_J: for (int j = 0; j < N_KV; j++) {
            #pragma HLS PIPELINE II=1
            HB_SC_H: for (int h = 0; h < N_HEADS; h++) {
                #pragma HLS UNROLL
                acc_t sum = 0;
                HB_SC_D: for (int d = 0; d < D_HEAD; d++) {
                    #pragma HLS UNROLL
#ifdef NARROW_MUL
                    sum += QKV_h[h][i][d] * QKV_h[N_HEADS + h][j][d];
#else
                    sum += (acc_t)QKV_h[h][i][d] * (acc_t)QKV_h[N_HEADS + h][j][d];
#endif
                }
                score_t sc = (score_t)(sum * (acc_t)SCALE);
                if (use_wij && j < N_MAX) sc += wij_bias[h * N_MAX + i][j];
                if (j < N_MAX && padding_mask[j]) sc = NEG_INF;
                scores[h][i][j] = sc;
            }
        }
    }
    prob_t attn_w[N_HEADS][N_MAX][N_KV];
    #pragma HLS ARRAY_PARTITION variable=attn_w dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=attn_w dim=3 complete
    HB_SM_H: for (int h = 0; h < N_HEADS; h++) {
        HB_SM_I: for (int i = 0; i < N_MAX; i++) {
            #pragma HLS PIPELINE II=1
            softmax_row<N_KV>(scores[h][i], attn_w[h][i]);
        }
    }
    HB_AV_I: for (int i = 0; i < N_MAX; i++) {
        HB_AV_D: for (int d = 0; d < D_HEAD; d++) {
            #pragma HLS PIPELINE II=1
            HB_AV_H: for (int h = 0; h < N_HEADS; h++) {
                #pragma HLS UNROLL
                acc_t sum = 0;
                HB_AV_J: for (int j = 0; j < N_KV; j++) {
                    #pragma HLS UNROLL
#ifdef NARROW_MUL
                    sum += attn_w[h][i][j] * QKV_h[2 * N_HEADS + h][j][d];
#else
                    sum += (acc_t)attn_w[h][i][j] * (acc_t)QKV_h[2 * N_HEADS + h][j][d];
#endif
                }
                context_f[i][h * D_HEAD + d] = (data_t)sum;
            }
        }
    }
}
#endif

static void obj_df_heads(
    const data_t QKV_h[3 * N_HEADS][N_KV][D_HEAD],
    const score_t wij_bias[N_MAX * N_HEADS][N_KV], const bool use_wij,
    const bool padding_mask[N_MAX],
    data_t context_f[N_MAX][E_DIM])
{
#ifdef HEADS_UNROLL
    // Run the four heads at once. This is NOT the OBJ_HEADS_PARALLEL attempt
    // below, which made each head its own dataflow process and paid for four
    // private copies of Q/K/V through split and merge stages (block II 5762 ->
    // 7850, slower). Unrolling the loop inside one process needs no copies:
    // the arrays are partitioned by head, so all four read their own slice in
    // the same cycle. The cost is four times the head datapath, which is
    // affordable once the linear-layer multiplies are in fabric.
    // (Literal 4, not N_HEADS: the preprocessor does not expand inside pragmas.)
    #pragma HLS ARRAY_PARTITION variable=QKV_h dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=context_f dim=2 block factor=4
    #pragma HLS ARRAY_PARTITION variable=wij_bias dim=1 block factor=4
#endif
    HEAD_LOOP: for (int h = 0; h < N_HEADS; h++) {
#ifdef HEADS_UNROLL
        #pragma HLS UNROLL
#endif
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
#ifdef HEADS_BATCHED
    #pragma HLS ARRAY_PARTITION variable=QKV_h dim=0 complete
    #pragma HLS ARRAY_PARTITION variable=context_f dim=2 complete
    obj_df_heads_batched(QKV_h, wij_bias, use_wij, padding_mask, context_f);
#else
    obj_df_heads(QKV_h, wij_bias, use_wij, padding_mask, context_f);
#endif
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