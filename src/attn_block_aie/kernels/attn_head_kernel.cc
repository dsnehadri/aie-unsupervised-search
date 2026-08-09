// memory budget per tile (split into pre/post)
//
// pre stage:  Q,K,V projection + Q*K^T scaled  -> emits scores + V
// post stage: + wij (obj only) + softmax + AV  -> emits head_out
//
// scores_f softmax buffer is per-row (N_KV_PAD floats) instead of the
// full matrix to keep stack under budget.

#include "attn_head_kernel.h"
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <adf.h>
#include <math.h>
#include <cstdio>

// The aiecompiler dedups generated tile wrappers by function name. Defining
// "obj_attn_head_pre" identically in N TUs (one per head) collapses to a
// single wrapper and all N tiles end up running head 0's code. Make each
// (type, stage, head, layer) combo have a distinct function name via macros.
#define _HEAD_FN_3(t, s, h, l) t##_attn_head_##s##_h##h##_L##l
#define _HEAD_FN_2(t, s, h, l) _HEAD_FN_3(t, s, h, l)
#define HEAD_PRE_FN   _HEAD_FN_2(ATTN_TYPE_TAG, pre,  HEAD_IDX, ATTN_LAYER)
#define HEAD_POST_FN  _HEAD_FN_2(ATTN_TYPE_TAG, post, HEAD_IDX, ATTN_LAYER)

// Pipeline-wide scale: cand uses the wider Q6.9 throughout (data, weights,
// LN params) so the unnormalized cand_build sums fit. obj/cross use Q4.11.
// Scores have an additional, separate scale because Q*K^T values can grow
// large enough to saturate int16 at PIPE_SCALE; for cand we use the wider
// CAND_SCORE_SCALE=32 (analogous to PL's score_t<16,11>).
#if defined(ATTN_TYPE_CAND)
#define PIPE_SCALE        CAND_SCALE
#define PIPE_ACC_SHIFT    CAND_ACC_SHIFT
// Cand Q*K^T sum reaches ~18M; >> CAND_ACC_SHIFT(9) = 36K which wraps int16
// (aie::mmul to_vector saturation isn't reliable on x86sim). Shift more so the
// result fits int16 cleanly; scores are then at the narrower CAND_SCORE_SCALE.
#define PIPE_SCORE_SCALE  CAND_SCORE_SCALE
#define PIPE_SCORE_SHIFT  CAND_SCORE_SHIFT
#define PIPE_QKT_SHIFT    CAND_QKT_SHIFT
#else
#define PIPE_SCALE        DATA_SCALE
#define PIPE_ACC_SHIFT    ACC_SHIFT
#define PIPE_SCORE_SCALE  DATA_SCALE
#define PIPE_SCORE_SHIFT  DATA_FRAC_BITS
#define PIPE_QKT_SHIFT    ACC_SHIFT
#endif


// object self attention weights

#if defined(ATTN_TYPE_OBJ)
    #if ATTN_LAYER == 0
        #if HEAD_IDX == 0
            #include "weights/obj_head0_weights_L0.h"
        #elif HEAD_IDX == 1
            #include "weights/obj_head1_weights_L0.h"
        #elif HEAD_IDX == 2
            #include "weights/obj_head2_weights_L0.h"
        #elif HEAD_IDX == 3
            #include "weights/obj_head3_weights_L0.h"
        #endif
    #elif ATTN_LAYER == 1
        #if HEAD_IDX == 0
            #include "weights/obj_head0_weights_L1.h"
        #elif HEAD_IDX == 1
            #include "weights/obj_head1_weights_L1.h"
        #elif HEAD_IDX == 2
            #include "weights/obj_head2_weights_L1.h"
        #elif HEAD_IDX == 3
            #include "weights/obj_head3_weights_L1.h"
        #endif
    #endif
#endif

#if defined(ATTN_TYPE_CAND)
    #if ATTN_LAYER == 0
        #if HEAD_IDX == 0
            #include "weights/cand_head0_weights_L0.h"
        #elif HEAD_IDX == 1
            #include "weights/cand_head1_weights_L0.h"
        #elif HEAD_IDX == 2
            #include "weights/cand_head2_weights_L0.h"
        #elif HEAD_IDX == 3
            #include "weights/cand_head3_weights_L0.h"
        #endif
    #elif ATTN_LAYER == 1
        #if HEAD_IDX == 0
            #include "weights/cand_head0_weights_L1.h"
        #elif HEAD_IDX == 1
            #include "weights/cand_head1_weights_L1.h"
        #elif HEAD_IDX == 2
            #include "weights/cand_head2_weights_L1.h"
        #elif HEAD_IDX == 3
            #include "weights/cand_head3_weights_L1.h"
        #endif
    #endif
#endif

#if defined(ATTN_TYPE_CROSS)
    #if ATTN_LAYER == 0
        #if HEAD_IDX == 0
            #include "weights/cross_head0_weights_L0.h"
        #elif HEAD_IDX == 1
            #include "weights/cross_head1_weights_L0.h"
        #elif HEAD_IDX == 2
            #include "weights/cross_head2_weights_L0.h"
        #elif HEAD_IDX == 3
            #include "weights/cross_head3_weights_L0.h"
        #endif
    #elif ATTN_LAYER == 1
        #if HEAD_IDX == 0
            #include "weights/cross_head0_weights_L1.h"
        #elif HEAD_IDX == 1
            #include "weights/cross_head1_weights_L1.h"
        #elif HEAD_IDX == 2
            #include "weights/cross_head2_weights_L1.h"
        #elif HEAD_IDX == 3
            #include "weights/cross_head3_weights_L1.h"
        #endif
    #endif
#endif

// tiled gemm C[M][N] = A[M][K] x B[K][N]

template <int M, int K, int N>
static inline void gemm_tile(const int16* __restrict A, const int16* __restrict B, int16* __restrict C, int shift)
{
    for (int m = 0; m < M; m += 4) {
        for (int n = 0; n < N; n+= 4) {
            aie::mmul<4, 4, 4, int16, int16> acc;

            for (int k = 0; k < K; k += 4) {
                aie::vector<int16, 16> va;
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        va[i * 4 + j] = A[(m+i) * K + (k+j)];
                    }
                }
                aie::vector<int16, 16> vb;
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 4; j++)
                        vb[i * 4 + j] = B[(k+i) * N + (n+j)];   // row-major (aie::mmul expects this)
                if (k == 0) acc.mul(va, vb); else acc.mac(va, vb);
            }
            aie::vector<int16, 16> res = acc.to_vector<int16>(shift);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    C[(m+i) * N + (n+j)] =  res[i*4 + j];
        }
    }
}

// add bias to each row

template<int ROWS, int COLS>
static inline void add_bias(int16* __restrict mat, const int16* __restrict bias)
{
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            mat[r * COLS + c] = (int16)(mat[r * COLS + c] + bias[c]);
        }
    }
}

// scale scores by 1/sqrt(d_head). Operates on the score buffer which is at
// PIPE_SCORE_SCALE (separate from PIPE_SCALE so Q*Kt doesn't saturate int16).
static void scale_scores(int16* __restrict scores, int n_rows, int n_cols_pad, float inv_sqrt_d)
{
    int16 scale_fixed = (int16)(inv_sqrt_d * PIPE_SCORE_SCALE);
    for (int r = 0; r < n_rows; r++) {
        for (int c = 0; c < n_cols_pad; c++) {
            int32 product = (int32)scores[r * n_cols_pad + c] * (int32)scale_fixed;
            scores[r * n_cols_pad + c] = (int16)(product >> PIPE_SCORE_SHIFT);
        }
    }
}

// per-row in-place softmax. Reads at PIPE_SCORE_SCALE (potentially wider for
// cand) and writes attention weights at PIPE_SCALE (data scale used for AV).
// Fast scalar exp approximation (Schraudolph). Avoids pulling the full
// softfloat expf() implementation into AIE program memory (overflows the 16KB
// tile limit on hardware; x86 sim doesn't enforce this). Accurate to ~2-3%,
// within the 0.5 quantized tolerance. Range here is (x - row_max) <= 0.
static inline float fast_expf(float x)
{
    if (x < -87.0f) x = -87.0f;
    union { float f; int i; } u;
    u.i = (int)(12102203.0f * x + 1064866805.0f);
    return u.f;
}

static void softmax_row_inplace(int16* __restrict scores, int n_rows, int n_cols, int n_cols_pad)
{
    // PIPE_SCORE_SCALE is a power of two: reciprocal multiply is bit-exact
    // and avoids a softfloat divide per element (AIE1 emulates all fp32).
    const float inv_score_scale = 1.0f / PIPE_SCORE_SCALE;
    float row_f[64]; // upper bound: max(N_KV_PAD, T_KV) <= 16, padded
    for (int r = 0; r < n_rows; r++) {
        float row_max = -1e30f;
        for (int c = 0; c < n_cols; c++) {
            float val = (float)scores[r * n_cols_pad + c] * inv_score_scale;
            row_f[c] = val;
            if (val > row_max) row_max = val;
        }
        float sum = 0.0f;
        for (int c = 0; c < n_cols; c++) {
            float e = fast_expf(row_f[c] - row_max);
            row_f[c] = e;
            sum += e;
        }
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < n_cols_pad; c++) {
            float val = (c < n_cols) ? row_f[c] * inv_sum : 0.0f;
            scores[r * n_cols_pad + c] = (int16)(val * PIPE_SCALE);
        }
    }
}

// =====================================================================
// object self attention - split into pre + post
// =====================================================================

#if defined(ATTN_TYPE_OBJ)

// stage 1: Q/K/V projection + scores = Q*K^T scaled
#if defined(HEAD_STAGE_PRE)
void HEAD_PRE_FN(input_window_int16* __restrict x_in,
                       output_window_int16* __restrict scores_out,
                       output_window_int16* __restrict v_out)
{
    alignas(16) int16 X[N_MAX * E_DIM];
    for (int i = 0; i < N_MAX * E_DIM; i++) X[i] = window_readincr(x_in);

    // V first - persists into the output stream
    alignas(16) int16 V[N_KV_PAD * D_HEAD] = {0};
    gemm_tile<N_MAX, E_DIM, D_HEAD>(X, Wv, V, PIPE_ACC_SHIFT);
    add_bias<N_MAX, D_HEAD>(V, bv);
    for (int j = 0; j < D_HEAD; j++) V[N_MAX * D_HEAD + j] = bias_v_row[j];

    alignas(16) int16 scores[N_MAX * N_KV_PAD];
    {
        alignas(16) int16 Q[N_MAX * D_HEAD];
        gemm_tile<N_MAX, E_DIM, D_HEAD>(X, Wq, Q, PIPE_ACC_SHIFT);
        add_bias<N_MAX, D_HEAD>(Q, bq);

        alignas(16) int16 K[N_KV_PAD * D_HEAD] = {0};
        gemm_tile<N_MAX, E_DIM, D_HEAD>(X, Wk, K, PIPE_ACC_SHIFT);
        add_bias<N_MAX, D_HEAD>(K, bk);
        for (int j = 0; j < D_HEAD; j++) K[N_MAX * D_HEAD + j] = bias_k_row[j];

        alignas(16) int16 Kt[D_HEAD * N_KV_PAD];
        for (int i = 0; i < N_KV_PAD; i++)
            for (int j = 0; j < D_HEAD; j++)
                Kt[j * N_KV_PAD + i] = K[i * D_HEAD + j];

        gemm_tile<N_MAX, D_HEAD, N_KV_PAD>(Q, Kt, scores, PIPE_QKT_SHIFT);
    }

    scale_scores(scores, N_MAX, N_KV_PAD, 0.5f);

    for (int i = 0; i < N_MAX * N_KV_PAD; i++) window_writeincr(scores_out, scores[i]);
    for (int i = 0; i < N_KV_PAD * D_HEAD; i++) window_writeincr(v_out, V[i]);
}
#endif // HEAD_STAGE_PRE

// stage 2: + wij + softmax + attn*V
#if defined(HEAD_STAGE_POST)
void HEAD_POST_FN(input_window_int16* __restrict scores_in,
                        input_window_int16* __restrict v_in,
                        input_window_int16* __restrict wij_in,
                        output_window_int16* __restrict x_out)
{
    alignas(16) int16 scores[N_MAX * N_KV_PAD];
    for (int i = 0; i < N_MAX * N_KV_PAD; i++) scores[i] = window_readincr(scores_in);

    alignas(16) int16 V[N_KV_PAD * D_HEAD];
    for (int i = 0; i < N_KV_PAD * D_HEAD; i++) V[i] = window_readincr(v_in);

    // add wij row-by-row (no full wij array on stack)
    for (int r = 0; r < N_MAX; r++) {
        for (int c = 0; c < N_KV; c++) {
            int16 w = window_readincr(wij_in);
            int32 sum = (int32)scores[r * N_KV_PAD + c] + (int32)w;
            if (sum > 32767) sum = 32767;
            if (sum < -32768) sum = -32768;
            scores[r * N_KV_PAD + c] = (int16)sum;
        }
    }

    // softmax in-place: scores -> attn_w
    softmax_row_inplace(scores, N_MAX, N_KV, N_KV_PAD);

    alignas(16) int16 head_out[N_MAX * D_HEAD];
    gemm_tile<N_MAX, N_KV_PAD, D_HEAD>(scores, V, head_out, PIPE_ACC_SHIFT);

    for (int i = 0; i < N_MAX * D_HEAD; i++) window_writeincr(x_out, head_out[i]);
}
#endif // HEAD_STAGE_POST
#endif

// =====================================================================
// candidate self attention - split into pre + post
// =====================================================================

#if defined(ATTN_TYPE_CAND)

#if defined(HEAD_STAGE_PRE)
void HEAD_PRE_FN(input_window_int16* __restrict c_in,
                        output_window_int16* __restrict scores_out,
                        output_window_int16* __restrict v_out)
{
    alignas(16) int16 C[4 * E_DIM] = {0};
    for (int r = 0; r < T_DIM; r++)
        for (int c = 0; c < E_DIM; c++)
            C[r * E_DIM + c] = window_readincr(c_in);

    alignas(16) int16 V[T_KV * D_HEAD] = {0};
    gemm_tile<4, E_DIM, D_HEAD>(C, cand_Wv, V, PIPE_ACC_SHIFT);
    add_bias<T_DIM, D_HEAD>(V, cand_bv);
    for (int j = 0; j < D_HEAD; j++) V[T_DIM * D_HEAD + j] = cand_bias_v_row[j];

    alignas(16) int16 scores[4 * T_KV];
    {
        alignas(16) int16 Q[4 * D_HEAD];
        gemm_tile<4, E_DIM, D_HEAD>(C, cand_Wq, Q, PIPE_ACC_SHIFT);
        add_bias<T_DIM, D_HEAD>(Q, cand_bq);

        alignas(16) int16 K[T_KV * D_HEAD] = {0};
        gemm_tile<4, E_DIM, D_HEAD>(C, cand_Wk, K, PIPE_ACC_SHIFT);
        add_bias<T_DIM, D_HEAD>(K, cand_bk);
        for (int j = 0; j < D_HEAD; j++) K[T_DIM * D_HEAD + j] = cand_bias_k_row[j];

        alignas(16) int16 Kt[D_HEAD * T_KV];
        for (int i = 0; i < T_KV; i++)
            for (int j = 0; j < D_HEAD; j++)
                Kt[j * T_KV + i] = K[i * D_HEAD + j];

        gemm_tile<4, D_HEAD, T_KV>(Q, Kt, scores, PIPE_QKT_SHIFT);
    }

    scale_scores(scores, T_DIM, T_KV, 0.5f);

    for (int i = 0; i < 4 * T_KV; i++) window_writeincr(scores_out, scores[i]);
    for (int i = 0; i < T_KV * D_HEAD; i++) window_writeincr(v_out, V[i]);
}
#endif // HEAD_STAGE_PRE

#if defined(HEAD_STAGE_POST)
void HEAD_POST_FN(input_window_int16* __restrict scores_in,
                         input_window_int16* __restrict v_in,
                         output_window_int16* __restrict c_out)
{
    alignas(16) int16 scores[4 * T_KV];
    for (int i = 0; i < 4 * T_KV; i++) scores[i] = window_readincr(scores_in);

    alignas(16) int16 V[T_KV * D_HEAD];
    for (int i = 0; i < T_KV * D_HEAD; i++) V[i] = window_readincr(v_in);

    softmax_row_inplace(scores, T_DIM, T_KV, T_KV);

    alignas(16) int16 out[4 * D_HEAD];
    gemm_tile<4, T_KV, D_HEAD>(scores, V, out, PIPE_ACC_SHIFT);

    for (int r = 0; r < T_DIM; r++)
        for (int c = 0; c < D_HEAD; c++)
            window_writeincr(c_out, out[r * D_HEAD + c]);
}
#endif // HEAD_STAGE_POST
#endif

// =====================================================================
// cross attention - split into pre + post
// =====================================================================

#if defined(ATTN_TYPE_CROSS)

#if defined(HEAD_STAGE_PRE)
void HEAD_PRE_FN(input_window_int16* __restrict x_in,
                         input_window_int16* __restrict c_in,
                         output_window_int16* __restrict scores_out,
                         output_window_int16* __restrict v_out)
{
    alignas(16) int16 X[N_MAX * E_DIM];
    for (int i = 0; i < N_MAX * E_DIM; i++) X[i] = window_readincr(x_in);

    alignas(16) int16 C[4 * E_DIM] = {0};
    for (int r = 0; r < T_DIM; r++)
        for (int c = 0; c < E_DIM; c++)
            C[r * E_DIM + c] = window_readincr(c_in);

    alignas(16) int16 V[T_KV * D_HEAD] = {0};
    gemm_tile<4, E_DIM, D_HEAD>(C, cross_Wv, V, PIPE_ACC_SHIFT);
    add_bias<T_DIM, D_HEAD>(V, cross_bv);
    for (int j = 0; j < D_HEAD; j++) V[T_DIM * D_HEAD + j] = cross_bias_v_row[j];

    alignas(16) int16 scores[N_MAX * T_KV];
    {
        alignas(16) int16 Q[N_MAX * D_HEAD];
        gemm_tile<N_MAX, E_DIM, D_HEAD>(X, cross_Wq, Q, PIPE_ACC_SHIFT);
        add_bias<N_MAX, D_HEAD>(Q, cross_bq);

        alignas(16) int16 K[T_KV * D_HEAD] = {0};
        gemm_tile<4, E_DIM, D_HEAD>(C, cross_Wk, K, PIPE_ACC_SHIFT);
        add_bias<T_DIM, D_HEAD>(K, cross_bk);
        for (int j = 0; j < D_HEAD; j++) K[T_DIM * D_HEAD + j] = cross_bias_k_row[j];

        alignas(16) int16 Kt[D_HEAD * T_KV];
        for (int i = 0; i < T_KV; i++)
            for (int j = 0; j < D_HEAD; j++)
                Kt[j * T_KV + i] = K[i * D_HEAD + j];

        gemm_tile<N_MAX, D_HEAD, T_KV>(Q, Kt, scores, PIPE_QKT_SHIFT);
    }

    scale_scores(scores, N_MAX, T_KV, 0.5f);

    for (int i = 0; i < N_MAX * T_KV; i++) window_writeincr(scores_out, scores[i]);
    for (int i = 0; i < T_KV * D_HEAD; i++) window_writeincr(v_out, V[i]);
}
#endif // HEAD_STAGE_PRE

#if defined(HEAD_STAGE_POST)
void HEAD_POST_FN(input_window_int16* __restrict scores_in,
                          input_window_int16* __restrict v_in,
                          output_window_int16* __restrict x_out)
{
    alignas(16) int16 scores[N_MAX * T_KV];
    for (int i = 0; i < N_MAX * T_KV; i++) scores[i] = window_readincr(scores_in);

    alignas(16) int16 V[T_KV * D_HEAD];
    for (int i = 0; i < T_KV * D_HEAD; i++) V[i] = window_readincr(v_in);

    softmax_row_inplace(scores, N_MAX, T_KV, T_KV);

    alignas(16) int16 out[N_MAX * D_HEAD];
    gemm_tile<N_MAX, T_KV, D_HEAD>(scores, V, out, PIPE_ACC_SHIFT);

    for (int i = 0; i < N_MAX * D_HEAD; i++) window_writeincr(x_out, out[i]);
}
#endif // HEAD_STAGE_POST
#endif
