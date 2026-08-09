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

// vectorized tiled gemm: A packed 4x4-block-major, B row-major (gemm_utils.h)
#include "gemm_utils.h"

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

// Integer softmax -- no float anywhere (AIE1 emulates all fp32 in software;
// the softfloat_* calls were the measured tile-time wall). Replaces the
// previous Schraudolph-exp-in-float version:
//   exp(x - max) = 2^(-(max - x) * log2 e)
// with the exponent in Q20 integer arithmetic, a 64-entry 2^-frac LUT (Q15)
// plus a right shift for the integer part, and ONE integer divide per row
// for the normalization. LUT step 2^(1/64) -> ~0.5% max relative error,
// better than the ~2-3% of the Schraudolph float trick it replaces.
//
// Reads scores (row-major, at PIPE_SCORE_SCALE); writes attention weights at
// PIPE_SCALE into OUT in 4x4-block-packed layout (pk_idx) so the AV gemm
// consumes them with pure vector loads. Padded rows/cols are written zero.
#include "exp2_lut.h"

#if defined(ATTN_TYPE_CAND)
#define PIPE_SCALE_BITS CAND_FRAC_BITS
#else
#define PIPE_SCALE_BITS DATA_FRAC_BITS
#endif

template <int N_ROWS, int N_COLS, int N_PAD>
static void int_softmax_packed(const int16* __restrict scores, int16* __restrict out)
{
    // exponent coefficient: t = d * LOG2E_Q is the base-2 exponent in Q20
    constexpr int32 LOG2E_Q = (int32)(1.4426950408889634 / (double)PIPE_SCORE_SCALE * 1048576.0 + 0.5);
    constexpr int32 D_MAX   = (int32)((31u << 20) / (unsigned)LOG2E_Q); // beyond: 2^-31, call it 0
    constexpr int   W_SHIFT = 30 - PIPE_SCALE_BITS;

    for (int r = 0; r < N_ROWS; r++) {
        const int16* row = &scores[r * N_PAD];

        int32 imax = row[0];
        for (int c = 1; c < N_COLS; c++)
            if (row[c] > imax) imax = row[c];

        int32 e[N_PAD];
        int32 sum = 0;
        for (int c = 0; c < N_COLS; c++) {
            int32 d = imax - (int32)row[c];            // >= 0
            int32 v;
            if (d >= D_MAX) {
                v = 0;
            } else {
                int32 t = d * LOG2E_Q;                 // Q20, no overflow (d < D_MAX)
                int   k = t >> 20;                     // integer part
                int   f = (t >> 14) & 63;              // top 6 fraction bits
                v = EXP2_NEG_FRAC_LUT[f] >> k;         // Q15
            }
            e[c] = v;
            sum += v;
        }

        int32 inv = ((int32)1 << 30) / sum;            // sum >= 2^15 -> inv <= 2^15
        for (int c = 0; c < N_PAD; c++) {
            int32 w = 0;
            if (c < N_COLS)
                w = (e[c] * inv + ((int32)1 << (W_SHIFT - 1))) >> W_SHIFT; // e*inv <= 2^30
            out[pk_idx<N_PAD>(r, c)] = (int16)w;
        }
    }
    // zero padded rows so the packed AV operand is fully defined
    for (int r = N_ROWS; r < ((N_ROWS + 3) / 4) * 4; r++)
        for (int c = 0; c < N_PAD; c++)
            out[pk_idx<N_PAD>(r, c)] = 0;
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
    // read X directly into packed layout (same scalar read cost; the gemms
    // then run on pure vector loads)
    alignas(16) int16 Xp[N_MAX * E_DIM];
    for (int r = 0; r < N_MAX; r++)
        for (int c = 0; c < E_DIM; c++)
            Xp[pk_idx<E_DIM>(r, c)] = window_readincr(x_in);

    // V first - persists into the output stream
    alignas(16) int16 V[N_KV_PAD * D_HEAD] = {0};
    gemm_pk<N_MAX, E_DIM, D_HEAD>(Xp, Wv, V, PIPE_ACC_SHIFT);
    add_bias<N_MAX, D_HEAD>(V, bv);
    for (int j = 0; j < D_HEAD; j++) V[N_MAX * D_HEAD + j] = bias_v_row[j];

    alignas(16) int16 scores[N_MAX * N_KV_PAD];
    {
        // Q is N_MAX x 4: for K==4 the packed layout equals row-major, so Q
        // feeds the Q*Kt gemm below without repacking
        alignas(16) int16 Q[N_MAX * D_HEAD];
        gemm_pk<N_MAX, E_DIM, D_HEAD>(Xp, Wq, Q, PIPE_ACC_SHIFT);
        add_bias<N_MAX, D_HEAD>(Q, bq);

        alignas(16) int16 K[N_KV_PAD * D_HEAD] = {0};
        gemm_pk<N_MAX, E_DIM, D_HEAD>(Xp, Wk, K, PIPE_ACC_SHIFT);
        add_bias<N_MAX, D_HEAD>(K, bk);
        for (int j = 0; j < D_HEAD; j++) K[N_MAX * D_HEAD + j] = bias_k_row[j];

        alignas(16) int16 Kt[D_HEAD * N_KV_PAD];
        for (int i = 0; i < N_KV_PAD; i++)
            for (int j = 0; j < D_HEAD; j++)
                Kt[j * N_KV_PAD + i] = K[i * D_HEAD + j];

        gemm_pk<N_MAX, D_HEAD, N_KV_PAD>(Q, Kt, scores, PIPE_QKT_SHIFT);
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

    // integer softmax, emitted directly in packed layout for the AV gemm
    alignas(16) int16 attn_p[N_MAX * N_KV_PAD];
    int_softmax_packed<N_MAX, N_KV, N_KV_PAD>(scores, attn_p);

    alignas(16) int16 head_out[N_MAX * D_HEAD];
    gemm_pk<N_MAX, N_KV_PAD, D_HEAD>(attn_p, V, head_out, PIPE_ACC_SHIFT);

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
    alignas(16) int16 Cp[4 * E_DIM] = {0};  // packed; padded row 3 stays zero
    for (int r = 0; r < T_DIM; r++)
        for (int c = 0; c < E_DIM; c++)
            Cp[pk_idx<E_DIM>(r, c)] = window_readincr(c_in);

    alignas(16) int16 V[T_KV * D_HEAD] = {0};
    gemm_pk<4, E_DIM, D_HEAD>(Cp, cand_Wv, V, PIPE_ACC_SHIFT);
    add_bias<T_DIM, D_HEAD>(V, cand_bv);
    for (int j = 0; j < D_HEAD; j++) V[T_DIM * D_HEAD + j] = cand_bias_v_row[j];

    alignas(16) int16 scores[4 * T_KV];
    {
        alignas(16) int16 Q[4 * D_HEAD];
        gemm_pk<4, E_DIM, D_HEAD>(Cp, cand_Wq, Q, PIPE_ACC_SHIFT);
        add_bias<T_DIM, D_HEAD>(Q, cand_bq);

        alignas(16) int16 K[T_KV * D_HEAD] = {0};
        gemm_pk<4, E_DIM, D_HEAD>(Cp, cand_Wk, K, PIPE_ACC_SHIFT);
        add_bias<T_DIM, D_HEAD>(K, cand_bk);
        for (int j = 0; j < D_HEAD; j++) K[T_DIM * D_HEAD + j] = cand_bias_k_row[j];

        alignas(16) int16 Kt[D_HEAD * T_KV];
        for (int i = 0; i < T_KV; i++)
            for (int j = 0; j < D_HEAD; j++)
                Kt[j * T_KV + i] = K[i * D_HEAD + j];

        gemm_pk<4, D_HEAD, T_KV>(Q, Kt, scores, PIPE_QKT_SHIFT);
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

    // integer softmax (K==4: packed == row-major)
    alignas(16) int16 attn_p[4 * T_KV];
    int_softmax_packed<T_DIM, T_KV, T_KV>(scores, attn_p);

    alignas(16) int16 out[4 * D_HEAD];
    gemm_pk<4, T_KV, D_HEAD>(attn_p, V, out, PIPE_ACC_SHIFT);

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
    alignas(16) int16 Xp[N_MAX * E_DIM];
    for (int r = 0; r < N_MAX; r++)
        for (int c = 0; c < E_DIM; c++)
            Xp[pk_idx<E_DIM>(r, c)] = window_readincr(x_in);

    alignas(16) int16 Cp[4 * E_DIM] = {0};  // packed; padded row 3 stays zero
    for (int r = 0; r < T_DIM; r++)
        for (int c = 0; c < E_DIM; c++)
            Cp[pk_idx<E_DIM>(r, c)] = window_readincr(c_in);

    alignas(16) int16 V[T_KV * D_HEAD] = {0};
    gemm_pk<4, E_DIM, D_HEAD>(Cp, cross_Wv, V, PIPE_ACC_SHIFT);
    add_bias<T_DIM, D_HEAD>(V, cross_bv);
    for (int j = 0; j < D_HEAD; j++) V[T_DIM * D_HEAD + j] = cross_bias_v_row[j];

    alignas(16) int16 scores[N_MAX * T_KV];
    {
        alignas(16) int16 Q[N_MAX * D_HEAD];
        gemm_pk<N_MAX, E_DIM, D_HEAD>(Xp, cross_Wq, Q, PIPE_ACC_SHIFT);
        add_bias<N_MAX, D_HEAD>(Q, cross_bq);

        alignas(16) int16 K[T_KV * D_HEAD] = {0};
        gemm_pk<4, E_DIM, D_HEAD>(Cp, cross_Wk, K, PIPE_ACC_SHIFT);
        add_bias<T_DIM, D_HEAD>(K, cross_bk);
        for (int j = 0; j < D_HEAD; j++) K[T_DIM * D_HEAD + j] = cross_bias_k_row[j];

        alignas(16) int16 Kt[D_HEAD * T_KV];
        for (int i = 0; i < T_KV; i++)
            for (int j = 0; j < D_HEAD; j++)
                Kt[j * T_KV + i] = K[i * D_HEAD + j];

        gemm_pk<N_MAX, D_HEAD, T_KV>(Q, Kt, scores, PIPE_QKT_SHIFT);
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

    // integer softmax (K==4: packed == row-major)
    alignas(16) int16 attn_p[N_MAX * T_KV];
    int_softmax_packed<N_MAX, T_KV, T_KV>(scores, attn_p);

    alignas(16) int16 out[N_MAX * D_HEAD];
    gemm_pk<N_MAX, T_KV, D_HEAD>(attn_p, V, out, PIPE_ACC_SHIFT);

    for (int i = 0; i < N_MAX * D_HEAD; i++) window_writeincr(x_out, out[i]);
}
#endif // HEAD_STAGE_POST
#endif
