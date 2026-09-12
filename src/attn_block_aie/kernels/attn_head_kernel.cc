// memory budget per tile (split into pre/post)
//
// pre stage:  Q,K,V projection + Q*K^T scaled  -> emits scores + V
// post stage: + wij (obj only) + softmax + AV  -> emits head_out
//
// scores_f softmax buffer is per-row (N_KV_PAD floats) instead of the
// full matrix to keep stack under budget.

#include "attn_head_kernel.h"

// weight header directory: int16 slices by default, float slices for the
// FLOAT_AIE (unquantized x86sim reference) build
#define AIE_STR(x) #x
#define AIE_XSTR(x) AIE_STR(x)
#ifdef FLOAT_AIE
#define WEIGHTS_DIR weights_f32
#else
#define WEIGHTS_DIR weights
#endif
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

// Retrained-scale layout (attn_aie_types.h): data Q6.9, weights Q3.12 for
// all types. Scores: obj/cross Q8.7 (finer, avoids near-tie softmax flips);
// cand Q10.5 (needs the range -- retrained cand Q*K^T reaches ~320).
#define PIPE_SCALE        DATA_SCALE
#define PIPE_ACC_SHIFT    ACC_SHIFT      // data*weight gemms
#define PIPE_AV_SHIFT     AV_SHIFT       // attn*V gemm (both operands at data scale)
#if defined(ATTN_TYPE_CAND)
#define PIPE_SCORE_SCALE  CAND_SCORE_SCALE
#define PIPE_SCORE_SHIFT  CAND_SCORE_SHIFT
#define PIPE_QKT_SHIFT    CAND_QKT_SHIFT
#else
#define PIPE_SCORE_SCALE  SCORE_SCALE
#define PIPE_SCORE_SHIFT  SCORE_SHIFT
#define PIPE_QKT_SHIFT    QKT_SHIFT
#endif


// object self attention weights

#if defined(ATTN_TYPE_OBJ)
    #if ATTN_LAYER == 0
        #if HEAD_IDX == 0
            #include AIE_XSTR(WEIGHTS_DIR/obj_head0_weights_L0.h)
        #elif HEAD_IDX == 1
            #include AIE_XSTR(WEIGHTS_DIR/obj_head1_weights_L0.h)
        #elif HEAD_IDX == 2
            #include AIE_XSTR(WEIGHTS_DIR/obj_head2_weights_L0.h)
        #elif HEAD_IDX == 3
            #include AIE_XSTR(WEIGHTS_DIR/obj_head3_weights_L0.h)
        #endif
    #elif ATTN_LAYER == 1
        #if HEAD_IDX == 0
            #include AIE_XSTR(WEIGHTS_DIR/obj_head0_weights_L1.h)
        #elif HEAD_IDX == 1
            #include AIE_XSTR(WEIGHTS_DIR/obj_head1_weights_L1.h)
        #elif HEAD_IDX == 2
            #include AIE_XSTR(WEIGHTS_DIR/obj_head2_weights_L1.h)
        #elif HEAD_IDX == 3
            #include AIE_XSTR(WEIGHTS_DIR/obj_head3_weights_L1.h)
        #endif
    #endif
#endif

#if defined(ATTN_TYPE_CAND)
    #if ATTN_LAYER == 0
        #if HEAD_IDX == 0
            #include AIE_XSTR(WEIGHTS_DIR/cand_head0_weights_L0.h)
        #elif HEAD_IDX == 1
            #include AIE_XSTR(WEIGHTS_DIR/cand_head1_weights_L0.h)
        #elif HEAD_IDX == 2
            #include AIE_XSTR(WEIGHTS_DIR/cand_head2_weights_L0.h)
        #elif HEAD_IDX == 3
            #include AIE_XSTR(WEIGHTS_DIR/cand_head3_weights_L0.h)
        #endif
    #elif ATTN_LAYER == 1
        #if HEAD_IDX == 0
            #include AIE_XSTR(WEIGHTS_DIR/cand_head0_weights_L1.h)
        #elif HEAD_IDX == 1
            #include AIE_XSTR(WEIGHTS_DIR/cand_head1_weights_L1.h)
        #elif HEAD_IDX == 2
            #include AIE_XSTR(WEIGHTS_DIR/cand_head2_weights_L1.h)
        #elif HEAD_IDX == 3
            #include AIE_XSTR(WEIGHTS_DIR/cand_head3_weights_L1.h)
        #endif
    #endif
#endif

#if defined(ATTN_TYPE_CROSS)
    #if ATTN_LAYER == 0
        #if HEAD_IDX == 0
            #include AIE_XSTR(WEIGHTS_DIR/cross_head0_weights_L0.h)
        #elif HEAD_IDX == 1
            #include AIE_XSTR(WEIGHTS_DIR/cross_head1_weights_L0.h)
        #elif HEAD_IDX == 2
            #include AIE_XSTR(WEIGHTS_DIR/cross_head2_weights_L0.h)
        #elif HEAD_IDX == 3
            #include AIE_XSTR(WEIGHTS_DIR/cross_head3_weights_L0.h)
        #endif
    #elif ATTN_LAYER == 1
        #if HEAD_IDX == 0
            #include AIE_XSTR(WEIGHTS_DIR/cross_head0_weights_L1.h)
        #elif HEAD_IDX == 1
            #include AIE_XSTR(WEIGHTS_DIR/cross_head1_weights_L1.h)
        #elif HEAD_IDX == 2
            #include AIE_XSTR(WEIGHTS_DIR/cross_head2_weights_L1.h)
        #elif HEAD_IDX == 3
            #include AIE_XSTR(WEIGHTS_DIR/cross_head3_weights_L1.h)
        #endif
    #endif
#endif

#ifdef FLOAT_AIE
// ========================================================================
// FLOAT_AIE: unquantized x86sim reference. IDENTICAL kernel structure --
// same windows, same per-head slicing, bias_kv rows, mask row, wij add,
// emit order -- with plain float32 arithmetic. Proves the kernel logic is
// exact; all int16 deviation is quantization.
// ========================================================================

template <int M, int K, int N>
static void gemm_f(const float* A, const float* B, float* C)
{
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++) s += A[m * K + k] * B[k * N + n];
            C[m * N + n] = s;
        }
}

template <int R, int C>
static void add_bias_f(float* m, const float* b)
{
    for (int r = 0; r < R; r++) for (int c = 0; c < C; c++) m[r * C + c] += b[c];
}

static void softmax_rows_f(float* scores, int n_rows, int n_cols, int n_pad)
{
    for (int r = 0; r < n_rows; r++) {
        float* row = &scores[r * n_pad];
        float mx = row[0];
        for (int c = 1; c < n_cols; c++) if (row[c] > mx) mx = row[c];
        float sum = 0.0f;
        for (int c = 0; c < n_cols; c++) { row[c] = expf(row[c] - mx); sum += row[c]; }
        for (int c = 0; c < n_cols; c++) row[c] /= sum;
        for (int c = n_cols; c < n_pad; c++) row[c] = 0.0f;
    }
}

#if defined(ATTN_TYPE_OBJ)
#if defined(HEAD_STAGE_PRE)
void HEAD_PRE_FN(input_window_float* __restrict x_in,
                 output_window_float* __restrict scores_out,
                 output_window_float* __restrict v_out)
{
    float X[N_MAX * E_DIM];
    for (int i = 0; i < N_MAX * E_DIM; i++) X[i] = window_readincr(x_in);
    float kmask[E_DIM];
    for (int c = 0; c < E_DIM; c++) kmask[c] = window_readincr(x_in);

    float V[N_KV_PAD * D_HEAD] = {0};
    gemm_f<N_MAX, E_DIM, D_HEAD>(X, Wv, V);
    add_bias_f<N_MAX, D_HEAD>(V, bv);
    for (int j = 0; j < D_HEAD; j++) V[N_MAX * D_HEAD + j] = bias_v_row[j];

    float scores[N_MAX * N_KV_PAD] = {0};
    {
        float Q[N_MAX * D_HEAD];
        gemm_f<N_MAX, E_DIM, D_HEAD>(X, Wq, Q);
        add_bias_f<N_MAX, D_HEAD>(Q, bq);
        float K[N_KV_PAD * D_HEAD] = {0};
        gemm_f<N_MAX, E_DIM, D_HEAD>(X, Wk, K);
        add_bias_f<N_MAX, D_HEAD>(K, bk);
        for (int j = 0; j < D_HEAD; j++) K[N_MAX * D_HEAD + j] = bias_k_row[j];
        float Kt[D_HEAD * N_KV_PAD];
        for (int i = 0; i < N_KV_PAD; i++)
            for (int j = 0; j < D_HEAD; j++)
                Kt[j * N_KV_PAD + i] = K[i * D_HEAD + j];
        gemm_f<N_MAX, D_HEAD, N_KV_PAD>(Q, Kt, scores);
    }
    for (int i = 0; i < N_MAX * N_KV_PAD; i++) scores[i] *= 0.5f;
    for (int j = 0; j < N_MAX; j++)
        if (kmask[j] != 0.0f)
            for (int i = 0; i < N_MAX; i++) scores[i * N_KV_PAD + j] = -1e9f;

    for (int i = 0; i < N_MAX * N_KV_PAD; i++) window_writeincr(scores_out, scores[i]);
    for (int i = 0; i < N_KV_PAD * D_HEAD; i++) window_writeincr(v_out, V[i]);
}
#endif
#if defined(HEAD_STAGE_POST)
#if ATTN_LAYER == 0
void HEAD_POST_FN(input_window_float* __restrict scores_in,
                  input_window_float* __restrict v_in,
                  input_window_float* __restrict wij_in,
                  output_window_float* __restrict x_out)
#else
void HEAD_POST_FN(input_window_float* __restrict scores_in,
                  input_window_float* __restrict v_in,
                  output_window_float* __restrict x_out)
#endif
{
    float scores[N_MAX * N_KV_PAD];
    for (int i = 0; i < N_MAX * N_KV_PAD; i++) scores[i] = window_readincr(scores_in);
    float V[N_KV_PAD * D_HEAD];
    for (int i = 0; i < N_KV_PAD * D_HEAD; i++) V[i] = window_readincr(v_in);
#if ATTN_LAYER == 0
    for (int r = 0; r < N_MAX; r++)
        for (int c = 0; c < N_KV; c++)
            scores[r * N_KV_PAD + c] += window_readincr(wij_in);
#endif
    softmax_rows_f(scores, N_MAX, N_KV, N_KV_PAD);
    float out[N_MAX * D_HEAD];
    gemm_f<N_MAX, N_KV_PAD, D_HEAD>(scores, V, out);
    for (int i = 0; i < N_MAX * D_HEAD; i++) window_writeincr(x_out, out[i]);
}
#endif
#endif // ATTN_TYPE_OBJ

#if defined(ATTN_TYPE_CAND)
#if defined(HEAD_STAGE_PRE)
void HEAD_PRE_FN(input_window_float* __restrict c_in,
                 output_window_float* __restrict scores_out,
                 output_window_float* __restrict v_out)
{
    float C[4 * E_DIM] = {0};
    for (int r = 0; r < T_DIM; r++)
        for (int c = 0; c < E_DIM; c++) C[r * E_DIM + c] = window_readincr(c_in);

    float V[T_KV * D_HEAD] = {0};
    gemm_f<4, E_DIM, D_HEAD>(C, cand_Wv, V);
    add_bias_f<T_DIM, D_HEAD>(V, cand_bv);
    for (int j = 0; j < D_HEAD; j++) V[T_DIM * D_HEAD + j] = cand_bias_v_row[j];

    float scores[4 * T_KV] = {0};
    {
        float Q[4 * D_HEAD];
        gemm_f<4, E_DIM, D_HEAD>(C, cand_Wq, Q);
        add_bias_f<T_DIM, D_HEAD>(Q, cand_bq);
        float K[T_KV * D_HEAD] = {0};
        gemm_f<4, E_DIM, D_HEAD>(C, cand_Wk, K);
        add_bias_f<T_DIM, D_HEAD>(K, cand_bk);
        for (int j = 0; j < D_HEAD; j++) K[T_DIM * D_HEAD + j] = cand_bias_k_row[j];
        float Kt[D_HEAD * T_KV];
        for (int i = 0; i < T_KV; i++)
            for (int j = 0; j < D_HEAD; j++) Kt[j * T_KV + i] = K[i * D_HEAD + j];
        gemm_f<4, D_HEAD, T_KV>(Q, Kt, scores);
    }
    for (int i = 0; i < 4 * T_KV; i++) scores[i] *= 0.5f;
    for (int i = 0; i < 4 * T_KV; i++) window_writeincr(scores_out, scores[i]);
    for (int i = 0; i < T_KV * D_HEAD; i++) window_writeincr(v_out, V[i]);
}
#endif
#if defined(HEAD_STAGE_POST)
void HEAD_POST_FN(input_window_float* __restrict scores_in,
                  input_window_float* __restrict v_in,
                  output_window_float* __restrict c_out)
{
    float scores[4 * T_KV];
    for (int i = 0; i < 4 * T_KV; i++) scores[i] = window_readincr(scores_in);
    float V[T_KV * D_HEAD];
    for (int i = 0; i < T_KV * D_HEAD; i++) V[i] = window_readincr(v_in);
    softmax_rows_f(scores, T_DIM, T_KV, T_KV);
    float out[4 * D_HEAD];
    gemm_f<4, T_KV, D_HEAD>(scores, V, out);
    for (int r = 0; r < T_DIM; r++)
        for (int c = 0; c < D_HEAD; c++) window_writeincr(c_out, out[r * D_HEAD + c]);
}
#endif
#endif // ATTN_TYPE_CAND

#if defined(ATTN_TYPE_CROSS)
#if defined(HEAD_STAGE_PRE)
void HEAD_PRE_FN(input_window_float* __restrict x_in,
                 input_window_float* __restrict c_in,
                 output_window_float* __restrict scores_out,
                 output_window_float* __restrict v_out)
{
    float X[N_MAX * E_DIM];
    for (int i = 0; i < N_MAX * E_DIM; i++) X[i] = window_readincr(x_in);
    float C[4 * E_DIM] = {0};
    for (int r = 0; r < T_DIM; r++)
        for (int c = 0; c < E_DIM; c++) C[r * E_DIM + c] = window_readincr(c_in);

    float V[T_KV * D_HEAD] = {0};
    gemm_f<4, E_DIM, D_HEAD>(C, cross_Wv, V);
    add_bias_f<T_DIM, D_HEAD>(V, cross_bv);
    for (int j = 0; j < D_HEAD; j++) V[T_DIM * D_HEAD + j] = cross_bias_v_row[j];

    float scores[N_MAX * T_KV] = {0};
    {
        float Q[N_MAX * D_HEAD];
        gemm_f<N_MAX, E_DIM, D_HEAD>(X, cross_Wq, Q);
        add_bias_f<N_MAX, D_HEAD>(Q, cross_bq);
        float K[T_KV * D_HEAD] = {0};
        gemm_f<4, E_DIM, D_HEAD>(C, cross_Wk, K);
        add_bias_f<T_DIM, D_HEAD>(K, cross_bk);
        for (int j = 0; j < D_HEAD; j++) K[T_DIM * D_HEAD + j] = cross_bias_k_row[j];
        float Kt[D_HEAD * T_KV];
        for (int i = 0; i < T_KV; i++)
            for (int j = 0; j < D_HEAD; j++) Kt[j * T_KV + i] = K[i * D_HEAD + j];
        gemm_f<N_MAX, D_HEAD, T_KV>(Q, Kt, scores);
    }
    for (int i = 0; i < N_MAX * T_KV; i++) scores[i] *= 0.5f;
    for (int i = 0; i < N_MAX * T_KV; i++) window_writeincr(scores_out, scores[i]);
    for (int i = 0; i < T_KV * D_HEAD; i++) window_writeincr(v_out, V[i]);
}
#endif
#if defined(HEAD_STAGE_POST)
void HEAD_POST_FN(input_window_float* __restrict scores_in,
                  input_window_float* __restrict v_in,
                  output_window_float* __restrict x_out)
{
    float scores[N_MAX * T_KV];
    for (int i = 0; i < N_MAX * T_KV; i++) scores[i] = window_readincr(scores_in);
    float V[T_KV * D_HEAD];
    for (int i = 0; i < T_KV * D_HEAD; i++) V[i] = window_readincr(v_in);
    softmax_rows_f(scores, N_MAX, T_KV, T_KV);
    float out[N_MAX * D_HEAD];
    gemm_f<N_MAX, T_KV, D_HEAD>(scores, V, out);
    for (int i = 0; i < N_MAX * D_HEAD; i++) window_writeincr(x_out, out[i]);
}
#endif
#endif // ATTN_TYPE_CROSS

#else  // !FLOAT_AIE -- the deployed int16 kernels
// vectorized tiled gemm: A packed 4x4-block-major, B row-major (gemm_utils.h)
#include "gemm_utils.h"
#include "win_vec.h"

// Window traffic and the bias/scale passes are vector (win_vec.h); the
// scalar window loops were most of each head kernel's cycles. The bias adds
// saturate (the scalar add_bias wrapped, which only differed on overflow).

// scale scores by 1/sqrt(d_head). Operates on the score buffer which is at
// PIPE_SCORE_SCALE (separate from PIPE_SCALE so Q*Kt doesn't saturate int16).
// (product >> shift) per lane under the default floor rounding, as before.
template <int N>
static inline void scale_scores_v(int16* __restrict scores, float inv_sqrt_d)
{
    const int16 scale_fixed = (int16)(inv_sqrt_d * PIPE_SCORE_SCALE);
    for (int i = 0; i < N; i += 16) {
        const aie::vector<int16, 16> v = aie::load_v<16>(&scores[i]);
        aie::store_v(&scores[i], aie::mul(v, scale_fixed).template to_vector<int16>(PIPE_SCORE_SHIFT));
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

// K (ROWS_KV x 4, row-major) -> Kt (4 x ROWS_KV)
template <int ROWS_KV>
static inline void transpose_k4(const int16* __restrict K, int16* __restrict Kt)
{
    for (int i = 0; i < ROWS_KV; i++)
        for (int j = 0; j < D_HEAD; j++)
            Kt[j * ROWS_KV + i] = K[i * D_HEAD + j];
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
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    // read X into packed layout (the gemms then run on pure vector loads).
    // The window carries N_MAX+1 rows: the extra row is the padding mask
    // (nonzero = padded jet), so BOTH layers get true key masking without
    // extra PLIOs (fixes the padded-key leak: previously padded keys kept
    // bias-only scores instead of -inf).
    alignas(16) int16 Xp[N_MAX * E_DIM];
    win_read_packed16<N_MAX>(x_in, Xp);
    alignas(16) int16 kmask[E_DIM];
    aie::store_v(kmask, win_read16(x_in));

    // V first - persists into the output stream
    alignas(16) int16 V[N_KV_PAD * D_HEAD];
    zero_v<N_KV_PAD * D_HEAD>(V);
    gemm_pk<N_MAX, E_DIM, D_HEAD>(Xp, Wv, V, PIPE_ACC_SHIFT);
    add_bias_v4<N_MAX>(V, bv);
    for (int j = 0; j < D_HEAD; j++) V[N_MAX * D_HEAD + j] = bias_v_row[j];

    alignas(16) int16 scores[N_MAX * N_KV_PAD];
    {
        // Q is N_MAX x 4: for K==4 the packed layout equals row-major, so Q
        // feeds the Q*Kt gemm below without repacking
        alignas(16) int16 Q[N_MAX * D_HEAD];
        gemm_pk<N_MAX, E_DIM, D_HEAD>(Xp, Wq, Q, PIPE_ACC_SHIFT);
        add_bias_v4<N_MAX>(Q, bq);

        alignas(16) int16 K[N_KV_PAD * D_HEAD];
        zero_v<N_KV_PAD * D_HEAD>(K);
        gemm_pk<N_MAX, E_DIM, D_HEAD>(Xp, Wk, K, PIPE_ACC_SHIFT);
        add_bias_v4<N_MAX>(K, bk);
        for (int j = 0; j < D_HEAD; j++) K[N_MAX * D_HEAD + j] = bias_k_row[j];

        alignas(16) int16 Kt[D_HEAD * N_KV_PAD];
        transpose_k4<N_KV_PAD>(K, Kt);

        gemm_pk<N_MAX, D_HEAD, N_KV_PAD>(Q, Kt, scores, PIPE_QKT_SHIFT);
    }

    scale_scores_v<N_MAX * N_KV_PAD>(scores, 0.5f);

    // hard-mask padded keys: -32000 pushes softmax past D_MAX -> exactly 0
    for (int j = 0; j < N_MAX; j++)
        if (kmask[j] != 0)
            for (int i = 0; i < N_MAX; i++)
                scores[i * N_KV_PAD + j] = -32000;

    win_write_v<N_MAX * N_KV_PAD>(scores_out, scores);
    win_write_v<N_KV_PAD * D_HEAD>(v_out, V);
    aie::set_saturation(sat_save);
}
#endif // HEAD_STAGE_PRE

// stage 2: + wij (layer 0 only) + softmax + attn*V
// Layer 1 has no wij bias: previously the bridge streamed 624 ZEROS per
// event through the NoC just so this kernel could read-and-ignore them.
// The L1 variant now simply has no wij port.
#if defined(HEAD_STAGE_POST)
#if ATTN_LAYER == 0
void HEAD_POST_FN(input_window_int16* __restrict scores_in,
                        input_window_int16* __restrict v_in,
                        input_window_int16* __restrict wij_in,
                        output_window_int16* __restrict x_out)
#else
void HEAD_POST_FN(input_window_int16* __restrict scores_in,
                        input_window_int16* __restrict v_in,
                        output_window_int16* __restrict x_out)
#endif
{
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    alignas(16) int16 scores[N_MAX * N_KV_PAD];
    win_read_v<N_MAX * N_KV_PAD>(scores_in, scores);

    alignas(16) int16 V[N_KV_PAD * D_HEAD];
    win_read_v<N_KV_PAD * D_HEAD>(v_in, V);

#if ATTN_LAYER == 0
    // wij is N_MAX x N_KV row-major (156 words): vector reads for the first
    // 144, scalar for the last 12, then each row is an unaligned 16-lane
    // load with lanes N_KV.. masked to zero before the saturating add.
    alignas(16) int16 wl[N_MAX * N_KV_PAD];
    constexpr int WV = (N_MAX * N_KV) / 16 * 16;
    win_read_v<WV>(wij_in, wl);
    for (int i = WV; i < N_MAX * N_KV; i++) wl[i] = window_readincr(wij_in);
    alignas(16) int16 lane_mask[16];
    for (int c = 0; c < 16; c++) lane_mask[c] = (c < N_KV) ? 1 : 0;
    const aie::vector<int16, 16> mv = aie::load_v<16>(lane_mask);
    for (int r = 0; r < N_MAX; r++) {
        const aie::vector<int16, 16> w = aie::load_unaligned_v<16>(&wl[r * N_KV]);
        const aie::vector<int16, 16> wm = aie::mul(w, mv).template to_vector<int16>(0);
        aie::store_v(&scores[r * N_KV_PAD],
                     add_sat16(aie::load_v<16>(&scores[r * N_KV_PAD]), wm));
    }
#endif

    // integer softmax, emitted directly in packed layout for the AV gemm
    alignas(16) int16 attn_p[N_MAX * N_KV_PAD];
    int_softmax_packed<N_MAX, N_KV, N_KV_PAD>(scores, attn_p);

    alignas(16) int16 head_out[N_MAX * D_HEAD];
    gemm_pk<N_MAX, N_KV_PAD, D_HEAD>(attn_p, V, head_out, PIPE_AV_SHIFT);

    win_write_v<N_MAX * D_HEAD>(x_out, head_out);
    aie::set_saturation(sat_save);
}
#endif // HEAD_STAGE_POST
#endif

// =====================================================================
// candidate self attention
// =====================================================================

#if defined(ATTN_TYPE_CAND)

#if defined(HEAD_STAGE_PRE)
void HEAD_PRE_FN(input_window_int16* __restrict c_in,
                        output_window_int16* __restrict scores_out,
                        output_window_int16* __restrict v_out)
{
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    alignas(16) int16 Cp[4 * E_DIM];  // packed; padded row 3 is zero
    win_read_packed16<T_DIM>(c_in, Cp);

    alignas(16) int16 V[T_KV * D_HEAD];
    gemm_pk<4, E_DIM, D_HEAD>(Cp, cand_Wv, V, PIPE_ACC_SHIFT);
    add_bias_v4<4>(V, cand_bv);
    for (int j = 0; j < D_HEAD; j++) V[T_DIM * D_HEAD + j] = cand_bias_v_row[j];

    alignas(16) int16 scores[4 * T_KV];
    {
        alignas(16) int16 Q[4 * D_HEAD];
        gemm_pk<4, E_DIM, D_HEAD>(Cp, cand_Wq, Q, PIPE_ACC_SHIFT);
        add_bias_v4<4>(Q, cand_bq);

        alignas(16) int16 K[T_KV * D_HEAD];
        gemm_pk<4, E_DIM, D_HEAD>(Cp, cand_Wk, K, PIPE_ACC_SHIFT);
        add_bias_v4<4>(K, cand_bk);
        for (int j = 0; j < D_HEAD; j++) K[T_DIM * D_HEAD + j] = cand_bias_k_row[j];

        alignas(16) int16 Kt[D_HEAD * T_KV];
        transpose_k4<T_KV>(K, Kt);

        gemm_pk<4, D_HEAD, T_KV>(Q, Kt, scores, PIPE_QKT_SHIFT);
    }

    scale_scores_v<4 * T_KV>(scores, 0.5f);

    win_write_v<4 * T_KV>(scores_out, scores);
    win_write_v<T_KV * D_HEAD>(v_out, V);
    aie::set_saturation(sat_save);
}
#endif // HEAD_STAGE_PRE

#if defined(HEAD_STAGE_POST)
void HEAD_POST_FN(input_window_int16* __restrict scores_in,
                         input_window_int16* __restrict v_in,
                         output_window_int16* __restrict c_out)
{
    alignas(16) int16 scores[4 * T_KV];
    win_read_v<4 * T_KV>(scores_in, scores);

    alignas(16) int16 V[T_KV * D_HEAD];
    win_read_v<T_KV * D_HEAD>(v_in, V);

    // integer softmax (K==4: packed == row-major)
    alignas(16) int16 attn_p[4 * T_KV];
    int_softmax_packed<T_DIM, T_KV, T_KV>(scores, attn_p);

    alignas(16) int16 out[4 * D_HEAD];
    gemm_pk<4, T_KV, D_HEAD>(attn_p, V, out, PIPE_AV_SHIFT);

    // the candidate head output window is 12 words: scalar
    for (int r = 0; r < T_DIM; r++)
        for (int c = 0; c < D_HEAD; c++)
            window_writeincr(c_out, out[r * D_HEAD + c]);
}
#endif // HEAD_STAGE_POST
#endif

// =====================================================================
// cross attention: queries from x (jets), keys/values from c (candidates)
// =====================================================================

#if defined(ATTN_TYPE_CROSS)

#if defined(HEAD_STAGE_PRE)
void HEAD_PRE_FN(input_window_int16* __restrict x_in,
                         input_window_int16* __restrict c_in,
                         output_window_int16* __restrict scores_out,
                         output_window_int16* __restrict v_out)
{
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    alignas(16) int16 Xp[N_MAX * E_DIM];
    win_read_packed16<N_MAX>(x_in, Xp);

    alignas(16) int16 Cp[4 * E_DIM];  // packed; padded row 3 is zero
    win_read_packed16<T_DIM>(c_in, Cp);

    alignas(16) int16 V[T_KV * D_HEAD];
    gemm_pk<4, E_DIM, D_HEAD>(Cp, cross_Wv, V, PIPE_ACC_SHIFT);
    add_bias_v4<4>(V, cross_bv);
    for (int j = 0; j < D_HEAD; j++) V[T_DIM * D_HEAD + j] = cross_bias_v_row[j];

    alignas(16) int16 scores[N_MAX * T_KV];
    {
        alignas(16) int16 Q[N_MAX * D_HEAD];
        gemm_pk<N_MAX, E_DIM, D_HEAD>(Xp, cross_Wq, Q, PIPE_ACC_SHIFT);
        add_bias_v4<N_MAX>(Q, cross_bq);

        alignas(16) int16 K[T_KV * D_HEAD];
        gemm_pk<4, E_DIM, D_HEAD>(Cp, cross_Wk, K, PIPE_ACC_SHIFT);
        add_bias_v4<4>(K, cross_bk);
        for (int j = 0; j < D_HEAD; j++) K[T_DIM * D_HEAD + j] = cross_bias_k_row[j];

        alignas(16) int16 Kt[D_HEAD * T_KV];
        transpose_k4<T_KV>(K, Kt);

        gemm_pk<N_MAX, D_HEAD, T_KV>(Q, Kt, scores, PIPE_QKT_SHIFT);
    }

    scale_scores_v<N_MAX * T_KV>(scores, 0.5f);

    win_write_v<N_MAX * T_KV>(scores_out, scores);
    win_write_v<T_KV * D_HEAD>(v_out, V);
    aie::set_saturation(sat_save);
}
#endif // HEAD_STAGE_PRE

#if defined(HEAD_STAGE_POST)
void HEAD_POST_FN(input_window_int16* __restrict scores_in,
                          input_window_int16* __restrict v_in,
                          output_window_int16* __restrict x_out)
{
    alignas(16) int16 scores[N_MAX * T_KV];
    win_read_v<N_MAX * T_KV>(scores_in, scores);

    alignas(16) int16 V[T_KV * D_HEAD];
    win_read_v<T_KV * D_HEAD>(v_in, V);

    // integer softmax (K==4: packed == row-major)
    alignas(16) int16 attn_p[N_MAX * T_KV];
    int_softmax_packed<N_MAX, T_KV, T_KV>(scores, attn_p);

    alignas(16) int16 out[N_MAX * D_HEAD];
    gemm_pk<N_MAX, T_KV, D_HEAD>(attn_p, V, out, PIPE_AV_SHIFT);

    win_write_v<N_MAX * D_HEAD>(x_out, out);
}
#endif // HEAD_STAGE_POST
#endif
#endif // !FLOAT_AIE
