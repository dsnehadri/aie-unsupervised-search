// post-attention tiles:
//   post_a_proj: 4 head outputs interleaved + output projection + (skip) + LN
//   post_b1/b2:  FFN layers 0 and 1
//   post_c:      FFN layer 2 + skip with broadcast proj_out + LN
//
// proj_out from post_a fans out to both post_b (ffn input) and post_c
// (residual for the FFN skip).

#include "attn_post_kernel.h"

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

// aiecompiler dedups wrappers by function name (see attn_head_kernel.cc).
// Make each (type, stage, layer) post function unique.
#define _POST_FN_3(t, s, l) t##_post_##s##_L##l
#define _POST_FN_2(t, s, l) _POST_FN_3(t, s, l)
#define POST_A_PROJ_FN   _POST_FN_2(ATTN_TYPE_TAG, a_proj,   ATTN_LAYER)
#define POST_B1_FN       _POST_FN_2(ATTN_TYPE_TAG, b1,       ATTN_LAYER)
#define POST_B2_FN       _POST_FN_2(ATTN_TYPE_TAG, b2,       ATTN_LAYER)
#define POST_C_FN        _POST_FN_2(ATTN_TYPE_TAG, c,        ATTN_LAYER)

// Pipeline-wide scale: cand uses Q6.9; obj/cross use Q4.11. See attn_head_kernel.cc.
#if defined(ATTN_TYPE_CAND)
#define PIPE_SCALE      CAND_SCALE
#define PIPE_ACC_SHIFT  CAND_ACC_SHIFT
#else
#define PIPE_SCALE      DATA_SCALE
#define PIPE_ACC_SHIFT  ACC_SHIFT
#endif

#if defined(ATTN_TYPE_OBJ)
    #if ATTN_LAYER == 0
        #include AIE_XSTR(WEIGHTS_DIR/obj_post_weights_L0.h)
    #elif ATTN_LAYER == 1
        #include AIE_XSTR(WEIGHTS_DIR/obj_post_weights_L1.h)
    #endif
#endif

#if defined(ATTN_TYPE_CAND)
    #if ATTN_LAYER == 0
        #include AIE_XSTR(WEIGHTS_DIR/cand_post_weights_L0.h)
    #elif ATTN_LAYER == 1
        #include AIE_XSTR(WEIGHTS_DIR/cand_post_weights_L1.h)
    #endif
#endif

#if defined(ATTN_TYPE_CROSS)
    #if ATTN_LAYER == 0
        #include AIE_XSTR(WEIGHTS_DIR/cross_post_weights_L0.h)
    #elif ATTN_LAYER == 1
        #include AIE_XSTR(WEIGHTS_DIR/cross_post_weights_L1.h)
    #endif
#endif

#if defined(ATTN_TYPE_OBJ) || defined(ATTN_TYPE_CROSS)
    #define POST_N_ROWS N_MAX // 12
    #define POST_N_ROWS_PAD 12
#elif defined(ATTN_TYPE_CAND)
    #define POST_N_ROWS T_DIM
    #define POST_N_ROWS_PAD 4
#else
    #error "define ATTN_TYPE_OBJ, ATTN_TYPE_CAND or ATTN_TYPE_CROSS"
#endif

#ifdef FLOAT_AIE
// ================= FLOAT_AIE: unquantized reference (see head kernel) =====
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
static void add_bias_f(float* m, const float* b, int R, int C)
{
    for (int r = 0; r < R; r++) for (int c = 0; c < C; c++) m[r * C + c] += b[c];
}
static void layernorm_f(float* x, int n_rows, int n_cols, const float* g, const float* b)
{
    const float eps = 1e-5f;
    for (int r = 0; r < n_rows; r++) {
        float sum = 0.0f;
        for (int c = 0; c < n_cols; c++) sum += x[r * n_cols + c];
        float mean = sum / n_cols;
        float var = 0.0f;
        for (int c = 0; c < n_cols; c++) { float d = x[r * n_cols + c] - mean; var += d * d; }
        var /= n_cols;
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int c = 0; c < n_cols; c++)
            x[r * n_cols + c] = g[c] * (x[r * n_cols + c] - mean) * inv_std + b[c];
    }
}
static void relu_f(float* x, int n) { for (int i = 0; i < n; i++) if (x[i] < 0) x[i] = 0; }

#if defined(POST_STAGE_A_PROJ)
void POST_A_PROJ_FN(input_window_float* __restrict head0_in,
                    input_window_float* __restrict head1_in,
                    input_window_float* __restrict head2_in,
                    input_window_float* __restrict head3_in,
                    input_window_float* __restrict residual_in,
                    output_window_float* __restrict proj_out)
{
    float concat[POST_N_ROWS * E_DIM];
    input_window_float* __restrict heads[N_HEADS] = {head0_in, head1_in, head2_in, head3_in};
    for (int r = 0; r < POST_N_ROWS; r++)
        for (int h = 0; h < N_HEADS; h++)
            for (int d = 0; d < D_HEAD; d++)
                concat[r * E_DIM + h * D_HEAD + d] = window_readincr(heads[h]);

    float proj[POST_N_ROWS * E_DIM];
    gemm_f<POST_N_ROWS, E_DIM, E_DIM>(concat, Wout, proj);
    add_bias_f(proj, bout, POST_N_ROWS, E_DIM);
#if !defined(ATTN_TYPE_CROSS)
    for (int i = 0; i < POST_N_ROWS * E_DIM; i++) proj[i] += window_readincr(residual_in);
#else
    for (int i = 0; i < POST_N_ROWS * E_DIM; i++) (void)window_readincr(residual_in);
#endif
    layernorm_f(proj, POST_N_ROWS, E_DIM, post_attn_ln_gamma, post_attn_ln_beta);
    for (int i = 0; i < POST_N_ROWS * E_DIM; i++) window_writeincr(proj_out, proj[i]);
}
#endif

#if defined(POST_STAGE_B1)
void POST_B1_FN(input_window_float* __restrict proj_in, output_window_float* __restrict ffn0_out)
{
    float in[POST_N_ROWS * E_DIM];
    for (int i = 0; i < POST_N_ROWS * E_DIM; i++) in[i] = window_readincr(proj_in);
    float out[POST_N_ROWS * E_DIM];
    gemm_f<POST_N_ROWS, E_DIM, E_DIM>(in, ffn_W0, out);
    add_bias_f(out, ffn_b0, POST_N_ROWS, E_DIM);
    layernorm_f(out, POST_N_ROWS, E_DIM, ffn_ln_gamma0, ffn_ln_beta0);
    relu_f(out, POST_N_ROWS * E_DIM);
    for (int i = 0; i < POST_N_ROWS * E_DIM; i++) window_writeincr(ffn0_out, out[i]);
}
#endif

#if defined(POST_STAGE_B2)
void POST_B2_FN(input_window_float* __restrict ffn0_in, output_window_float* __restrict ffn1_out)
{
    float in[POST_N_ROWS * E_DIM];
    for (int i = 0; i < POST_N_ROWS * E_DIM; i++) in[i] = window_readincr(ffn0_in);
    float out[POST_N_ROWS * E_DIM];
    gemm_f<POST_N_ROWS, E_DIM, E_DIM>(in, ffn_W1, out);
    add_bias_f(out, ffn_b1, POST_N_ROWS, E_DIM);
    layernorm_f(out, POST_N_ROWS, E_DIM, ffn_ln_gamma1, ffn_ln_beta1);
    relu_f(out, POST_N_ROWS * E_DIM);
    for (int i = 0; i < POST_N_ROWS * E_DIM; i++) window_writeincr(ffn1_out, out[i]);
}
#endif

#if defined(POST_STAGE_C)
void POST_C_FN(input_window_float* __restrict ffn_in,
               input_window_float* __restrict residual_b_in,
               output_window_float* __restrict x_out)
{
    float ffn1[POST_N_ROWS * E_DIM];
    for (int i = 0; i < POST_N_ROWS * E_DIM; i++) ffn1[i] = window_readincr(ffn_in);
    float ffn2[POST_N_ROWS * E_DIM];
    gemm_f<POST_N_ROWS, E_DIM, E_DIM>(ffn1, ffn_W2, ffn2);
    add_bias_f(ffn2, ffn_b2, POST_N_ROWS, E_DIM);
    layernorm_f(ffn2, POST_N_ROWS, E_DIM, ffn_ln_gamma2, ffn_ln_beta2);
    relu_f(ffn2, POST_N_ROWS * E_DIM);
    for (int i = 0; i < POST_N_ROWS * E_DIM; i++) ffn2[i] += window_readincr(residual_b_in);
    layernorm_f(ffn2, POST_N_ROWS, E_DIM, post_ffn_ln_gamma, post_ffn_ln_beta);
    for (int i = 0; i < POST_N_ROWS * E_DIM; i++) window_writeincr(x_out, ffn2[i]);
}
#endif

#else  // !FLOAT_AIE -- the deployed int16 kernels
// vectorized tiled gemm: A packed 4x4-block-major, B row-major (gemm_utils.h)
#include "gemm_utils.h"

// ---------------------------------------------------------------------------
// Integer layer norm, vector int16 with a 32-bit reciprocal-square-root
// lookup. AIE1 has no scalar FPU and no 64-bit ALU: the float version was
// ~2300 cycles/row of softfloat calls, and a first integer version (64-bit
// restoring sqrt + 64-bit divide, commit 49dcb21) was ~4100 cycles/row of
// emulated 64-bit ops. This one uses only 32-bit scalar ops and 16-bit
// vector multiplies.
//
// Math (x, gamma, beta all at PIPE_SCALE = 2^F; n_cols = 16):
//   d    = 16*x - sum(x)                       exact, = 16*(x - mean)
//   V    = sum d^2 + 256*EPS_V                 EPS_V = eps*16*2^(2F)
//   y    = 4*g*d / sqrt(V) + b                 (2^F factors cancel)
// The mean is never rounded (a half-LSB mean error is a ~0.2 error in every
// output of a low-spread row).
//
// Fixed-point plan, per row:
//   dn   = d * 2^-kd, kd so that |dn| <= 2^14, kd may be negative (an exact
//          upshift for low-spread rows, so S below keeps full precision;
//          kd >= KD_MIN keeps the eps term inside int32)
//   S    = sum (dn^2 >> 2)  <= 2^30               (vector mul, reduce)
//   W    = S + 64*EPS_V/2^(2kd)  ->  V = 4*2^(2kd)*W   (EPS_V = eps*16*2^(2F), kept with 2 fraction bits)
//   Wn   = W << e, e even, Wn in [2^30, 2^32)  ->  M = Wn/2^32 in [0.25,1)
//   R    = 1/sqrt(M) from LN_RSQRT_LUT (385 entries, linear interp), Rq = R*2^14
//   1/sqrt(V) = R * 2^(e/2 - kd - 17)
//   dn2  = (dn*Rq) >> sd, sd so that |dn2| <= 2^15  (vector, rounded)
//   y    = (g*dn2) >> (29 - sd - e/2) + b           (vector, rounded, saturated)
// Every rounding is a vector srs with round-to-nearest-even; the rounding and
// saturation modes are set here and restored, so gemm_pk is unaffected.
// Precision: dn and dn2 keep 14-15 bits of the row's largest |d|, R is good
// to 1e-5; the unit test (scratchpad lntest/t2.cpp) shows max 0.5 LSB vs the
// float reference over 20k random rows, including near-constant rows.
// ---------------------------------------------------------------------------
#include "ln_rsqrt_lut.h"

static constexpr int bitlen32_ce(uint32 v) { return v ? 1 + bitlen32_ce(v >> 1) : 0; }

static inline int bitlen32(uint32 v)
{
    int n = 0;
    if (v >> 16) { n += 16; v >>= 16; }
    if (v >> 8)  { n += 8;  v >>= 8;  }
    if (v >> 4)  { n += 4;  v >>= 4;  }
    if (v >> 2)  { n += 2;  v >>= 2;  }
    if (v >> 1)  { n += 1;  v >>= 1;  }
    return n + (int)v;
}

static void layernorm_row(int16* __restrict x, int n_rows, int n_cols,
                          const int16* __restrict gamma,
                          const int16* __restrict beta)
{
    // n_cols is always E_DIM = 16 here
    // eps term with 2 extra fraction bits: rounding EPS_V itself to an integer
    // is a 0.14% error at F=9, which is 2 LSB on a high-gain, low-spread row
    constexpr int32 EPS_W4 = (int32)(256.0f * 1e-5f * 16.0f * PIPE_SCALE * PIPE_SCALE + 0.5f); // 4*64*EPS_V
    constexpr int KD_MIN = -((32 - bitlen32_ce((uint32)EPS_W4)) / 2);         // EPS_W4 << (-2*KD_MIN-2) < 2^30

    const aie::rounding_mode   rnd_save = aie::swap_rounding(aie::rounding_mode::conv_even);
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);

    const aie::vector<int16, 16> gv = aie::load_v<16>(gamma);
    const aie::vector<int32, 16> bv = aie::from_vector<acc48>(aie::load_v<16>(beta)).to_vector<int32>(0); // int16 -> int32 (unpack() is int8-only in the 2022.2 API)

    for (int r = 0; r < n_rows; r++) {
        int16* row = x + r * n_cols;
        const aie::vector<int32, 16> x32 = aie::from_vector<acc48>(aie::load_v<16>(row)).to_vector<int32>(0);
        const int32 sum = aie::reduce_add(x32);
        const aie::vector<int32, 16> d32 =
            aie::sub(aie::upshift(x32, 4), aie::broadcast<int32, 16>(sum));   // |d| < 2^20
        const int32 m = aie::reduce_max(aie::abs(d32));

        int kd = bitlen32((uint32)m) - 14; if (kd < KD_MIN) kd = KD_MIN;     // |dn| <= 2^14
        const int up = kd < 0 ? -kd : 0, down = kd > 0 ? kd : 0;
        const aie::vector<int16, 16> dn = aie::from_vector<acc80>(d32, up).to_vector<int16>(down);
        const int32 S = aie::reduce_add(aie::mul(dn, dn).to_vector<int32>(2)); // <= 2^30
        const int32 W = S + (kd >= 0 ? (EPS_W4 >> (2 * kd + 2)) : (EPS_W4 << (2 * up - 2))); // < 2^31

        const int e = (32 - bitlen32((uint32)W)) & ~1;                        // even
        const uint32 Wn = (uint32)W << e;                                     // [2^30, 2^32)
        const int idx = (int)(Wn >> 23) - 128;                                // [0, 384)
        const int32 frac = (int32)((Wn >> 7) & 0xFFFF);
        const int32 l0 = LN_RSQRT_LUT[idx], l1 = LN_RSQRT_LUT[idx + 1];
        const int32 R16 = l0 + (((l1 - l0) * frac) >> 16);                    // Q16, (2^16, 2^17]
        int32 Rq = (R16 + 2) >> 2; if (Rq > 32767) Rq = 32767;                // Q14

        const int32 mq = kd >= 0 ? (m >> kd) : (m << up);                     // ~max |dn|, <= 2^14
        int sd = bitlen32((uint32)(mq * Rq)) - 15; if (sd < 0) sd = 0;        // |dn2| <= 2^15
        const aie::vector<int16, 16> dn2 = aie::mul(dn, (int16)Rq).to_vector<int16>(sd);

        int sy = 29 - sd - (e >> 1); if (sy < 0) sy = 0;                     // >= 11 in practice
        aie::vector<int32, 16> y32 = aie::mul(gv, dn2).to_vector<int32>(sy);
        y32 = aie::add(y32, bv);
        aie::store_v(row, aie::from_vector<acc80>(y32).to_vector<int16>(0));  // saturate
    }

    aie::set_rounding(rnd_save);
    aie::set_saturation(sat_save);
}

static void relu_inplace(int16* __restrict x, int n)
{
    for(int i = 0; i < n; i += 16) {
        int remaining = (n-i >= 16) ? 16 : n -i;
        if (remaining == 16) {
            aie::vector<int16, 16> v = aie::load_v<16>(&x[i]);
            v = aie::max(v, aie::broadcast<int16, 16>(0));
            aie::store_v(&x[i], v);
        } else {
            for(int j = 0; j < remaining; j++) {
                if (x[i + j] < 0) x[i + j] = 0;
            }
        }
    }
}

static void add_bias_sat(int16* __restrict mat, const int16* __restrict bias, int n_rows, int n_cols)
{
    for (int r = 0; r < n_rows; r++) {
        for (int c = 0; c < n_cols; c++) {
            int32 s = (int32)mat[r * n_cols + c] + (int32)bias[c];
            if (s > 32767) s = 32767;
            if (s < -32768) s = -32768;
            mat[r * n_cols + c] = (int16)s;
        }
    }
}

// =====================================================================
// post_a: concat 4 heads -> output projection (+skip, +LN)
// residual streamed in row-by-row so we don't buffer it on stack.
// =====================================================================
// The head windows are read and interleaved directly here; the former
// post_a_concat tile was pure data movement (one tile per subgraph, 6
// tiles across the design, doing scalar copies) and is gone.

#if defined(POST_STAGE_A_PROJ)
void POST_A_PROJ_FN(input_window_int16* __restrict head0_in,
                      input_window_int16* __restrict head1_in,
                      input_window_int16* __restrict head2_in,
                      input_window_int16* __restrict head3_in,
                      input_window_int16* __restrict residual_in,
                      output_window_int16* __restrict proj_out)
{
    // interleave the 4 head outputs straight into packed layout (gemm_utils.h)
    alignas(16) int16 concat[POST_N_ROWS_PAD * E_DIM] = {0};
    input_window_int16* __restrict heads[N_HEADS] = {head0_in, head1_in, head2_in, head3_in};
    for (int r = 0; r < POST_N_ROWS; r++)
        for (int h = 0; h < N_HEADS; h++)
            for (int d = 0; d < D_HEAD; d++)
                concat[pk_idx<E_DIM>(r, h * D_HEAD + d)] = window_readincr(heads[h]);

    alignas(16) int16 proj[POST_N_ROWS_PAD * E_DIM];
    gemm_pk<POST_N_ROWS_PAD, E_DIM, E_DIM>( concat, Wout, proj, PIPE_ACC_SHIFT);
    add_bias_sat(proj, bout, POST_N_ROWS, E_DIM);

    #if !defined(ATTN_TYPE_CROSS)
    for (int r = 0; r < POST_N_ROWS; r++) {
        for (int c = 0; c < E_DIM; c++) {
            int32 sum = (int32)proj[r*E_DIM+c] + (int32)window_readincr(residual_in);
            if (sum > 32767) sum = 32767;
            if (sum < -32768) sum = -32768;
            proj[r*E_DIM+c] = (int16)sum;
        }
    }
    #else
    for (int i = 0; i < POST_N_ROWS * E_DIM; i++) (void)window_readincr(residual_in);
    #endif

    layernorm_row(proj, POST_N_ROWS, E_DIM, post_attn_ln_gamma, post_attn_ln_beta);

    for (int i = 0; i < POST_N_ROWS * E_DIM; i++) window_writeincr(proj_out, proj[i]);
}
#endif // POST_STAGE_A_PROJ

// =====================================================================
// post_b split across TWO tiles to fit AIE-1 16 KB program memory:
//   post_b1: FFN layer 0
//   post_b2: FFN layer 1
// =====================================================================
#if defined(POST_STAGE_B1)
void POST_B1_FN(input_window_int16* __restrict proj_in,
                  output_window_int16* __restrict ffn0_out)
{
    alignas(16) int16 in[POST_N_ROWS_PAD * E_DIM] = {0};
    for (int r = 0; r < POST_N_ROWS; r++)
        for (int c = 0; c < E_DIM; c++)
            in[pk_idx<E_DIM>(r, c)] = window_readincr(proj_in);

    alignas(16) int16 out[POST_N_ROWS_PAD * E_DIM];
    gemm_pk<POST_N_ROWS_PAD, E_DIM, E_DIM>( in, ffn_W0, out, PIPE_ACC_SHIFT);
    add_bias_sat(out, ffn_b0, POST_N_ROWS, E_DIM);
    layernorm_row(out, POST_N_ROWS, E_DIM, ffn_ln_gamma0, ffn_ln_beta0);
    relu_inplace(out, POST_N_ROWS * E_DIM);

    for (int i = 0; i < POST_N_ROWS * E_DIM; i++) window_writeincr(ffn0_out, out[i]);
}
#endif // POST_STAGE_B1

#if defined(POST_STAGE_B2)
void POST_B2_FN(input_window_int16* __restrict ffn0_in,
                  output_window_int16* __restrict ffn1_out)
{
    alignas(16) int16 in[POST_N_ROWS_PAD * E_DIM] = {0};
    for (int r = 0; r < POST_N_ROWS; r++)
        for (int c = 0; c < E_DIM; c++)
            in[pk_idx<E_DIM>(r, c)] = window_readincr(ffn0_in);

    alignas(16) int16 out[POST_N_ROWS_PAD * E_DIM];
    gemm_pk<POST_N_ROWS_PAD, E_DIM, E_DIM>( in, ffn_W1, out, PIPE_ACC_SHIFT);
    add_bias_sat(out, ffn_b1, POST_N_ROWS, E_DIM);
    layernorm_row(out, POST_N_ROWS, E_DIM, ffn_ln_gamma1, ffn_ln_beta1);
    relu_inplace(out, POST_N_ROWS * E_DIM);

    for (int i = 0; i < POST_N_ROWS * E_DIM; i++) window_writeincr(ffn1_out, out[i]);
}
#endif // POST_STAGE_B2

// =====================================================================
// post_c: FFN layer 2 + skip with proj_out broadcast + LN
// residual_b is proj_out from post_a, fanned out by the graph
// =====================================================================
#if defined(POST_STAGE_C)
void POST_C_FN(input_window_int16* __restrict ffn_in,
                 input_window_int16* __restrict residual_b_in,
                 output_window_int16* __restrict x_out)
{
    alignas(16) int16 ffn1[POST_N_ROWS_PAD * E_DIM] = {0};
    for (int r = 0; r < POST_N_ROWS; r++)
        for (int c = 0; c < E_DIM; c++)
            ffn1[pk_idx<E_DIM>(r, c)] = window_readincr(ffn_in);

    alignas(16) int16 ffn2[POST_N_ROWS_PAD * E_DIM];
    gemm_pk<POST_N_ROWS_PAD, E_DIM, E_DIM>(ffn1, ffn_W2, ffn2, PIPE_ACC_SHIFT);
    add_bias_sat(ffn2, ffn_b2, POST_N_ROWS, E_DIM);
    layernorm_row(ffn2, POST_N_ROWS, E_DIM, ffn_ln_gamma2, ffn_ln_beta2);
    relu_inplace(ffn2, POST_N_ROWS * E_DIM);

    // skip with broadcast proj_out (streamed)
    for (int r = 0; r < POST_N_ROWS; r++) {
        for (int c = 0; c < E_DIM; c++) {
            int32 sum = (int32)ffn2[r*E_DIM+c] + (int32)window_readincr(residual_b_in);
            if (sum > 32767) sum = 32767;
            if (sum < -32768) sum = -32768;
            ffn2[r*E_DIM+c] = (int16)sum;
        }
    }

    layernorm_row(ffn2, POST_N_ROWS, E_DIM, post_ffn_ln_gamma, post_ffn_ln_beta);

    for (int i = 0; i < POST_N_ROWS * E_DIM; i++) window_writeincr(x_out, ffn2[i]);
}
#endif // POST_STAGE_C
#endif // !FLOAT_AIE
