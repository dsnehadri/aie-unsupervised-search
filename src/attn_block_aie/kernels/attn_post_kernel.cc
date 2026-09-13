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
#define POST_BC_FN       _POST_FN_2(ATTN_TYPE_TAG, bc,       ATTN_LAYER)

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
#include "win_vec.h"

#include "layernorm_int.h"

// Window traffic is vector (win_vec.h). The old scalar window loops and the
// byte-wise memset of the "= {0}" stack arrays were ~80% of every post
// kernel's cycles; gemm_pk and layernorm_row were already vector.
// The residual/bias adds saturate exactly as the scalar int32 clamps did,
// which needs the saturating srs mode; set once per kernel and restored.

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

// =====================================================================
// post_a: concat 4 heads -> output projection (+skip, +LN)
// =====================================================================
// The head windows are read straight into the packed layout. For 16 columns
// packed block (r/4, h) is rows 4(r/4)..4(r/4)+3 of head h, which is 16
// contiguous words of that head's window: one vector read per block.
// The residual window is a fresh window per invocation, so the words this
// kernel does not read (the object block's key-mask row, the cross block's
// unused residual) need no draining.

#if defined(POST_STAGE_A_PROJ)
void POST_A_PROJ_FN(input_window_int16* __restrict head0_in,
                      input_window_int16* __restrict head1_in,
                      input_window_int16* __restrict head2_in,
                      input_window_int16* __restrict head3_in,
                      input_window_int16* __restrict residual_in,
                      output_window_int16* __restrict proj_out)
{
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);

    alignas(16) int16 concat[POST_N_ROWS_PAD * E_DIM];
    input_window_int16* __restrict heads[N_HEADS] = {head0_in, head1_in, head2_in, head3_in};
#if defined(ATTN_TYPE_CAND)
    // 3 rows x 4 per head (12 words): scalar into block (0, h), row 3 zero
    zero_v<POST_N_ROWS_PAD * E_DIM>(concat);
    for (int h = 0; h < N_HEADS; h++)
        for (int i = 0; i < POST_N_ROWS * D_HEAD; i++)
            concat[h * 16 + i] = window_readincr(heads[h]);
#else
    for (int rb = 0; rb < POST_N_ROWS_PAD / 4; rb++)
        for (int h = 0; h < N_HEADS; h++)
            aie::store_v(concat + (rb * N_HEADS + h) * 16, win_read16(heads[h]));
#endif

    alignas(16) int16 proj[POST_N_ROWS_PAD * E_DIM];
    gemm_pk<POST_N_ROWS_PAD, E_DIM, E_DIM>( concat, Wout, proj, PIPE_ACC_SHIFT);
    add_bias_v16<POST_N_ROWS>(proj, bout);

    #if !defined(ATTN_TYPE_CROSS)
    for (int r = 0; r < POST_N_ROWS; r++)
        aie::store_v(proj + r * E_DIM, add_sat16(aie::load_v<16>(proj + r * E_DIM), win_read16(residual_in)));
    #endif

    layernorm_row(proj, POST_N_ROWS, E_DIM, post_attn_ln_gamma, post_attn_ln_beta);

    win_write_v<POST_N_ROWS * E_DIM>(proj_out, proj);
    aie::set_saturation(sat_save);
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
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    alignas(16) int16 in[POST_N_ROWS_PAD * E_DIM];
    win_read_packed16<POST_N_ROWS>(proj_in, in);

    alignas(16) int16 out[POST_N_ROWS_PAD * E_DIM];
    gemm_pk<POST_N_ROWS_PAD, E_DIM, E_DIM>( in, ffn_W0, out, PIPE_ACC_SHIFT);
    add_bias_v16<POST_N_ROWS>(out, ffn_b0);
    layernorm_row(out, POST_N_ROWS, E_DIM, ffn_ln_gamma0, ffn_ln_beta0);
    relu_inplace(out, POST_N_ROWS * E_DIM);

    win_write_v<POST_N_ROWS * E_DIM>(ffn0_out, out);
    aie::set_saturation(sat_save);
}
#endif // POST_STAGE_B1

#if defined(POST_STAGE_B2)
void POST_B2_FN(input_window_int16* __restrict ffn0_in,
                  output_window_int16* __restrict ffn1_out)
{
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    alignas(16) int16 in[POST_N_ROWS_PAD * E_DIM];
    win_read_packed16<POST_N_ROWS>(ffn0_in, in);

    alignas(16) int16 out[POST_N_ROWS_PAD * E_DIM];
    gemm_pk<POST_N_ROWS_PAD, E_DIM, E_DIM>( in, ffn_W1, out, PIPE_ACC_SHIFT);
    add_bias_v16<POST_N_ROWS>(out, ffn_b1);
    layernorm_row(out, POST_N_ROWS, E_DIM, ffn_ln_gamma1, ffn_ln_beta1);
    relu_inplace(out, POST_N_ROWS * E_DIM);

    win_write_v<POST_N_ROWS * E_DIM>(ffn1_out, out);
    aie::set_saturation(sat_save);
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
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    alignas(16) int16 ffn1[POST_N_ROWS_PAD * E_DIM];
    win_read_packed16<POST_N_ROWS>(ffn_in, ffn1);

    alignas(16) int16 ffn2[POST_N_ROWS_PAD * E_DIM];
    gemm_pk<POST_N_ROWS_PAD, E_DIM, E_DIM>(ffn1, ffn_W2, ffn2, PIPE_ACC_SHIFT);
    add_bias_v16<POST_N_ROWS>(ffn2, ffn_b2);
    layernorm_row(ffn2, POST_N_ROWS, E_DIM, ffn_ln_gamma2, ffn_ln_beta2);
    relu_inplace(ffn2, POST_N_ROWS * E_DIM);

    // skip with broadcast proj_out
    for (int r = 0; r < POST_N_ROWS; r++)
        aie::store_v(ffn2 + r * E_DIM, add_sat16(aie::load_v<16>(ffn2 + r * E_DIM), win_read16(residual_b_in)));

    layernorm_row(ffn2, POST_N_ROWS, E_DIM, post_ffn_ln_gamma, post_ffn_ln_beta);
    win_write_v<POST_N_ROWS * E_DIM>(x_out, ffn2);
    aie::set_saturation(sat_save);
}
#endif // POST_STAGE_C

// =====================================================================
// post_bc (POST_MERGED): FFN layers 0, 1, 2 + skip + LN in ONE kernel.
// Same arithmetic as b1 -> b2 -> c; the two intermediate windows become
// local packs (pack_local16), and the FFN residual is the proj window this
// kernel already reads. Saves two window hops per block on the one-event
// chain; costs interval, since one tile now does the work of three.
// =====================================================================
#if defined(POST_STAGE_BC)
void POST_BC_FN(input_window_int16* __restrict proj_in,
                  output_window_int16* __restrict x_out)
{
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    // proj: row-major copy for the skip, packed copy for the first gemm
    alignas(16) int16 proj[POST_N_ROWS_PAD * E_DIM];
    win_read_v<POST_N_ROWS * E_DIM>(proj_in, proj);
    if constexpr (POST_N_ROWS_PAD > POST_N_ROWS)
        zero_v<(POST_N_ROWS_PAD - POST_N_ROWS) * E_DIM>(proj + POST_N_ROWS * E_DIM);
    alignas(16) int16 pk[POST_N_ROWS_PAD * E_DIM];
    pack_local16<POST_N_ROWS_PAD>(proj, pk);

    // FFN layer 0
    alignas(16) int16 h[POST_N_ROWS_PAD * E_DIM];
    gemm_pk<POST_N_ROWS_PAD, E_DIM, E_DIM>(pk, ffn_W0, h, PIPE_ACC_SHIFT);
    add_bias_v16<POST_N_ROWS>(h, ffn_b0);
    layernorm_row(h, POST_N_ROWS, E_DIM, ffn_ln_gamma0, ffn_ln_beta0);
    relu_inplace(h, POST_N_ROWS * E_DIM);
    if constexpr (POST_N_ROWS_PAD > POST_N_ROWS)
        zero_v<(POST_N_ROWS_PAD - POST_N_ROWS) * E_DIM>(h + POST_N_ROWS * E_DIM);
    pack_local16<POST_N_ROWS_PAD>(h, pk);

    // FFN layer 1
    gemm_pk<POST_N_ROWS_PAD, E_DIM, E_DIM>(pk, ffn_W1, h, PIPE_ACC_SHIFT);
    add_bias_v16<POST_N_ROWS>(h, ffn_b1);
    layernorm_row(h, POST_N_ROWS, E_DIM, ffn_ln_gamma1, ffn_ln_beta1);
    relu_inplace(h, POST_N_ROWS * E_DIM);
    if constexpr (POST_N_ROWS_PAD > POST_N_ROWS)
        zero_v<(POST_N_ROWS_PAD - POST_N_ROWS) * E_DIM>(h + POST_N_ROWS * E_DIM);
    pack_local16<POST_N_ROWS_PAD>(h, pk);

    // FFN layer 2 + skip + LN (post_c)
    gemm_pk<POST_N_ROWS_PAD, E_DIM, E_DIM>(pk, ffn_W2, h, PIPE_ACC_SHIFT);
    add_bias_v16<POST_N_ROWS>(h, ffn_b2);
    layernorm_row(h, POST_N_ROWS, E_DIM, ffn_ln_gamma2, ffn_ln_beta2);
    relu_inplace(h, POST_N_ROWS * E_DIM);
    add_rows_v16<POST_N_ROWS>(h, proj);
    layernorm_row(h, POST_N_ROWS, E_DIM, post_ffn_ln_gamma, post_ffn_ln_beta);
    win_write_v<POST_N_ROWS * E_DIM>(x_out, h);
    aie::set_saturation(sat_save);
}
#endif // POST_STAGE_BC
#endif // !FLOAT_AIE
