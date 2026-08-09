// post-attention tiles:
//   post_a_proj: 4 head outputs interleaved + output projection + (skip) + LN
//   post_b1/b2:  FFN layers 0 and 1
//   post_c:      FFN layer 2 + skip with broadcast proj_out + LN
//
// proj_out from post_a fans out to both post_b (ffn input) and post_c
// (residual for the FFN skip).

#include "attn_post_kernel.h"
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
        #include "weights/obj_post_weights_L0.h"
    #elif ATTN_LAYER == 1
        #include "weights/obj_post_weights_L1.h"
    #endif
#endif

#if defined(ATTN_TYPE_CAND)
    #if ATTN_LAYER == 0
        #include "weights/cand_post_weights_L0.h"
    #elif ATTN_LAYER == 1
        #include "weights/cand_post_weights_L1.h"
    #endif
#endif

#if defined(ATTN_TYPE_CROSS)
    #if ATTN_LAYER == 0
        #include "weights/cross_post_weights_L0.h"
    #elif ATTN_LAYER == 1
        #include "weights/cross_post_weights_L1.h"
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

// vectorized tiled gemm: A packed 4x4-block-major, B row-major (gemm_utils.h)
#include "gemm_utils.h"

static void layernorm_row(int16* __restrict x, int n_rows, int n_cols,
                          const int16* __restrict gamma,
                          const int16* __restrict beta)
{
    const float eps = 1e-5f;
    // AIE1 has no fp32 hardware: every float divide is a softfloat call.
    // PIPE_SCALE and n_cols(=E_DIM) are powers of two, so multiplying by the
    // reciprocal is bit-exact; this removes ~50 emulated divides per row.
    const float inv_ps = 1.0f / PIPE_SCALE;      // compile-time constant
    const float inv_n  = 1.0f / n_cols;          // one divide per call
    // gamma/beta are per-column constants; convert once, not once per row
    float g_f[16], b_f[16];
    for (int c = 0; c < n_cols; c++) {
        g_f[c] = (float)gamma[c] * inv_ps;
        b_f[c] = (float)beta[c] * inv_ps;
    }
    for (int r = 0; r < n_rows; r++) {
        float row_f[16];
        float sum = 0.0f;
        for (int c = 0; c < n_cols; c++) {
            row_f[c] = (float)x[r * n_cols + c] * inv_ps;
            sum += row_f[c];
        }
        float mean = sum * inv_n;
        float var = 0.0f;
        for (int c = 0; c < n_cols; c++) {
            float d = row_f[c] - mean;
            var += d * d;
        }
        var *= inv_n;
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int c = 0; c < n_cols; c++) {
            float y = g_f[c] * (row_f[c] - mean) * inv_std + b_f[c];
            int32 y_fixed = (int32)(y*PIPE_SCALE);
            if (y_fixed > 32767) y_fixed = 32767;
            if (y_fixed < -32768) y_fixed = -32768;
            x[r * n_cols + c] = (int16)y_fixed;
        }
    }
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
