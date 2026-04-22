// concatenates head outputs, applies output projection, skip+LN, FFN, skip+LN

#include "attn_post_kernel.h"
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <adf.h>

// object self attention weights

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


// titled gmem C[M][N] += A[M][K] x B[K][N]

template <int M, int K, int N>
inline void gemm_tile(const int16* __restrict A, const int16* __restrict B, int16* __restrict C, int shift)
{
    for (int m = 0; m < M; m += 4) {
        for (int n = 0; n < N; n+= 4) {
            aie::mmul<4, 4, 4, int16, int16> acc;
            acc.zero();

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
                        vb[j * 4 + i] = B[(k+i) * N + (n+j)];
                acc.mac(va, vb);
            }
            aie::vector<int16, 16> res = acc.to_vector<int16>(shift);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    C[(m+i) * N + (n+j)] =  res[i*4 + j];
        }
    }
}

// helper layernorm (float internal)

inline void layernorm_row(int16* __restrict x, int n_rows, int n_cols,
                          const int16* __restrict gamma,
                          const int16* __restrict beta)

{
    const float eps = 1e-5f;

    for (int r = 0; r < n_rows; r++) {
        float row_f[16];
        float sum = 0.0f;
        for (int c = 0; c < n_cols; c++) {
            row_f[c] = (float)x[r * n_cols + c] / DATA_SCALE;
            sum += row_f[c];
        }
        float mean = sum / n_cols;

        float var = 0.0f;
        for (int c = 0; c < n_cols; c++) {
            float d =  row_f[c] - mean;
            var += d * d;
        }
        var /= n_cols;
        float inv_std = 1.0f / sqrtf(var + eps);

        for (int c = 0; c < n_cols; c++) {
            float g = (float)gamma[c] / DATA_SCALE;
            float b = (float)beta[c] / DATA_SCALE;
            float y = g * (row_f[c] - mean) * inv_std + b;
            int32 y_fixed = (int32)(y*DATA_SCALE);
            if (y_fixed > 32767) y_fixed = 32767;
            if (y_fixed < -32768) y_fixed = -32768;
            x[r * n_cols + c] = (int16)y_fixed;
        }
    }
}

inline void relu_inplace(int16* __restrict x, int n)
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

// add bias + saturate
inline void add_bias_sat(int16* __restrict mat, const int16* __restrict bias, int n_rows, int n_cols)
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

// saturating add

inline void skip_add(int16* __restrict dst, const int16* __restrict src, int n)
{
    for (int i = 0; i < n; i++) {
        int32 sum = (int32)dst[i] + (int32)src[i];
        if (sum > 32767) sum = 32767;
        if (sum < -32768) sum = -32768;
        dst[i] = (int16)sum;
    }
}

// main post attention kernel

void attn_post(input_window_int16* __restrict head0_in,
                input_window_int16* __restrict head1_in,
                input_window_int16* __restrict head2_in,
                input_window_int16* __restrict head3_in,
                input_window_int16* __restrict residual_in,
                output_window_int16* __restrict x_out)
{
    // concatenate heads: [N_ROWS_PAD][4*D_HEAD] = [N_ROWS_PAD][16]
    alignas(16) int16 concat[POST_N_ROWS_PAD * E_DIM] = {0};
    for (int r = 0; r < POST_N_ROWS; r++) {
        for (int c = 0; c < D_HEAD; c++)
            concat[r * E_DIM + 0 * D_HEAD + c] = window_readincr(head0_in);
        for (int c = 0; c < D_HEAD; c++)
            concat[r * E_DIM + 1 * D_HEAD + c] = window_readincr(head1_in);
        for (int c = 0; c < D_HEAD; c++)
            concat[r * E_DIM + 2 * D_HEAD + c] = window_readincr(head2_in);
        for (int c = 0; c < D_HEAD; c++)
            concat[r * E_DIM + 3 * D_HEAD + c] = window_readincr(head3_in);
    }

    // read residual for skip connection

    alignas(16) int16 residual[POST_N_ROWS_PAD * E_DIM] = {0};
    for (int i = 0; i < POST_N_ROWS * E_DIM; i++)
        residual[i] = window_readincr(residual_in);

    // output projection: [N_ROWS_PAD][16] x Wout[16][16] -> [N_ROWS_PAD][16]
    alignas(16) int16 proj[POST_N_ROWS_PAD * E_DIM];
    gemm_tile<POST_N_ROWS_PAD, E_DIM, E_DIM>(concat, Wout, proj, ACC_SHIFT);
    add_bias_sat(proj, bout, POST_N_ROWS, E_DIM);

    // skip connection + post-attention layernorm
    #if !defined(ATTN_TYPE_CROSS)
        skip_add(proj, residual, POST_N_ROWS*E_DIM);
    #endif
    layernorm_row(proj, POST_N_ROWS, E_DIM, post_attn_ln_gamma, post_attn_ln_beta);

    // FFN 
    // save residual for FFN skip connection
    alignas(16) int16 ffn_residual[POST_N_ROWS_PAD * E_DIM];
    for (int i = 0; i < POST_N_ROWS * E_DIM; i++)
        ffn_residual[i] = proj[i];
    
    alignas(16) int16 buf_a[POST_N_ROWS_PAD * E_DIM];
    alignas(16) int16 buf_b[POST_N_ROWS_PAD * E_DIM];

    // layer 0: linear + ln + relu
    gemm_tile<POST_N_ROWS_PAD, E_DIM, E_DIM>(proj, ffn_W0, buf_a, ACC_SHIFT);
    add_bias_sat(buf_a, ffn_b0, POST_N_ROWS, E_DIM);
    layernorm_row(buf_a, POST_N_ROWS, E_DIM, ffn_ln_gamma0, ffn_ln_beta0);
    relu_inplace(buf_a, POST_N_ROWS * E_DIM);

    // layer 1: linear + ln + relu
    gemm_tile<POST_N_ROWS_PAD, E_DIM, E_DIM>(buf_a, ffn_W1, buf_b, ACC_SHIFT);
    add_bias_sat(buf_b, ffn_b1, POST_N_ROWS, E_DIM);
    layernorm_row(buf_b, POST_N_ROWS, E_DIM, ffn_ln_gamma1, ffn_ln_beta1);
    relu_inplace(buf_b, POST_N_ROWS * E_DIM);

    // layer 2: linear + ln + relu
    gemm_tile<POST_N_ROWS_PAD, E_DIM, E_DIM>(buf_b, ffn_W2, buf_a, ACC_SHIFT);
    add_bias_sat(buf_a, ffn_b2, POST_N_ROWS, E_DIM);
    layernorm_row(buf_a, POST_N_ROWS, E_DIM, ffn_ln_gamma2, ffn_ln_beta2);
    relu_inplace(buf_a, POST_N_ROWS * E_DIM);

    // ffn skip connection + post ffn layernorm

    skip_add(buf_a, ffn_residual, POST_N_ROWS * E_DIM);
    layernorm_row(buf_a, POST_N_ROWS, E_DIM, post_ffn_ln_gamma, post_ffn_ln_beta);

    // write output

    for (int i = 0; i < POST_N_ROWS * E_DIM; i++) {
        window_writeincr(x_out, buf_a[i]);
    }

}