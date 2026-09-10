#include "embed_kernel.h"
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <adf.h>
#include "gemm_utils.h"

// Diagnostic: a scalar reference for the three matrix multiplies. x86sim agrees
// with the float model to 0.006 while the cycle-accurate simulator is off by
// 1.44, so something in the vector path behaves differently on the real core.
// Building with EMBED_SCALAR_GEMM swaps only the gemms, which separates a
// gemm_pk problem from a layer-norm or ReLU one.
#ifdef EMBED_SCALAR_GEMM
template <int M, int K, int N>
static void gemm_ref(const int16* __restrict Ap, const int16* __restrict B,
                     int16* __restrict C, int shift)
{
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            int64 acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int64)Ap[pk_idx<K>(m, k)] * (int64)B[k * N + n];
            int32 v = (int32)(acc >> shift);
            if (v > 32767) v = 32767;
            if (v < -32768) v = -32768;
            C[m * N + n] = (int16)v;
        }
}
#define GEMM_PK gemm_ref
#else
#define GEMM_PK gemm_pk
#endif
#include "weights/embed_weights.h"

// The embedding runs at the pipeline data scale, like the object and cross
// blocks. Layer norm is shared with the post-attention kernels.
#define PIPE_SCALE DATA_SCALE
#include "layernorm_int.h"

static void add_bias_sat(int16* __restrict mat, const int16* __restrict bias,
                         int n_rows, int n_cols)
{
    for (int r = 0; r < n_rows; r++)
        for (int c = 0; c < n_cols; c++) {
            int32 s = (int32)mat[r * n_cols + c] + (int32)bias[c];
            if (s > 32767) s = 32767;
            if (s < -32768) s = -32768;
            mat[r * n_cols + c] = (int16)s;
        }
}

static void relu_inplace(int16* __restrict x, int n)
{
    for (int i = 0; i < n; i += 16) {
        aie::vector<int16, 16> v = aie::load_v<16>(&x[i]);
        aie::store_v(&x[i], aie::max(v, aie::broadcast<int16, 16>(0)));
    }
}

void embed_mlp(input_window_int16* __restrict jets_in,
               output_window_int16* __restrict embed_out)
{
    // layer 0: 5 -> 16, read straight into packed layout with zero padding
    alignas(32) int16 a[EMBED_ROWS * EMBED_IN_PAD] = {0};
    for (int r = 0; r < EMBED_ROWS; r++)
        for (int c = 0; c < EMBED_IN; c++)
            a[pk_idx<EMBED_IN_PAD>(r, c)] = window_readincr(jets_in);
    // consume the window's alignment padding so the next event starts clean
    for (int i = EMBED_ROWS * EMBED_IN; i < EMBED_IN_WORDS; i++) (void)window_readincr(jets_in);

    alignas(32) int16 h[EMBED_ROWS * E_DIM];
    GEMM_PK<EMBED_ROWS, EMBED_IN_PAD, E_DIM>(a, embed_W0, h, ACC_SHIFT);
    add_bias_sat(h, embed_b0, EMBED_ROWS, E_DIM);
    layernorm_row(h, EMBED_ROWS, E_DIM, embed_ln0_g, embed_ln0_b);
    relu_inplace(h, EMBED_ROWS * E_DIM);

    // layer 1: 16 -> 16
    alignas(32) int16 ap[EMBED_ROWS * E_DIM];
    pack_a4<EMBED_ROWS, E_DIM>(h, ap);
    GEMM_PK<EMBED_ROWS, E_DIM, E_DIM>(ap, embed_W1, h, ACC_SHIFT);
    add_bias_sat(h, embed_b1, EMBED_ROWS, E_DIM);
    layernorm_row(h, EMBED_ROWS, E_DIM, embed_ln1_g, embed_ln1_b);
    relu_inplace(h, EMBED_ROWS * E_DIM);

    // layer 2: 16 -> 16, no norm
    pack_a4<EMBED_ROWS, E_DIM>(h, ap);
    alignas(32) int16 out[EMBED_ROWS * E_DIM];
    GEMM_PK<EMBED_ROWS, E_DIM, E_DIM>(ap, embed_W2, out, ACC_SHIFT);
    add_bias_sat(out, embed_b2, EMBED_ROWS, E_DIM);

    for (int i = 0; i < EMBED_ROWS * E_DIM; i++) window_writeincr(embed_out, out[i]);
}
