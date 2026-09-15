#include "embed_kernel.h"
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <adf.h>
#include "gemm_utils.h"
#include "win_vec.h"

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

// bias adds: add_bias_v16 (win_vec.h), the same saturating int32 add as before

static void relu_inplace(int16* __restrict x, int n)
{
    for (int i = 0; i < n; i += 16) {
        aie::vector<int16, 16> v = aie::load_v<16>(&x[i]);
        aie::store_v(&x[i], aie::max(v, aie::broadcast<int16, 16>(0)));
    }
}

// TRANSPOSED_EMBED: the transposed embedding overflowed the 16 KB program memory
// (16,900 B with three layers and two transposes) and was no faster in round 1;
// the embedding keeps the vector-I/O kernel unless asked for explicitly.
#if defined(TRANSPOSED) && defined(TRANSPOSED_EMBED)
#include "embed_kernel_t.cc"
#else
// EMBED_PIPE: the embedding as three tiles instead of one.
//
// The MLP is per jet, but written as three passes over the whole tensor it must
// finish every row of layer 0 before layer 1 starts, so nothing downstream can
// begin for its full 7.2 us -- and it sits at the head of the chain, so every
// block waits on it. Split into three kernels passing FOUR rows at a time (the
// width the packed gemm works in), layer 1 starts on rows 0-3 while layer 0 is
// on rows 4-7. The arithmetic is identical: a four-row gemm_pk_biasc call is
// exactly one packed block of the twelve-row call.
#if defined(EMBED_PIPE)
#define EMBED_GRP 4
void embed_mlp(input_window_int16* __restrict jets_in,
               output_stream_int16* __restrict embed_out)
{
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    alignas(16) int16 raw[EMBED_IN_WORDS];
    win_read_v<EMBED_IN_WORDS>(jets_in, raw);
    BIAS_REPC(b0_r, E_DIM, embed_b0);
    for (int g = 0; g < EMBED_ROWS / EMBED_GRP; g++) {
        alignas(32) int16 a[EMBED_GRP * EMBED_IN_PAD];
        zero_v<EMBED_GRP * EMBED_IN_PAD>(a);
        for (int r = 0; r < EMBED_GRP; r++)
            for (int c = 0; c < EMBED_IN; c++)
                a[pk_idx<EMBED_IN_PAD>(r, c)] = raw[(g * EMBED_GRP + r) * EMBED_IN + c];
        alignas(32) int16 h[EMBED_GRP * E_DIM];
        gemm_pk_biasc<EMBED_GRP, EMBED_IN_PAD, E_DIM>(a, embed_W0, h, ACC_SHIFT, b0_r.r);
        layernorm_row(h, EMBED_GRP, E_DIM, embed_ln0_g, embed_ln0_b);
        relu_inplace(h, EMBED_GRP * E_DIM);
        stream_write_v<EMBED_GRP * E_DIM>(embed_out, h);
    }
    aie::set_saturation(sat_save);
}

void embed_mlp1(input_stream_int16* __restrict in, output_stream_int16* __restrict out)
{
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    BIAS_REPC(b1_r, E_DIM, embed_b1);
    for (int g = 0; g < EMBED_ROWS / EMBED_GRP; g++) {
        alignas(32) int16 h[EMBED_GRP * E_DIM], ap[EMBED_GRP * E_DIM];
        for (int i = 0; i < EMBED_GRP * E_DIM; i += 16) aie::store_v(h + i, stream_read16(in));
        pack_local16<EMBED_GRP>(h, ap);
        gemm_pk_biasc<EMBED_GRP, E_DIM, E_DIM>(ap, embed_W1, h, ACC_SHIFT, b1_r.r);
        layernorm_row(h, EMBED_GRP, E_DIM, embed_ln1_g, embed_ln1_b);
        relu_inplace(h, EMBED_GRP * E_DIM);
        stream_write_v<EMBED_GRP * E_DIM>(out, h);
    }
    aie::set_saturation(sat_save);
}

void embed_mlp2(input_stream_int16* __restrict in, output_stream_int16* __restrict out)
{
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    BIAS_REPC(b2_r, E_DIM, embed_b2);
    for (int g = 0; g < EMBED_ROWS / EMBED_GRP; g++) {
        alignas(32) int16 h[EMBED_GRP * E_DIM], ap[EMBED_GRP * E_DIM], o[EMBED_GRP * E_DIM];
        for (int i = 0; i < EMBED_GRP * E_DIM; i += 16) aie::store_v(h + i, stream_read16(in));
        pack_local16<EMBED_GRP>(h, ap);
        gemm_pk_biasc<EMBED_GRP, E_DIM, E_DIM>(ap, embed_W2, o, ACC_SHIFT, b2_r.r);
        stream_write_v<EMBED_GRP * E_DIM>(out, o);
    }
    aie::set_saturation(sat_save);
}
#else
#if defined(CHAIN_STREAM)
void embed_mlp(input_window_int16* __restrict jets_in,
               output_stream_int16* __restrict embed_out)
#else
void embed_mlp(input_window_int16* __restrict jets_in,
               output_window_int16* __restrict embed_out)
#endif
{
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    // layer 0: 5 -> 16. The 64-word window (12 x 5 features, then padding)
    // comes in as 4 vectors; the 60 features are scattered into the packed
    // 12 x 8 layout from local memory, the padding columns stay zero.
    alignas(16) int16 raw[EMBED_IN_WORDS];
    win_read_v<EMBED_IN_WORDS>(jets_in, raw);
    alignas(32) int16 a[EMBED_ROWS * EMBED_IN_PAD];
    zero_v<EMBED_ROWS * EMBED_IN_PAD>(a);
    for (int r = 0; r < EMBED_ROWS; r++)
        for (int c = 0; c < EMBED_IN; c++)
            a[pk_idx<EMBED_IN_PAD>(r, c)] = raw[r * EMBED_IN + c];

    alignas(32) int16 h[EMBED_ROWS * E_DIM];
    BIAS_REPC(b0_r, E_DIM, embed_b0);
    gemm_pk_biasc<EMBED_ROWS, EMBED_IN_PAD, E_DIM>(a, embed_W0, h, ACC_SHIFT, b0_r.r);
    layernorm_row(h, EMBED_ROWS, E_DIM, embed_ln0_g, embed_ln0_b);
    relu_inplace(h, EMBED_ROWS * E_DIM);

    // layer 1: 16 -> 16
    alignas(32) int16 ap[EMBED_ROWS * E_DIM];
    pack_local16<EMBED_ROWS>(h, ap);
    BIAS_REPC(b1_r, E_DIM, embed_b1);
    gemm_pk_biasc<EMBED_ROWS, E_DIM, E_DIM>(ap, embed_W1, h, ACC_SHIFT, b1_r.r);
    layernorm_row(h, EMBED_ROWS, E_DIM, embed_ln1_g, embed_ln1_b);
    relu_inplace(h, EMBED_ROWS * E_DIM);

    // layer 2: 16 -> 16, no norm
    pack_local16<EMBED_ROWS>(h, ap);
    alignas(32) int16 out[EMBED_ROWS * E_DIM];
    BIAS_REPC(b2_r, E_DIM, embed_b2);
    gemm_pk_biasc<EMBED_ROWS, E_DIM, E_DIM>(ap, embed_W2, out, ACC_SHIFT, b2_r.r);

#if defined(CHAIN_STREAM)
    stream_write_v<EMBED_ROWS * E_DIM>(embed_out, out);
#else
    win_write_v<EMBED_ROWS * E_DIM>(embed_out, out);
#endif
    aie::set_saturation(sat_save);
}
#endif  // EMBED_PIPE
#endif // TRANSPOSED
