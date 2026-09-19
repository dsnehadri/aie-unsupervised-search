// Micro-benchmarks for the array: 12 rows through the layer norm (current and
// four-row interleaved) and the 16-key softmax (float polynomial and table).
// Each reads the object block's 13-row x window and writes 12 rows.
#include "../attn_block_aie/kernels/attn_aie_types.h"
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <adf.h>
#include "../attn_block_aie/kernels/gemm_utils.h"
#include "../attn_block_aie/kernels/win_vec.h"
#include "../attn_block_aie/kernels/weights/embed_weights.h"
#define PIPE_SCALE DATA_SCALE
#include "../attn_block_aie/kernels/layernorm_int.h"
#include "../attn_block_aie/kernels/softmax_vec.h"
#include "../attn_block_aie/kernels/softmax_lut.h"

static inline void rd12(input_window_int16* __restrict in, int16* __restrict x) { win_read_v<12 * 16>(in, x); }
static inline void wr12(output_window_int16* __restrict out, const int16* __restrict x) { win_write_v<12 * 16>(out, x); }

#if defined(BENCH_LN)
void bench_ln(input_window_int16* __restrict in, output_window_int16* __restrict out)
{
    alignas(32) int16 x[12 * 16];
    rd12(in, x);
    layernorm_row(x, 12, 16, embed_ln0_g, embed_ln0_b);
    wr12(out, x);
}
#endif
#if defined(BENCH_LN_IL)
void bench_ln_il(input_window_int16* __restrict in, output_window_int16* __restrict out)
{
    alignas(32) int16 x[12 * 16];
    rd12(in, x);
    layernorm_row(x, 12, 16, embed_ln0_g, embed_ln0_b);   // built with LN_IL4: four-row batches
    wr12(out, x);
}
#endif
#if defined(BENCH_SM_VEC)
void bench_sm_vec(input_window_int16* __restrict in, output_window_int16* __restrict out)
{
    alignas(32) int16 x[12 * 16], p[12 * 16];
    rd12(in, x);
    vec_softmax_packed<12, N_KV, N_KV_PAD>(x, p, (float)SCORE_SCALE, (float)DATA_SCALE);
    wr12(out, p);
}
#endif
#if defined(BENCH_SM_LUT)
void bench_sm_lut(input_window_int16* __restrict in, output_window_int16* __restrict out)
{
    alignas(32) int16 x[12 * 16], p[12 * 16];
    rd12(in, x);
    lut_softmax_packed<12, N_KV, N_KV_PAD>(x, p, (float)DATA_SCALE);
    wr12(out, p);
}
#endif
