// Pairwise bias MLP kernels, see pair_kernels.h. Each wrapper in tb/kernels
// defines one of PAIR_STAGE_FEAT, PAIR_STAGE_L0/L1/L2 (+ PAIR_CHAIN) or
// PAIR_STAGE_MERGE (+ PAIR_MERGE name, PAIR_MERGE_A / PAIR_MERGE_B row counts)
// and includes this file, so every tile gets its own symbol.
#include "pair_kernels.h"
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <adf.h>
#include "gemm_utils.h"
#include "win_vec.h"
#include "embed_kernel.h"        // EMBED_IN, EMBED_IN_WORDS: the raw jets window layout
#include "weights/pairwise_weights.h"

// the same data scale and integer layer norm as the embedding
#define PIPE_SCALE DATA_SCALE
#include "layernorm_int.h"

#define _PAIR_FN3(s, k) pair_##s##_c##k
#define _PAIR_FN2(s, k) _PAIR_FN3(s, k)
#define PAIR_L0_FN _PAIR_FN2(l0, PAIR_CHAIN)
#define PAIR_L1_FN _PAIR_FN2(l1, PAIR_CHAIN)
#define PAIR_L2_FN _PAIR_FN2(l2, PAIR_CHAIN)
#define _PAIR_MG3(n) pair_merge_##n
#define _PAIR_MG2(n) _PAIR_MG3(n)
#define PAIR_MERGE_FN _PAIR_MG2(PAIR_MERGE)

static inline void pair_relu(int16* __restrict x, int n)
{
    for (int i = 0; i < n; i += 16) {
        aie::vector<int16, 16> v = aie::load_v<16>(&x[i]);
        aie::store_v(&x[i], aie::max(v, aie::broadcast<int16, 16>(0)));
    }
}

#if defined(PAIR_STAGE_FEAT)
// Exactly the fabric's compute_pairwise: the difference and the two products
// are formed at full width and truncated to Q6.9 (an arithmetic shift is the
// fixed-point cast's floor). Feature 4 is the zero padding column.
void pair_feat(input_window_int16* __restrict jets_in,
               output_stream_int16* __restrict out_a,
               output_stream_int16* __restrict out_b)
{
    alignas(16) int16 raw[EMBED_IN_WORDS];
    win_read_v<EMBED_IN_WORDS>(jets_in, raw);
    alignas(32) int16 rows[PAIR_FEAT_WORDS];
    for (int i = 0; i < PAIR_CHAINS; i++) {
        const int32 eta_i = raw[i * EMBED_IN + 1], c_i = raw[i * EMBED_IN + 2], s_i = raw[i * EMBED_IN + 3];
        for (int j = 0; j < PAIR_ROWS; j++) {
            const int32 eta_j = raw[j * EMBED_IN + 1], c_j = raw[j * EMBED_IN + 2], s_j = raw[j * EMBED_IN + 3];
            rows[j * PAIR_K + 0] = (int16)(eta_i - eta_j);
            rows[j * PAIR_K + 1] = (int16)((c_i * c_j + s_i * s_j) >> DATA_FRAC_BITS);
            rows[j * PAIR_K + 2] = (int16)((s_i * c_j - c_i * s_j) >> DATA_FRAC_BITS);
            rows[j * PAIR_K + 3] = 0;
        }
        stream_write_v<PAIR_FEAT_WORDS>(i < PAIR_PER_OUT ? out_a : out_b, rows);
    }
}
#endif

#if defined(PAIR_STAGE_L0)
#if defined(PAIR_L0_WINDOW)
// PAIR_L0_WINDOW: every chain's first tile reads the raw jets window itself and
// forms only its own 12 feature rows, so the 12 chains start together instead
// of waiting in turn behind one feature kernel. Same arithmetic as pair_feat.
void PAIR_L0_FN(input_window_int16* __restrict jets_in, output_stream_int16* __restrict out)
{
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    alignas(16) int16 raw[EMBED_IN_WORDS];
    win_read_v<EMBED_IN_WORDS>(jets_in, raw);
    alignas(32) int16 rows[PAIR_FEAT_WORDS];
    {
        constexpr int i = PAIR_CHAIN;
        const int32 eta_i = raw[i * EMBED_IN + 1], c_i = raw[i * EMBED_IN + 2], s_i = raw[i * EMBED_IN + 3];
        for (int j = 0; j < PAIR_ROWS; j++) {
            const int32 eta_j = raw[j * EMBED_IN + 1], c_j = raw[j * EMBED_IN + 2], s_j = raw[j * EMBED_IN + 3];
            rows[j * PAIR_K + 0] = (int16)(eta_i - eta_j);
            rows[j * PAIR_K + 1] = (int16)((c_i * c_j + s_i * s_j) >> DATA_FRAC_BITS);
            rows[j * PAIR_K + 2] = (int16)((s_i * c_j - c_i * s_j) >> DATA_FRAC_BITS);
            rows[j * PAIR_K + 3] = 0;
        }
    }
#else
// layer 0 for chain PAIR_CHAIN: its 12 rows come on the shared stream with the
// other five chains' rows; the rest are read and dropped
void PAIR_L0_FN(input_stream_int16* __restrict in, output_stream_int16* __restrict out)
{
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    constexpr int mine = PAIR_CHAIN % PAIR_PER_OUT;
    alignas(32) int16 rows[PAIR_FEAT_WORDS];
    for (int c = 0; c < PAIR_PER_OUT; c++)
        for (int w = 0; w < PAIR_FEAT_WORDS; w += 16) {
            const v16_t v = stream_read16(in);
            if (c == mine) aie::store_v(rows + w, v);
        }
#endif
    BIAS_REPC(b0_r, E_DIM, pair_b0);
    for (int g = 0; g < PAIR_ROWS / 4; g++) {
        // K == 4: the packed layout is row-major, so the four rows pass straight in
        alignas(32) int16 h[4 * E_DIM];
        gemm_pk_biasc<4, PAIR_K, E_DIM>(rows + g * 4 * PAIR_K, pair_W0, h, ACC_SHIFT, b0_r.r);
        layernorm_row(h, 4, E_DIM, pair_ln0_g, pair_ln0_b);
        pair_relu(h, 4 * E_DIM);
        stream_write_v<4 * E_DIM>(out, h);
    }
    aie::set_saturation(sat_save);
}
#endif

#if defined(PAIR_STAGE_L1)
void PAIR_L1_FN(input_stream_int16* __restrict in, output_stream_int16* __restrict out)
{
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    BIAS_REPC(b1_r, E_DIM, pair_b1);
    for (int g = 0; g < PAIR_ROWS / 4; g++) {
        alignas(32) int16 h[4 * E_DIM], ap[4 * E_DIM];
        for (int i = 0; i < 4 * E_DIM; i += 16) aie::store_v(h + i, stream_read16(in));
        pack_local16<4>(h, ap);
        gemm_pk_biasc<4, E_DIM, E_DIM>(ap, pair_W1, h, ACC_SHIFT, b1_r.r);
        layernorm_row(h, 4, E_DIM, pair_ln1_g, pair_ln1_b);
        pair_relu(h, 4 * E_DIM);
        stream_write_v<4 * E_DIM>(out, h);
    }
    aie::set_saturation(sat_save);
}
#endif

#if defined(PAIR_STAGE_L2)
// layer 2, then the 16 -> 1 output layer (as a 16 -> 8 gemm whose other seven
// columns are zero), then the fabric's scale conversion; emits one 16-lane row
void PAIR_L2_FN(input_stream_int16* __restrict in, output_stream_int16* __restrict out)
{
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    BIAS_REPC(b2_r, E_DIM, pair_b2);
    BIAS_REPC(b3_r, 8, pair_b3);
    alignas(32) int16 row[E_DIM];
    zero_v<E_DIM>(row);
    for (int g = 0; g < PAIR_ROWS / 4; g++) {
        alignas(32) int16 h[4 * E_DIM], ap[4 * E_DIM], o[4 * 8];
        for (int i = 0; i < 4 * E_DIM; i += 16) aie::store_v(h + i, stream_read16(in));
        pack_local16<4>(h, ap);
        gemm_pk_biasc<4, E_DIM, E_DIM>(ap, pair_W2, h, ACC_SHIFT, b2_r.r);
        layernorm_row(h, 4, E_DIM, pair_ln2_g, pair_ln2_b);
        pair_relu(h, 4 * E_DIM);
        pack_local16<4>(h, ap);
        gemm_pk_biasc<4, E_DIM, 8>(ap, pair_W3, o, ACC_SHIFT, b3_r.r);
        for (int r = 0; r < 4; r++) {
            const int v = o[r * 8];                      // Q6.9
            row[g * 4 + r] = (int16)((v >> 4) << 2);     // -> Q10.5 by truncation -> Q8.7, as wij_send
        }
    }
    stream_write_v<E_DIM>(out, row);
    aie::set_saturation(sat_save);
}
#endif

#if defined(PAIR_STAGE_MERGE)
void PAIR_MERGE_FN(input_stream_int16* __restrict a_in,
                   input_stream_int16* __restrict b_in,
                   output_stream_int16* __restrict out)
{
    for (int r = 0; r < PAIR_MERGE_A; r++) stream_write16(out, stream_read16(a_in));
    for (int r = 0; r < PAIR_MERGE_B; r++) stream_write16(out, stream_read16(b_in));
}
#endif
