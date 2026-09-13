// Transposed embedding kernel (see transposed.h): X^T = W2^T relu(LN(W1^T relu(LN(W0^T J^T)))).
#include "transposed.h"

void embed_mlp(input_window_int16* __restrict jets_in,
               output_window_int16* __restrict embed_out)
{
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    static constexpr float EPS_Q2 = 1e-5f * (float)DATA_SCALE * (float)DATA_SCALE;
    alignas(16) int16 raw[EMBED_IN_WORDS];
    win_read_v<EMBED_IN_WORDS>(jets_in, raw);
    // J^T: EMBED_IN_PAD (8) feature rows x 16 jet lanes, padding zero
    alignas(16) int16 JT[EMBED_IN_PAD * 16];
    zero_v<EMBED_IN_PAD * 16>(JT);
    for (int j = 0; j < EMBED_ROWS; j++)
        for (int f = 0; f < EMBED_IN; f++)
            JT[f * 16 + j] = raw[j * EMBED_IN + f];

    PACKED_WT(W0_t, E_DIM, EMBED_IN_PAD, embed_W0);
    PACKED_WT(W1_t, E_DIM, E_DIM, embed_W1);
    PACKED_WT(W2_t, E_DIM, E_DIM, embed_W2);
    alignas(16) int16 HT[16 * 16], GT[16 * 16];
    BIAS_REP(W0_t_b, E_DIM, embed_b0);
    gemm_pk_bias<E_DIM, EMBED_IN_PAD, 16>(W0_t.p, JT, HT, ACC_SHIFT, W0_t_b.r);
    LN_PARAMS(embed_ln0_g_f, embed_ln0_g, embed_ln0_b);
    ln_lanes(HT, embed_ln0_g_f.g, embed_ln0_g_f.b, EPS_Q2);
    relu_rows<E_DIM>(HT);
    BIAS_REP(W1_t_b, E_DIM, embed_b1);
    gemm_pk_bias<E_DIM, E_DIM, 16>(W1_t.p, HT, GT, ACC_SHIFT, W1_t_b.r);
    LN_PARAMS(embed_ln1_g_f, embed_ln1_g, embed_ln1_b);
    ln_lanes(GT, embed_ln1_g_f.g, embed_ln1_g_f.b, EPS_Q2);
    relu_rows<E_DIM>(GT);
    BIAS_REP(W2_t_b, E_DIM, embed_b2);
    gemm_pk_bias<E_DIM, E_DIM, 16>(W2_t.p, GT, HT, ACC_SHIFT, W2_t_b.r);
    alignas(16) int16 Xr[EMBED_ROWS * E_DIM];
    from_T<EMBED_ROWS>(HT, Xr);
    win_write_v<EMBED_ROWS * E_DIM>(embed_out, Xr);
    aie::set_saturation(sat_save);
}
