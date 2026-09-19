// Table softmax for the 16-key head post kernels (SOFTMAX_LUT).
//
// Same structure as vec_softmax_packed (softmax_vec.h): pass 1 forms the
// exponentials and row sums, one vector reciprocal serves all rows, pass 2
// normalises and packs. Only the exponential differs: instead of a degree-4
// polynomial in the 8-lane float unit, e = EXP_NEG_Q15[d] with d = max - x at
// the Q8.7 score scale (one 16-bit table load per element, 2.6 KB). The table
// holds exp rounded to 2^-15, about the polynomial's accuracy, but the
// rounding differs, so the outputs are not bit-identical to the float version.
#ifndef SOFTMAX_LUT_H
#define SOFTMAX_LUT_H

#include <aie_api/aie.hpp>
#include "win_vec.h"
#include "exp_neg_q15_lut.h"

template <int N_ROWS, int N_COLS, int N_PAD>
static void lut_softmax_packed(const int16* __restrict scores, int16* __restrict out, float out_scale)
{
    static_assert(N_PAD == 16, "table softmax handles 16-lane rows");
    static_assert(N_ROWS % 4 == 0, "rows are packed four at a time");
    typedef aie::vector<float, 8>  v8f;
    typedef aie::vector<int32, 8>  v8i;
    typedef aie::vector<int32, 16> v16i;

    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);

    alignas(16) int32 lane_bias[16];
    alignas(16) float lane_w[16];
    for (int c = 0; c < 16; c++) { lane_bias[c] = (c < N_COLS) ? 0 : -(1 << 20); lane_w[c] = (c < N_COLS) ? 1.0f : 0.0f; }
    const v16i bias = aie::load_v<16>(lane_bias);
    const v8f  w0 = aie::load_v<8>(lane_w), w1 = aie::load_v<8>(lane_w + 8);
    const v16i zero16 = aie::zeros<int32, 16>();
    const v16i dmax = aie::broadcast<int32, 16>(EXP_LUT_N - 1);
    auto fmul = [](const v8f& a, const v8f& b) -> v8f { return aie::mul(a, b).template to_vector<float>(); };

    // pass 1: table exponentials per row and the row sums
    alignas(32) float ev[N_ROWS * 16];
    alignas(16) float sums[16];
    alignas(32) int32 idx[16];
    alignas(32) int32 e32[16];
    for (int r = 0; r < N_ROWS; r++) {
        const v16_t x = aie::load_v<16>(scores + r * N_PAD);
        const v16i x32 = aie::from_vector<acc48>(x).template to_vector<int32>(0);
        const int32 mx = aie::reduce_max(aie::add(x32, bias));
        const v16i d32 = aie::min(aie::max(aie::sub(aie::broadcast<int32, 16>(mx), x32), zero16), dmax);
        aie::store_v(idx, d32);
        for (int c = 0; c < 16; c++) e32[c] = EXP_NEG_Q15[idx[c]];
        const v8f e0 = fmul(aie::to_float(aie::load_v<8>(e32)), w0);
        const v8f e1 = fmul(aie::to_float(aie::load_v<8>(e32 + 8)), w1);
        aie::store_v(ev + r * 16, e0);
        aie::store_v(ev + r * 16 + 8, e1);
        sums[r] = aie::reduce_add(aie::add(e0, e1));
    }
    for (int r = N_ROWS; r < 16; r++) sums[r] = 1.0f;

    alignas(16) float invs[16];
    {
        const v8f two = aie::broadcast<float, 8>(2.0f);
        const v8i magic = aie::broadcast<int32, 8>(0x7EF311C7);
        const v8f osc = aie::broadcast<float, 8>(out_scale);
        for (int h = 0; h < 2; h++) {
            const v8f sv = aie::load_v<8>(sums + 8 * h);
            v8f y = aie::sub(magic, sv.template cast_to<int32>()).template cast_to<float>();
            for (int it = 0; it < 3; it++)
                y = fmul(y, aie::sub(two, fmul(sv, y)));
            aie::store_v(invs + 8 * h, fmul(y, osc));
        }
    }

    v16_t rows[4];
    for (int r = 0; r < N_ROWS; r++) {
        const v8f inv = aie::broadcast<float, 8>(invs[r]);
        const v8i q0 = aie::to_fixed<int32>(fmul(aie::load_v<8>(ev + r * 16), inv));
        const v8i q1 = aie::to_fixed<int32>(fmul(aie::load_v<8>(ev + r * 16 + 8), inv));
        rows[r & 3] = aie::from_vector<acc80>(aie::concat(q0, q1)).template to_vector<int16>(0);
        if ((r & 3) == 3)
            pack_rows4(rows[0], rows[1], rows[2], rows[3], out + (r / 4) * 64);
    }

    aie::set_saturation(sat_save);
}

#endif // SOFTMAX_LUT_H
