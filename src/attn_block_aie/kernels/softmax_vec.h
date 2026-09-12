// Vector float softmax for the 16-key head post kernels (object attention).
//
// The integer softmax (int_softmax_packed) runs 156 scalar exponentials per
// event with a branch, a table lookup and a variable shift each, and was the
// slowest kernel of the block once the window traffic was vector (10.1 us).
// AIE1 has an 8-lane fp32 vector unit (no vector exp), so per row:
//   d = max - x  (int32 lanes, >= 0)                 t = d * log2e / score_scale
//   k = round(t), f = t - k in [-0.5, 0.5]           2^-f by a degree-4 polynomial
//   e = p(f) * 2^-k by subtracting k from the float exponent bits, k clamped
//   sum over the valid lanes (vector reduce), one vector Newton reciprocal for all rows,
//   w = e * out_scale / sum -> int32 -> saturating int16, packed 4x4 blocks.
// Invalid lanes (keys >= N_COLS) are pushed out of the max by a lane bias and
// multiplied by a 0/1 lane weight, so no mask type is needed.
// Accuracy: the polynomial is exact to ~4e-5 relative on |f| <= 0.5 (the
// integer LUT was ~0.5%), so this is closer to the PyTorch softmax, not
// bit-identical to the integer version.
#ifndef SOFTMAX_VEC_H
#define SOFTMAX_VEC_H

#include <aie_api/aie.hpp>
#include "win_vec.h"

template <int N_ROWS, int N_COLS, int N_PAD>
static void vec_softmax_packed(const int16* __restrict scores, int16* __restrict out,
                               float score_scale, float out_scale)
{
    static_assert(N_PAD == 16, "vector softmax handles 16-lane rows");
    static_assert(N_ROWS % 4 == 0, "rows are packed four at a time");
    typedef aie::vector<float, 8>  v8f;
    typedef aie::vector<int32, 8>  v8i;
    typedef aie::vector<int32, 16> v16i;

    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);

    // lane constants: bias pushes invalid keys out of the max, weight zeroes them
    alignas(16) int32 lane_bias[16];
    alignas(16) float lane_w[16];
    for (int c = 0; c < 16; c++) { lane_bias[c] = (c < N_COLS) ? 0 : -(1 << 20); lane_w[c] = (c < N_COLS) ? 1.0f : 0.0f; }
    const v16i bias = aie::load_v<16>(lane_bias);
    const v8f  w0 = aie::load_v<8>(lane_w), w1 = aie::load_v<8>(lane_w + 8);

    const float c_t = 1.4426950408889634f / score_scale;   // log2 e at score scale
    const v8f a1 = aie::broadcast<float, 8>(-0.69314718f);
    const v8f a2 = aie::broadcast<float, 8>( 0.24022651f);
    const v8f a3 = aie::broadcast<float, 8>(-0.05550411f);
    const v8f a4 = aie::broadcast<float, 8>( 0.00961813f);
    const v8f one = aie::broadcast<float, 8>(1.0f);
    const v8i kmax = aie::broadcast<int32, 8>(60);

    // float vector products come back as FP accumulators: bring them back to vectors
    auto fmul = [](const v8f& a, const v8f& b) -> v8f { return aie::mul(a, b).template to_vector<float>(); };

    auto exp_half = [&](const v8i& d, const v8f& w) -> v8f {
        const v8f t  = fmul(aie::to_float(d), aie::broadcast<float, 8>(c_t));
        const v8i k  = aie::to_fixed<int32>(t);                 // round to nearest
        const v8f f  = aie::sub(t, aie::to_float(k));          // |f| <= ~0.5
        v8f p = aie::add(a3, fmul(f, a4));
        p = aie::add(a2, fmul(f, p));
        p = aie::add(a1, fmul(f, p));
        p = aie::add(one, fmul(f, p));                           // 2^-f
        const v8i kc = aie::min(k, kmax);
        const v8i eb = aie::sub(p.template cast_to<int32>(), aie::upshift(kc, 23));
        return fmul(eb.template cast_to<float>(), w);             // * 2^-k, invalid lanes 0
    };

    // pass 1: exponentials per row (kept in local memory) and the row sums
    alignas(32) float ev[N_ROWS * 16];
    alignas(16) float sums[16];
    for (int r = 0; r < N_ROWS; r++) {
        const v16_t x = aie::load_v<16>(scores + r * N_PAD);
        const v16i x32 = aie::from_vector<acc48>(x).template to_vector<int32>(0);
        const int32 mx = aie::reduce_max(aie::add(x32, bias));
        const v16i d32 = aie::sub(aie::broadcast<int32, 16>(mx), x32);
        const v8f e0 = exp_half(d32.template extract<8>(0), w0);
        const v8f e1 = exp_half(d32.template extract<8>(1), w1);
        aie::store_v(ev + r * 16, e0);
        aie::store_v(ev + r * 16 + 8, e1);
        sums[r] = aie::reduce_add(aie::add(e0, e1));
    }
    for (int r = N_ROWS; r < 16; r++) sums[r] = 1.0f;

    // one vector reciprocal for all rows: bit-trick seed, three Newton steps
    // (y = y (2 - s y)), then the output scale folded in. Replaces 12 scalar
    // float divides (softfloat, ~150 cycles each).
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

    // pass 2: normalise, saturate to int16, pack four rows at a time
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

#endif // SOFTMAX_VEC_H
