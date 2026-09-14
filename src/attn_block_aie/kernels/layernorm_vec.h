// Float-vector layer norm (LN_VEC), a drop-in for layernorm_row.
//
// The integer layer norm (layernorm_int.h) is exact to half an LSB but it is a
// long DEPENDENT chain: three vector reductions with a dozen scalar steps
// between them (bit lengths, variable shifts, a 385-entry rsqrt table with a
// linear interpolation), about 135 cycles per row and almost all of it
// latency, not work. With one row in flight at a time -- which is what the
// streaming post pipeline does -- there is nothing to hide it behind, and it
// was 56% of the slowest kernel in the block.
//
// AIE1 has an 8-lane fp32 vector unit, so a row of E_DIM = 16 is two vectors:
//   m    = sum(x) / 16                      (one reduce, no scalar arithmetic)
//   d    = x - m
//   v    = sum(d*d) / 16 + eps
//   r    = 1/sqrt(v)   bit-trick seed 0x5f3759df + two Newton steps
//   y    = gamma * d * r + beta             (round to nearest, saturating)
// x, gamma and beta are all at PIPE_SCALE = 2^F. d and sqrt(v) carry the same
// 2^F, so d*r is unitless and gamma*d*r + beta lands back at 2^F with no
// rescaling; eps is scaled by 2^(2F) to match v.
//
// Accuracy: fp32 keeps 24 bits, the row sums reach 2^34 at most, and the two
// Newton steps put r within ~1e-6 relative. Against the integer version the
// outputs differ by at most one LSB, so this is NOT bit-identical -- the
// blocks are checked against the golden tensors with the usual 0.5 tolerance
// and the hardware scores are checked by AUC. SOFTMAX_INT's sibling switch is
// LN_INT: define it to go back to the integer path.
#ifndef LAYERNORM_VEC_H
#define LAYERNORM_VEC_H

#include <aie_api/aie.hpp>

static void layernorm_row_vf(int16* __restrict x, int n_rows, int n_cols,
                             const int16* __restrict gamma,
                             const int16* __restrict beta)
{
    typedef aie::vector<float, 8> v8f;
    typedef aie::vector<int32, 8> v8i;

    const aie::rounding_mode   rnd_save = aie::swap_rounding(aie::rounding_mode::conv_even);
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);

    // float vector products come back as FP accumulators
    auto fmul = [](const v8f& a, const v8f& b) -> v8f { return aie::mul(a, b).template to_vector<float>(); };
    auto i32 = [](const aie::vector<int16, 16>& v) {
        return aie::from_vector<acc48>(v).template to_vector<int32>(0);
    };

    const aie::vector<int32, 16> g32 = i32(aie::load_v<16>(gamma));
    const aie::vector<int32, 16> b32 = i32(aie::load_v<16>(beta));
    const v8f g0 = aie::to_float(g32.template extract<8>(0)), g1 = aie::to_float(g32.template extract<8>(1));
    const v8f bf0 = aie::to_float(b32.template extract<8>(0)), bf1 = aie::to_float(b32.template extract<8>(1));

    const v8f inv16 = aie::broadcast<float, 8>(1.0f / 16.0f);
    const v8f epsv  = aie::broadcast<float, 8>(1e-5f * (float)PIPE_SCALE * (float)PIPE_SCALE);
    const v8f half  = aie::broadcast<float, 8>(0.5f);
    const v8f onef  = aie::broadcast<float, 8>(1.5f);
    const v8i magic = aie::broadcast<int32, 8>(0x5f3759df);

    for (int r = 0; r < n_rows; r++) {
        int16* __restrict row = x + r * n_cols;
        const aie::vector<int32, 16> x32 = i32(aie::load_v<16>(row));
        const v8f f0 = aie::to_float(x32.template extract<8>(0));
        const v8f f1 = aie::to_float(x32.template extract<8>(1));

        const v8f mv = fmul(aie::broadcast<float, 8>(aie::reduce_add(aie::add(f0, f1))), inv16);
        const v8f d0 = aie::sub(f0, mv), d1 = aie::sub(f1, mv);

        const v8f sq = aie::add(fmul(d0, d0), fmul(d1, d1));
        const v8f vv = aie::add(fmul(aie::broadcast<float, 8>(aie::reduce_add(sq)), inv16), epsv);

        // 1/sqrt(vv): magic seed, then y = y * (1.5 - 0.5 * vv * y * y) twice
        v8f y = aie::sub(magic, aie::downshift(vv.template cast_to<int32>(), 1)).template cast_to<float>();
        const v8f hv = fmul(vv, half);
        y = fmul(y, aie::sub(onef, fmul(hv, fmul(y, y))));
        y = fmul(y, aie::sub(onef, fmul(hv, fmul(y, y))));

        const v8f o0 = aie::add(fmul(fmul(d0, y), g0), bf0);
        const v8f o1 = aie::add(fmul(fmul(d1, y), g1), bf1);
        const aie::vector<int32, 16> oi = aie::concat(aie::to_fixed<int32>(o0), aie::to_fixed<int32>(o1));
        aie::store_v(row, aie::from_vector<acc80>(oi).to_vector<int16>(0));   // saturating
    }

    aie::set_rounding(rnd_save);
    aie::set_saturation(sat_save);
}

#endif // LAYERNORM_VEC_H
