// Integer layer norm shared by the post-attention kernels and the jet
// embedding kernel. Define PIPE_SCALE (the 2^F of the fixed-point format)
// before including this header.
#ifndef LAYERNORM_INT_H
#define LAYERNORM_INT_H

#include <aie_api/aie.hpp>

// ---------------------------------------------------------------------------
// Integer layer norm, vector int16 with a 32-bit reciprocal-square-root
// lookup. AIE1 has no scalar FPU and no 64-bit ALU: the float version was
// ~2300 cycles/row of softfloat calls, and a first integer version (64-bit
// restoring sqrt + 64-bit divide, commit 49dcb21) was ~4100 cycles/row of
// emulated 64-bit ops. This one uses only 32-bit scalar ops and 16-bit
// vector multiplies.
//
// Math (x, gamma, beta all at PIPE_SCALE = 2^F; n_cols = 16):
//   d    = 16*x - sum(x)                       exact, = 16*(x - mean)
//   V    = sum d^2 + 256*EPS_V                 EPS_V = eps*16*2^(2F)
//   y    = 4*g*d / sqrt(V) + b                 (2^F factors cancel)
// The mean is never rounded (a half-LSB mean error is a ~0.2 error in every
// output of a low-spread row).
//
// Fixed-point plan, per row:
//   dn   = d * 2^-kd, kd so that |dn| <= 2^14, kd may be negative (an exact
//          upshift for low-spread rows, so S below keeps full precision;
//          kd >= KD_MIN keeps the eps term inside int32)
//   S    = sum (dn^2 >> 2)  <= 2^30               (vector mul, reduce)
//   W    = S + 64*EPS_V/2^(2kd)  ->  V = 4*2^(2kd)*W   (EPS_V = eps*16*2^(2F), kept with 2 fraction bits)
//   Wn   = W << e, e even, Wn in [2^30, 2^32)  ->  M = Wn/2^32 in [0.25,1)
//   R    = 1/sqrt(M) from LN_RSQRT_LUT (385 entries, linear interp), Rq = R*2^14
//   1/sqrt(V) = R * 2^(e/2 - kd - 17)
//   dn2  = (dn*Rq) >> sd, sd so that |dn2| <= 2^15  (vector, rounded)
//   y    = (g*dn2) >> (29 - sd - e/2) + b           (vector, rounded, saturated)
// Every rounding is a vector srs with round-to-nearest-even; the rounding and
// saturation modes are set here and restored, so gemm_pk is unaffected.
// Precision: dn and dn2 keep 14-15 bits of the row's largest |d|, R is good
// to 1e-5; the unit test (scratchpad lntest/t2.cpp) shows max 0.5 LSB vs the
// float reference over 20k random rows, including near-constant rows.
// ---------------------------------------------------------------------------
#include "ln_rsqrt_lut.h"

static constexpr int bitlen32_ce(uint32 v) { return v ? 1 + bitlen32_ce(v >> 1) : 0; }

static inline int bitlen32(uint32 v)
{
    int n = 0;
    if (v >> 16) { n += 16; v >>= 16; }
    if (v >> 8)  { n += 8;  v >>= 8;  }
    if (v >> 4)  { n += 4;  v >>= 4;  }
    if (v >> 2)  { n += 2;  v >>= 2;  }
    if (v >> 1)  { n += 1;  v >>= 1;  }
    return n + (int)v;
}

static void layernorm_row(int16* __restrict x, int n_rows, int n_cols,
                          const int16* __restrict gamma,
                          const int16* __restrict beta)
{
    // n_cols is always E_DIM = 16 here
    // eps term with 2 extra fraction bits: rounding EPS_V itself to an integer
    // is a 0.14% error at F=9, which is 2 LSB on a high-gain, low-spread row
    constexpr int32 EPS_W4 = (int32)(256.0f * 1e-5f * 16.0f * PIPE_SCALE * PIPE_SCALE + 0.5f); // 4*64*EPS_V
    constexpr int KD_MIN = -((32 - bitlen32_ce((uint32)EPS_W4)) / 2);         // EPS_W4 << (-2*KD_MIN-2) < 2^30

    const aie::rounding_mode   rnd_save = aie::swap_rounding(aie::rounding_mode::conv_even);
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);

    const aie::vector<int16, 16> gv = aie::load_v<16>(gamma);
    const aie::vector<int32, 16> bv = aie::from_vector<acc48>(aie::load_v<16>(beta)).to_vector<int32>(0); // int16 -> int32 (unpack() is int8-only in the 2022.2 API)

    for (int r = 0; r < n_rows; r++) {
        int16* row = x + r * n_cols;
        const aie::vector<int32, 16> x32 = aie::from_vector<acc48>(aie::load_v<16>(row)).to_vector<int32>(0);
        const int32 sum = aie::reduce_add(x32);
        const aie::vector<int32, 16> d32 =
            aie::sub(aie::upshift(x32, 4), aie::broadcast<int32, 16>(sum));   // |d| < 2^20
        const int32 m = aie::reduce_max(aie::abs(d32));

        int kd = bitlen32((uint32)m) - 14; if (kd < KD_MIN) kd = KD_MIN;     // |dn| <= 2^14
        const int up = kd < 0 ? -kd : 0, down = kd > 0 ? kd : 0;
        const aie::vector<int16, 16> dn = aie::from_vector<acc80>(d32, up).to_vector<int16>(down);
        const int32 S = aie::reduce_add(aie::mul(dn, dn).to_vector<int32>(2)); // <= 2^30
        const int32 W = S + (kd >= 0 ? (EPS_W4 >> (2 * kd + 2)) : (EPS_W4 << (2 * up - 2))); // < 2^31

        const int e = (32 - bitlen32((uint32)W)) & ~1;                        // even
        const uint32 Wn = (uint32)W << e;                                     // [2^30, 2^32)
        const int idx = (int)(Wn >> 23) - 128;                                // [0, 384)
        const int32 frac = (int32)((Wn >> 7) & 0xFFFF);
        const int32 l0 = LN_RSQRT_LUT[idx], l1 = LN_RSQRT_LUT[idx + 1];
        const int32 R16 = l0 + (((l1 - l0) * frac) >> 16);                    // Q16, (2^16, 2^17]
        int32 Rq = (R16 + 2) >> 2; if (Rq > 32767) Rq = 32767;                // Q14

        const int32 mq = kd >= 0 ? (m >> kd) : (m << up);                     // ~max |dn|, <= 2^14
        int sd = bitlen32((uint32)(mq * Rq)) - 15; if (sd < 0) sd = 0;        // |dn2| <= 2^15
        const aie::vector<int16, 16> dn2 = aie::mul(dn, (int16)Rq).to_vector<int16>(sd);

        int sy = 29 - sd - (e >> 1); if (sy < 0) sy = 0;                     // >= 11 in practice
        aie::vector<int32, 16> y32 = aie::mul(gv, dn2).to_vector<int32>(sy);
        y32 = aie::add(y32, bv);
        aie::store_v(row, aie::from_vector<acc80>(y32).to_vector<int16>(0));  // saturate
    }

    aie::set_rounding(rnd_save);
    aie::set_saturation(sat_save);
}

#endif // LAYERNORM_INT_H
