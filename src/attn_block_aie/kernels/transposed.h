// Transposed dataflow for the int16 attention kernels (TRANSPOSED build).
//
// Every activation is kept as X^T: 16 rows (features) x 16 lanes (jets or
// candidates; 12 or 3 valid, the rest zero). Consequences:
//   * a linear layer Y = X W is Y^T = W^T X^T: the constant W^T is packed once
//     per tile as the mmul A operand and X^T is the B operand as it is, so the
//     activations are never packed or unpacked;
//   * layer norm over the features and softmax over the keys are per-lane
//     operations over the rows: vector adds/max/muls with no reductions, no
//     lane masks and no per-row scalar chains (the per-row version cost ~135
//     cycles a row because it was latency-bound, ~10 us per block);
//   * the four heads' outputs (4 x 16 each) stack as rows into the 16 x 16
//     concat, so post_a_proj does no interleaving.
// The PL interface is unchanged: row-major tensors cross the block edges and
// are transposed on the tiles (scalar, ~0.5 us; a vector version can follow).
// Layer norm and softmax run in fp32 (8-lane vectors), as the first deployed
// float kernels did, so the outputs are not bit-identical to the integer
// versions; the PyTorch golden check bounds the difference.
#ifndef TRANSPOSED_H
#define TRANSPOSED_H

#include <aie_api/aie.hpp>
#include <adf.h>
#include "gemm_utils.h"
#include "win_vec.h"

typedef aie::vector<int16, 16> v16s;
typedef aie::vector<int32, 16> v16i;
typedef aie::vector<int32, 8>  v8i;
typedef aie::vector<float, 8>  v8f;

// W^T as a packed mmul A operand (M_OUT x K_IN), built once per tile from the
// row-major K_IN x M_OUT weight the headers ship: W[k * M_OUT + m].
template <int M_OUT, int K_IN>
struct PackedWT {
    alignas(16) int16 p[M_OUT * K_IN];
    bool ready = false;
    inline void build(const int16* __restrict W)
    {
        for (int m = 0; m < M_OUT; m++)
            for (int k = 0; k < K_IN; k++)
                p[pk_idx<K_IN>(m, k)] = W[k * M_OUT + m];
        ready = true;
    }
};
#define PACKED_WT(name, M, K, W) static PackedWT<M, K> name; if (!name.ready) name.build(W)

// row-major R x 16 -> 16 x 16 transposed, lanes >= R zero
template <int R>
static inline void to_T(const int16* __restrict A, int16* __restrict T)
{
    for (int f = 0; f < 16; f++)
        for (int j = 0; j < 16; j++)
            T[f * 16 + j] = (j < R) ? A[j * 16 + f] : (int16)0;
}
// 16 x 16 transposed -> row-major R x 16
template <int R>
static inline void from_T(const int16* __restrict T, int16* __restrict A)
{
    for (int j = 0; j < R; j++)
        for (int f = 0; f < 16; f++)
            A[j * 16 + f] = T[f * 16 + j];
}

static inline v8f fmul8(const v8f& a, const v8f& b) { return aie::mul(a, b).template to_vector<float>(); }

// 1/sqrt(v) and 1/v for 8 lanes: bit-trick seed + Newton (no vector rsqrt/div on AIE1)
static inline v8f rsqrt8(const v8f& v)
{
    const v8f half = aie::broadcast<float, 8>(0.5f), th = aie::broadcast<float, 8>(1.5f);
    v8f y = aie::sub(aie::broadcast<int32, 8>(0x5f3759df), aie::downshift(v.template cast_to<int32>(), 1)).template cast_to<float>();
    const v8f hv = fmul8(half, v);
    for (int it = 0; it < 3; it++) y = fmul8(y, aie::sub(th, fmul8(hv, fmul8(y, y))));
    return y;
}
static inline v8f recip8(const v8f& v)
{
    const v8f two = aie::broadcast<float, 8>(2.0f);
    v8f y = aie::sub(aie::broadcast<int32, 8>(0x7EF311C7), v.template cast_to<int32>()).template cast_to<float>();
    for (int it = 0; it < 3; it++) y = fmul8(y, aie::sub(two, fmul8(v, y)));
    return y;
}

// int16 row -> two float halves, and back (saturating; callers set saturate mode)
static inline void row_to_f(const int16* __restrict row, v8f& f0, v8f& f1)
{
    const v16i x32 = aie::from_vector<acc48>(aie::load_v<16>(row)).template to_vector<int32>(0);
    f0 = aie::to_float(x32.template extract<8>(0));
    f1 = aie::to_float(x32.template extract<8>(1));
}
static inline v16s f_to_row(const v8f& f0, const v8f& f1)
{
    const v8i q0 = aie::to_fixed<int32>(f0), q1 = aie::to_fixed<int32>(f1);
    return aie::from_vector<acc80>(aie::concat(q0, q1)).template to_vector<int16>(0);
}

// Layer norm over the 16 rows of XT, per lane. x, gamma, beta at the same
// fixed-point scale S; y = (x - mean) / sqrt(var + eps) * gamma + beta, eps in
// x^2 units (eps * S^2).
static inline void ln_lanes(int16* __restrict XT, const int16* __restrict gamma,
                            const int16* __restrict beta, float eps_q2)
{
    alignas(32) float xf[16 * 16];
    v8f s0 = aie::zeros<float, 8>(), s1 = aie::zeros<float, 8>();
    for (int r = 0; r < 16; r++) {
        v8f f0, f1; row_to_f(XT + r * 16, f0, f1);
        aie::store_v(xf + r * 16, f0); aie::store_v(xf + r * 16 + 8, f1);
        s0 = aie::add(s0, f0); s1 = aie::add(s1, f1);
    }
    const v8f inv16 = aie::broadcast<float, 8>(1.0f / 16.0f);
    const v8f m0 = fmul8(s0, inv16), m1 = fmul8(s1, inv16);
    v8f v0 = aie::zeros<float, 8>(), v1 = aie::zeros<float, 8>();
    for (int r = 0; r < 16; r++) {
        const v8f d0 = aie::sub(aie::load_v<8>(xf + r * 16), m0);
        const v8f d1 = aie::sub(aie::load_v<8>(xf + r * 16 + 8), m1);
        v0 = aie::add(v0, fmul8(d0, d0)); v1 = aie::add(v1, fmul8(d1, d1));
    }
    const v8f epsv = aie::broadcast<float, 8>(eps_q2);
    const v8f r0 = rsqrt8(aie::add(fmul8(v0, inv16), epsv));
    const v8f r1 = rsqrt8(aie::add(fmul8(v1, inv16), epsv));
    for (int r = 0; r < 16; r++) {
        const v8f g = aie::broadcast<float, 8>((float)gamma[r]);
        const v8f b = aie::broadcast<float, 8>((float)beta[r]);
        const v8f y0 = aie::add(fmul8(fmul8(aie::sub(aie::load_v<8>(xf + r * 16), m0), r0), g), b);
        const v8f y1 = aie::add(fmul8(fmul8(aie::sub(aie::load_v<8>(xf + r * 16 + 8), m1), r1), g), b);
        aie::store_v(XT + r * 16, f_to_row(y0, y1));
    }
}

// per-row bias (saturating), ReLU, residual, on ROWS rows of 16 lanes
template <int ROWS>
static inline void bias_rows(int16* __restrict XT, const int16* __restrict b)
{
    for (int r = 0; r < ROWS; r++)
        aie::store_v(XT + r * 16, add_sat16(aie::load_v<16>(XT + r * 16), aie::broadcast<int16, 16>(b[r])));
}
template <int ROWS>
static inline void relu_rows(int16* __restrict XT)
{
    const v16s z = aie::zeros<int16, 16>();
    for (int r = 0; r < ROWS; r++)
        aie::store_v(XT + r * 16, aie::max(aie::load_v<16>(XT + r * 16), z));
}
template <int ROWS>
static inline void scale_rows(int16* __restrict XT, int16 scale, int shift)
{
    for (int r = 0; r < ROWS; r++)
        aie::store_v(XT + r * 16, aie::mul(aie::load_v<16>(XT + r * 16), scale).template to_vector<int16>(shift));
}

// Softmax over the key rows 0..KEYS-1 of ST (keys x lanes, scores at
// score_scale), per lane; PT gets ROWS_OUT rows at out_scale, rows >= KEYS zero.
// exp(-d) = 2^-(d log2e / score_scale): k = round(t), degree-4 poly for 2^-f,
// 2^-k by exponent-bit subtraction, sums by vector adds, one vector reciprocal.
template <int KEYS, int ROWS_OUT>
static inline void softmax_lanes(const int16* __restrict ST, int16* __restrict PT,
                                 float score_scale, float out_scale)
{
    v16i m = aie::from_vector<acc48>(aie::load_v<16>(ST)).template to_vector<int32>(0);
    for (int r = 1; r < KEYS; r++)
        m = aie::max(m, aie::from_vector<acc48>(aie::load_v<16>(ST + r * 16)).template to_vector<int32>(0));
    const v8f c_t = aie::broadcast<float, 8>(1.4426950408889634f / score_scale);
    const v8f a1 = aie::broadcast<float, 8>(-0.69314718f), a2 = aie::broadcast<float, 8>(0.24022651f);
    const v8f a3 = aie::broadcast<float, 8>(-0.05550411f), a4 = aie::broadcast<float, 8>(0.00961813f);
    const v8f one = aie::broadcast<float, 8>(1.0f);
    const v8i kmax = aie::broadcast<int32, 8>(60);
    auto exp_half = [&](const v8i& d) -> v8f {
        const v8f t = fmul8(aie::to_float(d), c_t);
        const v8i k = aie::to_fixed<int32>(t);
        const v8f f = aie::sub(t, aie::to_float(k));
        v8f p = aie::add(a3, fmul8(f, a4));
        p = aie::add(a2, fmul8(f, p));
        p = aie::add(a1, fmul8(f, p));
        p = aie::add(one, fmul8(f, p));
        return aie::sub(p.template cast_to<int32>(), aie::upshift(aie::min(k, kmax), 23)).template cast_to<float>();
    };
    alignas(32) float ef[KEYS * 16];
    v8f s0 = aie::zeros<float, 8>(), s1 = aie::zeros<float, 8>();
    for (int r = 0; r < KEYS; r++) {
        const v16i x32 = aie::from_vector<acc48>(aie::load_v<16>(ST + r * 16)).template to_vector<int32>(0);
        const v16i d = aie::sub(m, x32);
        const v8f e0 = exp_half(d.template extract<8>(0)), e1 = exp_half(d.template extract<8>(1));
        aie::store_v(ef + r * 16, e0); aie::store_v(ef + r * 16 + 8, e1);
        s0 = aie::add(s0, e0); s1 = aie::add(s1, e1);
    }
    const v8f osc = aie::broadcast<float, 8>(out_scale);
    const v8f i0 = fmul8(recip8(s0), osc), i1 = fmul8(recip8(s1), osc);
    for (int r = 0; r < KEYS; r++)
        aie::store_v(PT + r * 16, f_to_row(fmul8(aie::load_v<8>(ef + r * 16), i0), fmul8(aie::load_v<8>(ef + r * 16 + 8), i1)));
    const v16s z = aie::zeros<int16, 16>();
    for (int r = KEYS; r < ROWS_OUT; r++) aie::store_v(PT + r * 16, z);
}

#endif // TRANSPOSED_H
