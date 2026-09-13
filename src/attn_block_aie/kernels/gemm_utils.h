// Vectorized tiled GEMM for the attention kernels.
//
// The previous gemm_tile packed va/vb with 32 scalar lane-inserts per
// 4x4x4 mmul step (64 MACs) -- the profiler showed the vector unit ~idle
// while tiles burned scalar cycles building operands. Here every operand
// is a vector load:
//   - A is kept in 4x4-block-packed layout ("pk"): block (r/4, c/4) is 16
//     contiguous int16. Callers fill A packed during the window-read loop
//     (same scalar cost as before) via pk_idx<K>(). For K==4 the packed
//     layout IS row-major, so 4-column matrices (Q, attention weights vs
//     T_KV, ...) can be passed through untouched.
//   - B stays row-major. N==4: each k-block of 4 rows is 16 contiguous
//     int16 (all QKV projection weights are ExD_HEAD -> no repacking).
//     N%8==0: rows are loaded as aligned 8-lane vectors (mmul<4,4,8>).
//   - C is written with vector stores (contiguous for N==4, 8-lane row
//     segments otherwise).
// All base buffers must be alignas(16); all dims multiples of 4.

#ifndef GEMM_UTILS_H
#define GEMM_UTILS_H

#include <aie_api/aie.hpp>

// packed index for element (r, c) of an M x K matrix
template <int K>
static inline constexpr int pk_idx(int r, int c)
{
    return ((r / 4) * (K / 4) + (c / 4)) * 16 + (r % 4) * 4 + (c % 4);
}

// scalar fallback pack (used where the data is produced row-major, e.g.
// softmax output feeding the AV gemm)
template <int M, int K>
static inline void pack_a4(const int16* __restrict A, int16* __restrict P)
{
    for (int r = 0; r < M; r++)
        for (int c = 0; c < K; c++)
            P[pk_idx<K>(r, c)] = A[r * K + c];
}

// C[M][N] = Ap[M][K] (packed) x B[K][N] (row-major), >> shift
template <int M, int K, int N>
static inline void gemm_pk(const int16* __restrict Ap, const int16* __restrict B,
                           int16* __restrict C, int shift)
{
    static_assert(M % 4 == 0 && K % 4 == 0, "gemm_pk: M,K must be multiples of 4");
    static_assert(N == 4 || N % 8 == 0, "gemm_pk: N must be 4 or a multiple of 8");

    if constexpr (N == 4) {
        for (int m = 0; m < M; m += 4) {
            aie::mmul<4, 4, 4, int16, int16> acc;
            for (int k = 0; k < K; k += 4) {
                aie::vector<int16, 16> va = aie::load_v<16>(&Ap[((m / 4) * (K / 4) + (k / 4)) * 16]);
                aie::vector<int16, 16> vb = aie::load_v<16>(&B[k * 4]);
                if (k == 0) acc.mul(va, vb); else acc.mac(va, vb);
            }
            aie::store_v(&C[m * 4], acc.template to_vector<int16>(shift));
        }
    } else {
        for (int m = 0; m < M; m += 4) {
            for (int n = 0; n < N; n += 8) {
                aie::mmul<4, 4, 8, int16, int16> acc;
                for (int k = 0; k < K; k += 4) {
                    aie::vector<int16, 16> va = aie::load_v<16>(&Ap[((m / 4) * (K / 4) + (k / 4)) * 16]);
                    aie::vector<int16, 32> vb = aie::concat(
                        aie::load_v<8>(&B[(k + 0) * N + n]), aie::load_v<8>(&B[(k + 1) * N + n]),
                        aie::load_v<8>(&B[(k + 2) * N + n]), aie::load_v<8>(&B[(k + 3) * N + n]));
                    if (k == 0) acc.mul(va, vb); else acc.mac(va, vb);
                }
                aie::vector<int16, 32> res = acc.template to_vector<int16>(shift);
                aie::store_v(&C[(m + 0) * N + n], res.template extract<8>(0));
                aie::store_v(&C[(m + 1) * N + n], res.template extract<8>(1));
                aie::store_v(&C[(m + 2) * N + n], res.template extract<8>(2));
                aie::store_v(&C[(m + 3) * N + n], res.template extract<8>(3));
            }
        }
    }
}

// C = Ap x B + bias per row, bias added in the accumulator (at 2^shift), one
// saturating srs. biasrep holds 8 copies of each row's bias, row-major
// (M x 8), so a 4-row block is one 32-lane vector. Replaces a separate
// saturating-add pass (~40 cycles a row on AIE1).
template <int M, int K, int N>
__attribute__((noinline)) static void gemm_pk_bias(const int16* __restrict Ap, const int16* __restrict B,
                                int16* __restrict C, int shift, const int16* __restrict biasrep)
{
    static_assert(M % 4 == 0 && K % 4 == 0 && N % 8 == 0, "gemm_pk_bias: M,K multiples of 4, N of 8");
    for (int m = 0; m < M; m += 4) {
        const aie::vector<int16, 32> bv = aie::load_v<32>(&biasrep[m * 8]);
        const aie::vector<int16, 32> sc = aie::broadcast<int16, 32>((int16)(1 << shift));   // bias at 2^shift
        for (int n = 0; n < N; n += 8) {
            aie::mmul<4, 4, 8, int16, int16> acc;
            for (int k = 0; k < K; k += 4) {
                aie::vector<int16, 16> va = aie::load_v<16>(&Ap[((m / 4) * (K / 4) + (k / 4)) * 16]);
                aie::vector<int16, 32> vb = aie::concat(
                    aie::load_v<8>(&B[(k + 0) * N + n]), aie::load_v<8>(&B[(k + 1) * N + n]),
                    aie::load_v<8>(&B[(k + 2) * N + n]), aie::load_v<8>(&B[(k + 3) * N + n]));
                if (k == 0) acc.mul(va, vb); else acc.mac(va, vb);
            }
            const aie::accum<acc48, 32> sum = aie::mac(acc.to_accum(), bv, sc);
            aie::vector<int16, 32> res = sum.template to_vector<int16>(shift);
            aie::store_v(&C[(m + 0) * N + n], res.template extract<8>(0));
            aie::store_v(&C[(m + 1) * N + n], res.template extract<8>(1));
            aie::store_v(&C[(m + 2) * N + n], res.template extract<8>(2));
            aie::store_v(&C[(m + 3) * N + n], res.template extract<8>(3));
        }
    }
}

// Row-major variant: C[m][n] += bias[n] (and optionally a full residual
// R[m][n]) inside the accumulator. biasrep holds, per 8-column chunk, the 8
// bias values repeated for 4 rows (N/8 x 32 lanes), built once per tile.
// (acc + b * 2^shift) >> shift == (acc >> shift) + b exactly, so this matches
// the old "gemm, then saturating add" except when the gemm alone saturated.
template <int M, int K, int N>
static inline void gemm_pk_biasc(const int16* __restrict Ap, const int16* __restrict B,
                                 int16* __restrict C, int shift, const int16* __restrict biasrep,
                                 const int16* __restrict R = nullptr)
{
    static_assert(M % 4 == 0 && K % 4 == 0 && N % 8 == 0, "gemm_pk_biasc: M,K multiples of 4, N of 8");
    const aie::vector<int16, 32> sc = aie::broadcast<int16, 32>((int16)(1 << shift));
    for (int m = 0; m < M; m += 4) {
        for (int n = 0; n < N; n += 8) {
            aie::mmul<4, 4, 8, int16, int16> acc;
            for (int k = 0; k < K; k += 4) {
                aie::vector<int16, 16> va = aie::load_v<16>(&Ap[((m / 4) * (K / 4) + (k / 4)) * 16]);
                aie::vector<int16, 32> vb = aie::concat(
                    aie::load_v<8>(&B[(k + 0) * N + n]), aie::load_v<8>(&B[(k + 1) * N + n]),
                    aie::load_v<8>(&B[(k + 2) * N + n]), aie::load_v<8>(&B[(k + 3) * N + n]));
                if (k == 0) acc.mul(va, vb); else acc.mac(va, vb);
            }
            aie::accum<acc48, 32> sum = aie::mac(acc.to_accum(), aie::load_v<32>(&biasrep[(n / 8) * 32]), sc);
            if (R) {
                const aie::vector<int16, 32> rv = aie::concat(
                    aie::load_v<8>(&R[(m + 0) * N + n]), aie::load_v<8>(&R[(m + 1) * N + n]),
                    aie::load_v<8>(&R[(m + 2) * N + n]), aie::load_v<8>(&R[(m + 3) * N + n]));
                sum = aie::mac(sum, rv, sc);
            }
            aie::vector<int16, 32> res = sum.template to_vector<int16>(shift);
            aie::store_v(&C[(m + 0) * N + n], res.template extract<8>(0));
            aie::store_v(&C[(m + 1) * N + n], res.template extract<8>(1));
            aie::store_v(&C[(m + 2) * N + n], res.template extract<8>(2));
            aie::store_v(&C[(m + 3) * N + n], res.template extract<8>(3));
        }
    }
}

// bias[N] -> per 8-column chunk, 8 values repeated for 4 rows (N/8 x 32)
template <int N>
struct BiasRepC {
    alignas(32) int16 r[(N / 8) * 32];
    bool ready = false;
    inline void build(const int16* __restrict b)
    {
        for (int c = 0; c < N / 8; c++)
            for (int row = 0; row < 4; row++)
                for (int i = 0; i < 8; i++) r[c * 32 + row * 8 + i] = b[c * 8 + i];
        ready = true;
    }
};
#define BIAS_REPC(name, N, b) static BiasRepC<N> name; if (!name.ready) name.build(b)

#endif // GEMM_UTILS_H
