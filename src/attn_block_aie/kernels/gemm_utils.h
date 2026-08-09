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

#endif // GEMM_UTILS_H
