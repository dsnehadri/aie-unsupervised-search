// Vector window I/O and small vector helpers for the int16 attention kernels.
//
// The aiesimulator profile (figs/aie_obj_block_profile.txt, 2026-09-12) showed
// that after the vector gemm and layer norm the post kernels still spent about
// 80% of their cycles in scalar window_readincr/window_writeincr loops (about
// 18 cycles per int16: each call is a cyclic pointer update) and in a byte-wise
// memset of the "= {0}" stack arrays. Every window here is a whole number of
// 16-lane vectors except the candidate head output (12 words) and the wij bias
// (156 words), which keep a scalar tail.
//
// Packed layout (gemm_utils.h pk_idx): block (r/4, c/4) of a K=16 matrix is 16
// contiguous int16. Four row-major rows r0..r3 become four blocks with three
// interleave_zip steps: zip(r0,r1,4) and zip(r2,r3,4) group 4-lane chunks of
// row pairs, zip(.,.,8) then puts the four rows of each chunk together.
#ifndef WIN_VEC_H
#define WIN_VEC_H

#include <aie_api/aie.hpp>
#include <adf.h>

typedef aie::vector<int16, 16> v16_t;

static inline v16_t win_read16(input_window_int16* __restrict w)
{
    return v16_t(window_readincr_v16(w));
}

static inline void win_write16(output_window_int16* __restrict w, const v16_t& v)
{
    window_writeincr(w, v.to_native());
}

// rows r0..r3 (row-major, 16 columns) -> packed blocks P[0..63]
static inline void pack_rows4(const v16_t& r0, const v16_t& r1,
                              const v16_t& r2, const v16_t& r3,
                              int16* __restrict P)
{
    const auto x = aie::interleave_zip(r0, r1, 4);
    const auto y = aie::interleave_zip(r2, r3, 4);
    const auto b01 = aie::interleave_zip(x.first,  y.first,  8);
    const auto b23 = aie::interleave_zip(x.second, y.second, 8);
    aie::store_v(P,      b01.first);
    aie::store_v(P + 16, b01.second);
    aie::store_v(P + 32, b23.first);
    aie::store_v(P + 48, b23.second);
}

// ROWS row-major rows of 16 from a window -> packed. ROWS % 4 rows of the last
// block are read, the rest of that block is zero.
template <int ROWS>
static inline void win_read_packed16(input_window_int16* __restrict w, int16* __restrict P)
{
    constexpr int FULL = ROWS / 4;
    for (int b = 0; b < FULL; b++) {
        const v16_t r0 = win_read16(w);
        const v16_t r1 = win_read16(w);
        const v16_t r2 = win_read16(w);
        const v16_t r3 = win_read16(w);
        pack_rows4(r0, r1, r2, r3, P + b * 64);
    }
    if constexpr (ROWS % 4 != 0) {
        v16_t z = aie::zeros<int16, 16>();
        v16_t r0 = win_read16(w);
        v16_t r1 = z, r2 = z;
        if (ROWS % 4 > 1) r1 = win_read16(w);
        if (ROWS % 4 > 2) r2 = win_read16(w);
        pack_rows4(r0, r1, r2, z, P + FULL * 64);
    }
}

// row-major local matrix (ROWS x 16, ROWS % 4 == 0) -> packed
template <int ROWS>
static inline void pack_local16(const int16* __restrict A, int16* __restrict P)
{
    for (int b = 0; b < ROWS / 4; b++) {
        const v16_t r0 = aie::load_v<16>(A + (b * 4 + 0) * 16);
        const v16_t r1 = aie::load_v<16>(A + (b * 4 + 1) * 16);
        const v16_t r2 = aie::load_v<16>(A + (b * 4 + 2) * 16);
        const v16_t r3 = aie::load_v<16>(A + (b * 4 + 3) * 16);
        pack_rows4(r0, r1, r2, r3, P + b * 64);
    }
}

// N contiguous int16 (N % 16 == 0), local -> window / window -> local
template <int N>
static inline void win_write_v(output_window_int16* __restrict w, const int16* __restrict A)
{
    for (int i = 0; i < N; i += 16) win_write16(w, aie::load_v<16>(A + i));
}
template <int N>
static inline void win_read_v(input_window_int16* __restrict w, int16* __restrict A)
{
    for (int i = 0; i < N; i += 16) aie::store_v(A + i, win_read16(w));
}

template <int N>
static inline void zero_v(int16* __restrict A)
{
    const v16_t z = aie::zeros<int16, 16>();
    for (int i = 0; i < N; i += 16) aie::store_v(A + i, z);
}

// saturating int16 add, a + b clamped to [-32768, 32767]. The int32 detour
// and the acc80 srs are the same ops layernorm_row uses; the srs saturates
// because the callers set aie::saturation_mode::saturate.
static inline v16_t add_sat16(const v16_t& a, const v16_t& b)
{
    const aie::vector<int32, 16> a32 = aie::from_vector<acc48>(a).to_vector<int32>(0);
    const aie::vector<int32, 16> b32 = aie::from_vector<acc48>(b).to_vector<int32>(0);
    return aie::from_vector<acc80>(aie::add(a32, b32)).to_vector<int16>(0);
}

// mat[ROWS][16] += bias[16], saturating (replaces the scalar add_bias_sat)
template <int ROWS>
static inline void add_bias_v16(int16* __restrict mat, const int16* __restrict bias)
{
    const v16_t bv = aie::load_v<16>(bias);
    for (int r = 0; r < ROWS; r++)
        aie::store_v(mat + r * 16, add_sat16(aie::load_v<16>(mat + r * 16), bv));
}

// mat[ROWS][4] += bias[4], saturating; 4 rows per vector
template <int ROWS>
static inline void add_bias_v4(int16* __restrict mat, const int16* __restrict bias)
{
    alignas(16) int16 rep[16];
    for (int i = 0; i < 16; i++) rep[i] = bias[i & 3];
    const v16_t bv = aie::load_v<16>(rep);
    for (int r = 0; r < ROWS; r += 4)
        aie::store_v(mat + r * 4, add_sat16(aie::load_v<16>(mat + r * 4), bv));
}

// mat[ROWS][16] += add[ROWS][16], saturating
template <int ROWS>
static inline void add_rows_v16(int16* __restrict mat, const int16* __restrict add)
{
    for (int r = 0; r < ROWS; r++)
        aie::store_v(mat + r * 16, add_sat16(aie::load_v<16>(mat + r * 16),
                                             aie::load_v<16>(add + r * 16)));
}

#endif // WIN_VEC_H
