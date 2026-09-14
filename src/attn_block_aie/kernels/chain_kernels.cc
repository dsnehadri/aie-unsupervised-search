#include "chain_kernels.h"
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <adf.h>
#include "win_vec.h"

// Arithmetic matches the fabric stages bit for bit: the fabric's data_t is a
// 16-bit fixed-point type at scale 512 with wrap-around on overflow, so the
// candidate sums and the isr bias wrap here too (saturation mode "none").

template <int N>
static inline void write5(const int16* __restrict buf, output_window_int16* o0, output_window_int16* o1,
                          output_window_int16* o2, output_window_int16* o3, output_window_int16* o4)
{
    for (int i = 0; i < N; i += 16) {
        const v16_t v = aie::load_v<16>(buf + i);
        win_write16(o0, v); win_write16(o1, v); win_write16(o2, v); win_write16(o3, v); win_write16(o4, v);
    }
}

static inline void assemble(input_window_int16* __restrict x_in, input_window_int16* __restrict mask_in,
    output_window_int16* o0, output_window_int16* o1, output_window_int16* o2, output_window_int16* o3, output_window_int16* o4,
    bool zero_padded)
{
    alignas(16) int16 mask[E_DIM];
    aie::store_v(mask, win_read16(mask_in));
    alignas(16) int16 xm[(N_MAX + 1) * E_DIM];
    win_read_v<N_MAX * E_DIM>(x_in, xm);
    const v16_t z = aie::zeros<int16, 16>();
    if (zero_padded)
        for (int j = 0; j < N_MAX; j++)
            if (mask[j] != 0) aie::store_v(xm + j * E_DIM, z);
    for (int j = 0; j < E_DIM; j++) xm[N_MAX * E_DIM + j] = (j < N_MAX && mask[j] != 0) ? 1 : 0;   // mask row
    write5<(N_MAX + 1) * E_DIM>(xm, o0, o1, o2, o3, o4);
}

void chain_assemble_zero(input_window_int16* __restrict x_in, input_window_int16* __restrict mask_in,
    output_window_int16* __restrict o0, output_window_int16* __restrict o1, output_window_int16* __restrict o2,
    output_window_int16* __restrict o3, output_window_int16* __restrict o4)
{ assemble(x_in, mask_in, o0, o1, o2, o3, o4, true); }

void chain_assemble(input_window_int16* __restrict x_in, input_window_int16* __restrict mask_in,
    output_window_int16* __restrict o0, output_window_int16* __restrict o1, output_window_int16* __restrict o2,
    output_window_int16* __restrict o3, output_window_int16* __restrict o4)
{ assemble(x_in, mask_in, o0, o1, o2, o3, o4, false); }

void chain_post_obj(input_window_int16* __restrict x_in, input_window_int16* __restrict mask_in,
    output_window_int16* __restrict x0, output_window_int16* __restrict x1, output_window_int16* __restrict x2,
    output_window_int16* __restrict x3, output_window_int16* __restrict x4,
    output_window_int16* __restrict c0, output_window_int16* __restrict c1, output_window_int16* __restrict c2,
    output_window_int16* __restrict c3, output_window_int16* __restrict c4)
{
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::none);
    alignas(16) int16 mask[E_DIM];
    aie::store_v(mask, win_read16(mask_in));
    alignas(16) int16 xr[N_MAX * E_DIM];
    win_read_v<N_MAX * E_DIM>(x_in, xr);
    alignas(16) int16 c[T_DIM * E_DIM];
    zero_v<T_DIM * E_DIM>(c);
    const v16_t z = aie::zeros<int16, 16>();
    for (int j = 0; j < N_MAX; j++) {
        int16* row = xr + j * E_DIM;
        if (mask[j] != 0) aie::store_v(row, z);                  // remask
        row[2] = (int16)(row[2] - (int16)(1 << DATA_FRAC_BITS)); // isr bias: -1.0 at Q6.9, wraps
        int16 best = row[0]; int t = 0;                          // argmax over the first T_DIM
        for (int k = 1; k < T_DIM; k++) if (row[k] > best) { best = row[k]; t = k; }
        const aie::vector<int32, 16> a = aie::from_vector<acc48>(aie::load_v<16>(c + t * E_DIM)).to_vector<int32>(0);
        const aie::vector<int32, 16> b = aie::from_vector<acc48>(aie::load_v<16>(row)).to_vector<int32>(0);
        aie::store_v(c + t * E_DIM, aie::from_vector<acc80>(aie::add(a, b)).to_vector<int16>(0));   // wraps
    }
    write5<N_MAX * E_DIM>(xr, x0, x1, x2, x3, x4);
    write5<T_DIM * E_DIM>(c, c0, c1, c2, c3, c4);
    aie::set_saturation(sat_save);
}

void chain_fanout_c(input_window_int16* __restrict c_in,
    output_window_int16* __restrict o0, output_window_int16* __restrict o1, output_window_int16* __restrict o2,
    output_window_int16* __restrict o3, output_window_int16* __restrict o4)
{
    alignas(16) int16 c[T_DIM * E_DIM];
    win_read_v<T_DIM * E_DIM>(c_in, c);
    write5<T_DIM * E_DIM>(c, o0, o1, o2, o3, o4);
}

void chain_fanout_c4(input_window_int16* __restrict c_in,
    output_window_int16* __restrict o0, output_window_int16* __restrict o1, output_window_int16* __restrict o2,
    output_window_int16* __restrict o3)
{
    alignas(16) int16 c[T_DIM * E_DIM];
    win_read_v<T_DIM * E_DIM>(c_in, c);
    for (int i = 0; i < T_DIM * E_DIM; i += 16) {
        const v16_t v = aie::load_v<16>(c + i);
        win_write16(o0, v); win_write16(o1, v); win_write16(o2, v); win_write16(o3, v);
    }
}
