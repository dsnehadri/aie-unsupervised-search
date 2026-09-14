#include "chain_kernels.h"
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <adf.h>
#include "win_vec.h"

// Arithmetic matches the fabric stages bit for bit: the fabric's data_t is a
// 16-bit fixed-point type at scale 512 with wrap-around on overflow, so the
// candidate sums and the isr bias wrap here too (saturation mode "none").

static inline void assemble(input_window_int16* __restrict x_in, input_window_int16* __restrict mask_in,
                            output_window_int16* __restrict x_out, bool zero_padded)
{
    alignas(16) int16 mask[E_DIM];
    aie::store_v(mask, win_read16(mask_in));
    const v16_t z = aie::zeros<int16, 16>();
    for (int j = 0; j < N_MAX; j++) {
        const v16_t row = win_read16(x_in);
        win_write16(x_out, (zero_padded && mask[j] != 0) ? z : row);
    }
    alignas(16) int16 mrow[E_DIM];
    for (int j = 0; j < E_DIM; j++) mrow[j] = (j < N_MAX && mask[j] != 0) ? 1 : 0;   // mask row
    win_write16(x_out, aie::load_v<16>(mrow));
}
void chain_assemble_zero(input_window_int16* __restrict x_in, input_window_int16* __restrict mask_in,
                         output_window_int16* __restrict x_out) { assemble(x_in, mask_in, x_out, true); }
void chain_assemble(input_window_int16* __restrict x_in, input_window_int16* __restrict mask_in,
                    output_window_int16* __restrict x_out) { assemble(x_in, mask_in, x_out, false); }

void chain_post_obj(input_window_int16* __restrict x_in, input_window_int16* __restrict mask_in,
                    output_window_int16* __restrict x_out, output_window_int16* __restrict c_out)
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
    win_write_v<N_MAX * E_DIM>(x_out, xr);
    win_write_v<T_DIM * E_DIM>(c_out, c);
    aie::set_saturation(sat_save);
}

template <int N>
static inline void dup2(input_window_int16* __restrict in, output_window_int16* __restrict o0, output_window_int16* __restrict o1)
{
    for (int i = 0; i < N; i += 16) { const v16_t v = win_read16(in); win_write16(o0, v); win_write16(o1, v); }
}
void chain_dup2_208(input_window_int16* __restrict in, output_window_int16* __restrict o0, output_window_int16* __restrict o1) { dup2<208>(in, o0, o1); }
void chain_dup2_192(input_window_int16* __restrict in, output_window_int16* __restrict o0, output_window_int16* __restrict o1) { dup2<192>(in, o0, o1); }
void chain_dup2_48 (input_window_int16* __restrict in, output_window_int16* __restrict o0, output_window_int16* __restrict o1) { dup2<48>(in, o0, o1); }
