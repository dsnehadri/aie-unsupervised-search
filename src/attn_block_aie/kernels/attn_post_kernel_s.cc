// POST_STREAM: the post-attention pipeline at row granularity. a_proj reads the
// four head windows and the residual window (a tile has only two stream
// inputs) and streams projected rows; b1, b2 and c each take a row, apply
// their linear layer (16 vector-scalar macs, bias and residual in the
// accumulator), layer norm and ReLU, and pass it on, so row r+1 is in a_proj
// while row r is in c. The arithmetic is the same as the window kernels:
// products accumulate exactly in acc48, the layer norm is per row anyway.
#include "gemm_utils.h"
#include "win_vec.h"
#include "layernorm_int.h"

// y = x . W (row-major 16 x 16) + bias [+ resid], >> shift, saturating
static inline v16_t row_lin16(const int16* __restrict x, const int16* __restrict W,
                              const int16* __restrict bias, int shift, const int16* __restrict resid = nullptr)
{
    aie::accum<acc48, 16> acc = aie::mul(aie::load_v<16>(W), x[0]);
    for (int k = 1; k < 16; k++) acc = aie::mac(acc, aie::load_v<16>(W + k * 16), x[k]);
    const int16 sc = (int16)(1 << shift);
    acc = aie::mac(acc, aie::load_v<16>(bias), sc);
    if (resid) acc = aie::mac(acc, aie::load_v<16>(resid), sc);
    return acc.template to_vector<int16>(shift);
}
static inline v16_t row_read(input_stream_int16* __restrict s)
{
    const aie::vector<int16, 8> a(readincr_v8(s));
    const aie::vector<int16, 8> b(readincr_v8(s));
    return aie::concat(a, b);
}
static inline void row_write(output_stream_int16* __restrict s, const v16_t& v)
{
    writeincr_v8(s, v.template extract<8>(0).to_native());
    writeincr_v8(s, v.template extract<8>(1).to_native());
}
static inline v16_t relu16(const v16_t& v) { return aie::max(v, aie::zeros<int16, 16>()); }

#if defined(POST_STAGE_A_PROJ)
void POST_A_PROJ_FN(input_window_int16* __restrict head0_in,
                      input_window_int16* __restrict head1_in,
                      input_window_int16* __restrict head2_in,
                      input_window_int16* __restrict head3_in,
                      input_window_int16* __restrict residual_in,
                      output_stream_int16* __restrict proj_out)
{
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    input_window_int16* __restrict heads[N_HEADS] = {head0_in, head1_in, head2_in, head3_in};
    alignas(16) int16 xrow[E_DIM], row[E_DIM];
    for (int r = 0; r < POST_N_ROWS; r++) {
        for (int h = 0; h < N_HEADS; h++)
            for (int d = 0; d < D_HEAD; d++) xrow[h * D_HEAD + d] = window_readincr(heads[h]);
#if !defined(ATTN_TYPE_CROSS)
        alignas(16) int16 res[E_DIM];
        aie::store_v(res, win_read16(residual_in));
        aie::store_v(row, row_lin16(xrow, Wout, bout, PIPE_ACC_SHIFT, res));
#else
        aie::store_v(row, row_lin16(xrow, Wout, bout, PIPE_ACC_SHIFT));
#endif
        layernorm_row(row, 1, E_DIM, post_attn_ln_gamma, post_attn_ln_beta);
        row_write(proj_out, aie::load_v<16>(row));
    }
    aie::set_saturation(sat_save);
}
#endif

#if defined(POST_STAGE_B1)
void POST_B1_FN(input_stream_int16* __restrict proj_in, output_stream_int16* __restrict ffn0_out)
{
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    alignas(16) int16 xrow[E_DIM], row[E_DIM];
    for (int r = 0; r < POST_N_ROWS; r++) {
        aie::store_v(xrow, row_read(proj_in));
        aie::store_v(row, row_lin16(xrow, ffn_W0, ffn_b0, PIPE_ACC_SHIFT));
        layernorm_row(row, 1, E_DIM, ffn_ln_gamma0, ffn_ln_beta0);
        row_write(ffn0_out, relu16(aie::load_v<16>(row)));
    }
    aie::set_saturation(sat_save);
}
#endif

#if defined(POST_STAGE_B2)
void POST_B2_FN(input_stream_int16* __restrict ffn0_in, output_stream_int16* __restrict ffn1_out)
{
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    alignas(16) int16 xrow[E_DIM], row[E_DIM];
    for (int r = 0; r < POST_N_ROWS; r++) {
        aie::store_v(xrow, row_read(ffn0_in));
        aie::store_v(row, row_lin16(xrow, ffn_W1, ffn_b1, PIPE_ACC_SHIFT));
        layernorm_row(row, 1, E_DIM, ffn_ln_gamma1, ffn_ln_beta1);
        row_write(ffn1_out, relu16(aie::load_v<16>(row)));
    }
    aie::set_saturation(sat_save);
}
#endif

#if defined(POST_STAGE_C)
void POST_C_FN(input_stream_int16* __restrict ffn_in,
                 input_stream_int16* __restrict residual_b_in,
                 output_stream_int16* __restrict x_out)
{
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    alignas(16) int16 xrow[E_DIM], row[E_DIM];
    for (int r = 0; r < POST_N_ROWS; r++) {
        aie::store_v(xrow, row_read(ffn_in));
        aie::store_v(row, row_lin16(xrow, ffn_W2, ffn_b2, PIPE_ACC_SHIFT));
        layernorm_row(row, 1, E_DIM, ffn_ln_gamma2, ffn_ln_beta2);
        const v16_t y = add_sat16(relu16(aie::load_v<16>(row)), row_read(residual_b_in));   // skip with proj
        aie::store_v(row, y);
        layernorm_row(row, 1, E_DIM, post_ffn_ln_gamma, post_ffn_ln_beta);
        row_write(x_out, aie::load_v<16>(row));
    }
    aie::set_saturation(sat_save);
}
#endif
