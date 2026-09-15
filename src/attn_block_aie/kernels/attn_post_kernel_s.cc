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

#if defined(ROW_LOOKAHEAD)
// One row of lookahead: compute row r+1's linear layer in the same iteration as
// row r's layer norm. The norm is a dependent chain of reductions and scalar
// steps, about 135 of the roughly 215 cycles a stage spends per row, and the
// next row's products do not depend on it, so putting the two in one iteration
// lets the scheduler overlap them. Each row still gets linear, norm, ReLU in
// that order, so the outputs are bit-identical.
static inline void lin_ln_relu_stage(input_stream_int16* __restrict in,
                                     output_stream_int16* __restrict out,
                                     const int16* __restrict W, const int16* __restrict B,
                                     const int16* __restrict G, const int16* __restrict BT)
{
    alignas(16) int16 xrow[E_DIM], row[E_DIM];
    aie::store_v(xrow, row_read(in));
    v16_t lin = row_lin16(xrow, W, B, PIPE_ACC_SHIFT);
    for (int r = 1; r < POST_N_ROWS; r++) {
        aie::store_v(xrow, row_read(in));
        const v16_t nxt = row_lin16(xrow, W, B, PIPE_ACC_SHIFT);   // row r+1, independent
        aie::store_v(row, lin);
        layernorm_row(row, 1, E_DIM, G, BT);                       // row r
        row_write(out, relu16(aie::load_v<16>(row)));
        lin = nxt;
    }
    aie::store_v(row, lin);
    layernorm_row(row, 1, E_DIM, G, BT);
    row_write(out, relu16(aie::load_v<16>(row)));
}
#endif

#if defined(POST_STAGE_A_PROJ)
#if defined(HEAD_STREAM_T)
// HEAD_STREAM: the head outputs arrive as rows, already paired by the two merge
// kernels, so a row is one 8-lane read from each. This replaces the gather from
// four windows entirely, and the kernel now starts on row 0 while the head-post
// kernels are still working on row 4.
void POST_A_PROJ_FN(input_stream_int16* __restrict h01_in,
                      input_stream_int16* __restrict h23_in,
                      input_window_int16* __restrict residual_in,
                      output_stream_int16* __restrict proj_out)
{
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    alignas(16) int16 xrow[E_DIM], row[E_DIM];
    for (int r = 0; r < POST_N_ROWS; r++) {
        const aie::vector<int16, 8> lo(readincr_v8(h01_in));   // [h0 h1] of this row
        const aie::vector<int16, 8> hi(readincr_v8(h23_in));   // [h2 h3]
        aie::store_v(xrow, aie::concat(lo, hi));
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
#else
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

    // One row is D_HEAD = 4 words from each of the four head windows, and the
    // smallest 16-bit vector this device loads is 8 lanes, so take TWO rows
    // from every head at once and separate them with two 4-lane zips:
    //   A = [h0r0 h0r1 h1r0 h1r1], B = [h2r0 h2r1 h3r0 h3r1]
    //   zip(A,B,4)   -> [h0r0 h2r0 h0r1 h2r1] , [h1r0 h3r0 h1r1 h3r1]
    //   zip(lo,hi,4) -> [h0r0 h1r0 h2r0 h3r0] = row 0, and row 1 in the other half
    // The scalar gather this replaces cost ~18 cycles per word, 16 per row.
    auto do_row = [&](const v16_t& xv) {
        aie::store_v(xrow, xv);
#if !defined(ATTN_TYPE_CROSS)
        alignas(16) int16 res[E_DIM];
        aie::store_v(res, win_read16(residual_in));
        aie::store_v(row, row_lin16(xrow, Wout, bout, PIPE_ACC_SHIFT, res));
#else
        aie::store_v(row, row_lin16(xrow, Wout, bout, PIPE_ACC_SHIFT));
#endif
        layernorm_row(row, 1, E_DIM, post_attn_ln_gamma, post_attn_ln_beta);
        row_write(proj_out, aie::load_v<16>(row));
    };

    int r = 0;
    for (; r + 1 < POST_N_ROWS; r += 2) {
        const aie::vector<int16, 8> a0(window_readincr_v8(head0_in));
        const aie::vector<int16, 8> a1(window_readincr_v8(head1_in));
        const aie::vector<int16, 8> a2(window_readincr_v8(head2_in));
        const aie::vector<int16, 8> a3(window_readincr_v8(head3_in));
        const auto z = aie::interleave_zip(aie::concat(a0, a1), aie::concat(a2, a3), 4);
        const auto w = aie::interleave_zip(z.first, z.second, 4);
        do_row(w.first);
        do_row(w.second);
    }
    for (; r < POST_N_ROWS; r++) {           // odd tail (candidate blocks: 3 rows)
        for (int h = 0; h < N_HEADS; h++)
            for (int d = 0; d < D_HEAD; d++) xrow[h * D_HEAD + d] = window_readincr(heads[h]);
        do_row(aie::load_v<16>(xrow));
    }
    aie::set_saturation(sat_save);
}
#endif // HEAD_STREAM_T
#endif

#if defined(POST_STAGE_B1)
void POST_B1_FN(input_stream_int16* __restrict proj_in, output_stream_int16* __restrict ffn0_out)
{
#if defined(ROW_LOOKAHEAD)
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    lin_ln_relu_stage(proj_in, ffn0_out, ffn_W0, ffn_b0, ffn_ln_gamma0, ffn_ln_beta0);
    aie::set_saturation(sat_save);
#else
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    alignas(16) int16 xrow[E_DIM], row[E_DIM];
    for (int r = 0; r < POST_N_ROWS; r++) {
        aie::store_v(xrow, row_read(proj_in));
        aie::store_v(row, row_lin16(xrow, ffn_W0, ffn_b0, PIPE_ACC_SHIFT));
        layernorm_row(row, 1, E_DIM, ffn_ln_gamma0, ffn_ln_beta0);
        row_write(ffn0_out, relu16(aie::load_v<16>(row)));
    }
    aie::set_saturation(sat_save);
#endif
}
#endif

#if defined(POST_STAGE_B2)
void POST_B2_FN(input_stream_int16* __restrict ffn0_in, output_stream_int16* __restrict ffn1_out)
{
#if defined(ROW_LOOKAHEAD)
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    lin_ln_relu_stage(ffn0_in, ffn1_out, ffn_W1, ffn_b1, ffn_ln_gamma1, ffn_ln_beta1);
    aie::set_saturation(sat_save);
#else
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    alignas(16) int16 xrow[E_DIM], row[E_DIM];
    for (int r = 0; r < POST_N_ROWS; r++) {
        aie::store_v(xrow, row_read(ffn0_in));
        aie::store_v(row, row_lin16(xrow, ffn_W1, ffn_b1, PIPE_ACC_SHIFT));
        layernorm_row(row, 1, E_DIM, ffn_ln_gamma1, ffn_ln_beta1);
        row_write(ffn1_out, relu16(aie::load_v<16>(row)));
    }
    aie::set_saturation(sat_save);
#endif
}
#endif

#if defined(POST_SPLIT_C)
// POST_SPLIT_C: this was the slowest stage in the block -- a linear layer and
// TWO layer norms on every row, about 1.6x the per-row cost of b1 or b2 -- and
// a row pipeline drains at the rate of its slowest stage. Split in two:
//   c1: FFN layer 2 -> layer norm -> ReLU
//   c : add the projection residual -> layer norm -> out
// Each half now costs about what b1 and b2 do. The arithmetic and its order are
// untouched, so the outputs are bit-identical.
#if defined(POST_STAGE_C1)
void POST_C1_FN(input_stream_int16* __restrict ffn_in, output_stream_int16* __restrict ffn_out)
{
#if defined(ROW_LOOKAHEAD)
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    lin_ln_relu_stage(ffn_in, ffn_out, ffn_W2, ffn_b2, ffn_ln_gamma2, ffn_ln_beta2);
    aie::set_saturation(sat_save);
#else
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    alignas(16) int16 xrow[E_DIM], row[E_DIM];
    for (int r = 0; r < POST_N_ROWS; r++) {
        aie::store_v(xrow, row_read(ffn_in));
        aie::store_v(row, row_lin16(xrow, ffn_W2, ffn_b2, PIPE_ACC_SHIFT));
        layernorm_row(row, 1, E_DIM, ffn_ln_gamma2, ffn_ln_beta2);
        row_write(ffn_out, relu16(aie::load_v<16>(row)));
    }
    aie::set_saturation(sat_save);
#endif
}
#endif
#if defined(POST_STAGE_C)
void POST_C_FN(input_stream_int16* __restrict c1_in,
                 input_stream_int16* __restrict residual_b_in,
                 output_stream_int16* __restrict x_out)
{
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    alignas(16) int16 row[E_DIM];
    for (int r = 0; r < POST_N_ROWS; r++) {
        aie::store_v(row, add_sat16(row_read(c1_in), row_read(residual_b_in)));   // skip with proj
        layernorm_row(row, 1, E_DIM, post_ffn_ln_gamma, post_ffn_ln_beta);
        row_write(x_out, aie::load_v<16>(row));
    }
    aie::set_saturation(sat_save);
}
#endif
#else
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
#endif  // POST_SPLIT_C
