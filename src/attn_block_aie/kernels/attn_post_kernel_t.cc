// Transposed post kernels (see transposed.h). Included by attn_post_kernel.cc
// when TRANSPOSED is defined. Head outputs arrive as O^T (4 x 16 lanes) and
// stack into the 16 x 16 concat; proj/ffn windows carry 16 x 16 transposed
// tensors; post_c writes the block output row-major again for the PL.
#include "transposed.h"

static constexpr float LN_EPS_Q2 = 1e-5f * (float)PIPE_SCALE * (float)PIPE_SCALE;
#define T_ROWS POST_N_ROWS            // valid lanes: 12 (obj/cross) or 3 (cand)

#if defined(POST_STAGE_A_PROJ)
void POST_A_PROJ_FN(input_window_int16* __restrict head0_in,
                      input_window_int16* __restrict head1_in,
                      input_window_int16* __restrict head2_in,
                      input_window_int16* __restrict head3_in,
                      input_window_int16* __restrict residual_in,
                      output_window_int16* __restrict proj_out)
{
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    alignas(16) int16 CT[16 * 16];                 // concat^T: head h = rows 4h..4h+3
    win_read_v<D_HEAD * 16>(head0_in, CT);
    win_read_v<D_HEAD * 16>(head1_in, CT + 4 * 16);
    win_read_v<D_HEAD * 16>(head2_in, CT + 8 * 16);
    win_read_v<D_HEAD * 16>(head3_in, CT + 12 * 16);

    PACKED_WT(Wo_t, E_DIM, E_DIM, Wout);
    alignas(16) int16 PT[16 * 16];
    BIAS_REP(Wo_t_b, E_DIM, bout);
    gemm_pk_bias<E_DIM, E_DIM, 16>(Wo_t.p, CT, PT, PIPE_ACC_SHIFT, Wo_t_b.r);
#if !defined(ATTN_TYPE_CROSS)
    alignas(16) int16 Rr[T_ROWS * E_DIM], RT[16 * 16];
    win_read_v<T_ROWS * E_DIM>(residual_in, Rr);
    to_T<T_ROWS>(Rr, RT);
    add_rows_v16<E_DIM>(PT, RT);
#endif
    LN_PARAMS(post_attn_ln_gamma_f, post_attn_ln_gamma, post_attn_ln_beta);
    ln_lanes(PT, post_attn_ln_gamma_f.g, post_attn_ln_gamma_f.b, LN_EPS_Q2);
    win_write_v<16 * 16>(proj_out, PT);
    aie::set_saturation(sat_save);
}
#endif

#if defined(POST_STAGE_B1) || defined(POST_STAGE_B2)
#if defined(POST_STAGE_B1)
void POST_B1_FN(input_window_int16* __restrict in_w, output_window_int16* __restrict out_w)
#else
void POST_B2_FN(input_window_int16* __restrict in_w, output_window_int16* __restrict out_w)
#endif
{
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    alignas(16) int16 XT[16 * 16], YT[16 * 16];
    win_read_v<16 * 16>(in_w, XT);
#if defined(POST_STAGE_B1)
    PACKED_WT(W_t, E_DIM, E_DIM, ffn_W0);
    BIAS_REP(W_t_b, E_DIM, ffn_b0);
    gemm_pk_bias<E_DIM, E_DIM, 16>(W_t.p, XT, YT, PIPE_ACC_SHIFT, W_t_b.r);
    LN_PARAMS(ffn_ln_gamma0_f, ffn_ln_gamma0, ffn_ln_beta0);
    ln_lanes(YT, ffn_ln_gamma0_f.g, ffn_ln_gamma0_f.b, LN_EPS_Q2);
#else
    PACKED_WT(W_t, E_DIM, E_DIM, ffn_W1);
    BIAS_REP(W_t_b, E_DIM, ffn_b1);
    gemm_pk_bias<E_DIM, E_DIM, 16>(W_t.p, XT, YT, PIPE_ACC_SHIFT, W_t_b.r);
    LN_PARAMS(ffn_ln_gamma1_f, ffn_ln_gamma1, ffn_ln_beta1);
    ln_lanes(YT, ffn_ln_gamma1_f.g, ffn_ln_gamma1_f.b, LN_EPS_Q2);
#endif
    relu_rows<E_DIM>(YT);
    win_write_v<16 * 16>(out_w, YT);
    aie::set_saturation(sat_save);
}
#endif

#if defined(POST_STAGE_C)
void POST_C_FN(input_window_int16* __restrict ffn_in,
                 input_window_int16* __restrict residual_b_in,
                 output_window_int16* __restrict x_out)
{
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    alignas(16) int16 XT[16 * 16], YT[16 * 16], RT[16 * 16];
    win_read_v<16 * 16>(ffn_in, XT);
    win_read_v<16 * 16>(residual_b_in, RT);
    PACKED_WT(W_t, E_DIM, E_DIM, ffn_W2);
    BIAS_REP(W_t_b, E_DIM, ffn_b2);
    gemm_pk_bias<E_DIM, E_DIM, 16>(W_t.p, XT, YT, PIPE_ACC_SHIFT, W_t_b.r);
    LN_PARAMS(ffn_ln_gamma2_f, ffn_ln_gamma2, ffn_ln_beta2);
    ln_lanes(YT, ffn_ln_gamma2_f.g, ffn_ln_gamma2_f.b, LN_EPS_Q2);
    relu_rows<E_DIM>(YT);
    add_rows_v16<E_DIM>(YT, RT);
    LN_PARAMS(post_ffn_ln_gamma_f, post_ffn_ln_gamma, post_ffn_ln_beta);
    ln_lanes(YT, post_ffn_ln_gamma_f.g, post_ffn_ln_gamma_f.b, LN_EPS_Q2);
    // back to row-major for the PL
    alignas(16) int16 Yr[T_ROWS * E_DIM];
    from_T<T_ROWS>(YT, Yr);
    win_write_v<T_ROWS * E_DIM>(x_out, Yr);
    aie::set_saturation(sat_save);
}
#endif
