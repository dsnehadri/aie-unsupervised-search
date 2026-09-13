// Transposed head kernels (see transposed.h). Included by attn_head_kernel.cc
// when TRANSPOSED is defined; same function names and window ports, but the
// scores window carries S^T (keys x query lanes), the V window carries V^T
// (d_head x keys, packed for the AV gemm) and the head output is O^T
// (d_head x lanes). Window sizes for these ports change in aie_graph.h.
#include "transposed.h"

static constexpr float LN_EPS_Q2 = 1e-5f * (float)PIPE_SCALE * (float)PIPE_SCALE;

// ===================================================================== object
#if defined(ATTN_TYPE_OBJ)
#if defined(HEAD_STAGE_PRE)
void HEAD_PRE_FN(input_window_int16* __restrict x_in,
                       output_window_int16* __restrict scores_out,
                       output_window_int16* __restrict v_out)
{
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    // X row-major (12 rows) + key-padding mask row -> X^T (16 x 16, lanes >= 12 zero)
    alignas(16) int16 Xr[N_MAX * E_DIM];
    win_read_v<N_MAX * E_DIM>(x_in, Xr);
    alignas(16) int16 kmask[E_DIM];
    aie::store_v(kmask, win_read16(x_in));
    alignas(16) int16 XT[16 * 16];
    to_T<N_MAX>(Xr, XT);

    PACKED_WT(Wq_t, D_HEAD, E_DIM, Wq);
    PACKED_WT(Wk_t, D_HEAD, E_DIM, Wk);
    PACKED_WT(Wv_t, D_HEAD, E_DIM, Wv);
    alignas(16) int16 QT[D_HEAD * 16], KT[D_HEAD * 16], VT[D_HEAD * 16];
    gemm_pk<D_HEAD, E_DIM, 16>(Wq_t.p, XT, QT, PIPE_ACC_SHIFT); bias_rows<D_HEAD>(QT, bq);
    gemm_pk<D_HEAD, E_DIM, 16>(Wk_t.p, XT, KT, PIPE_ACC_SHIFT); bias_rows<D_HEAD>(KT, bk);
    gemm_pk<D_HEAD, E_DIM, 16>(Wv_t.p, XT, VT, PIPE_ACC_SHIFT); bias_rows<D_HEAD>(VT, bv);
    // key lane N_MAX is the learned bias key/value; lanes beyond it are zero
    for (int d = 0; d < D_HEAD; d++) {
        KT[d * 16 + N_MAX] = bias_k_row[d]; VT[d * 16 + N_MAX] = bias_v_row[d];
        for (int j = N_MAX + 1; j < 16; j++) { KT[d * 16 + j] = 0; VT[d * 16 + j] = 0; }
    }
    // S^T (keys x query lanes) = K (keys x d, row-major == packed for K=4) . Q^T
    alignas(16) int16 Kp[16 * D_HEAD];
    for (int j = 0; j < 16; j++)
        for (int d = 0; d < D_HEAD; d++) Kp[j * D_HEAD + d] = KT[d * 16 + j];
    alignas(16) int16 ST[16 * 16];
    gemm_pk<16, D_HEAD, 16>(Kp, QT, ST, PIPE_QKT_SHIFT);
    scale_rows<16>(ST, (int16)(0.5f * PIPE_SCORE_SCALE), PIPE_SCORE_SHIFT);
    // padded keys: whole key row to -32000 (softmax -> exactly 0)
    const v16s neg = aie::broadcast<int16, 16>(-32000);
    for (int j = 0; j < N_MAX; j++)
        if (kmask[j] != 0) aie::store_v(ST + j * 16, neg);

    win_write_v<16 * 16>(scores_out, ST);
    win_write_v<D_HEAD * 16>(v_out, VT);
    aie::set_saturation(sat_save);
}
#endif

#if defined(HEAD_STAGE_POST)
#if ATTN_LAYER == 0
void HEAD_POST_FN(input_window_int16* __restrict scores_in,
                        input_window_int16* __restrict v_in,
                        input_window_int16* __restrict wij_in,
                        output_window_int16* __restrict x_out)
#else
void HEAD_POST_FN(input_window_int16* __restrict scores_in,
                        input_window_int16* __restrict v_in,
                        output_window_int16* __restrict x_out)
#endif
{
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    alignas(16) int16 ST[16 * 16];
    win_read_v<16 * 16>(scores_in, ST);
    alignas(16) int16 VT[D_HEAD * 16];
    win_read_v<D_HEAD * 16>(v_in, VT);
#if ATTN_LAYER == 0
    // wij arrives query-major (N_MAX x N_KV); build its transpose fully with
    // scalar stores, then add row vectors (aligned loads only)
    alignas(16) int16 WT[16 * 16];
    zero_v<16 * 16>(WT);
    for (int q = 0; q < N_MAX; q++)
        for (int k = 0; k < N_KV; k++)
            WT[k * 16 + q] = window_readincr(wij_in);
    for (int k = 0; k < N_KV; k++)
        aie::store_v(ST + k * 16, add_sat16(aie::load_v<16>(ST + k * 16), aie::load_v<16>(WT + k * 16)));
#endif
    alignas(16) int16 PT[16 * 16];
    softmax_lanes<N_KV, 16>(ST, PT, (float)PIPE_SCORE_SCALE, (float)PIPE_SCALE);
    // O^T (d x lanes) = V^T (d x keys, packed) . P^T (keys x lanes)
    alignas(16) int16 Vp[D_HEAD * 16];
    pack_rows4(aie::load_v<16>(VT), aie::load_v<16>(VT + 16), aie::load_v<16>(VT + 32), aie::load_v<16>(VT + 48), Vp);
    alignas(16) int16 OT[D_HEAD * 16];
    gemm_pk<D_HEAD, 16, 16>(Vp, PT, OT, PIPE_AV_SHIFT);
    win_write_v<D_HEAD * 16>(x_out, OT);
    aie::set_saturation(sat_save);
}
#endif
#endif // ATTN_TYPE_OBJ

// ============================================== candidate (3 lanes) and cross
// Both attend to the T_DIM candidates (+ bias key = T_KV keys). The candidate
// block's queries are the candidates themselves, the cross block's queries are
// the N_MAX jets. Keys/values come from C^T in both.
#if defined(ATTN_TYPE_CAND) || defined(ATTN_TYPE_CROSS)
#if defined(ATTN_TYPE_CAND)
#define T_Wq cand_Wq
#define T_Wk cand_Wk
#define T_Wv cand_Wv
#define T_bq cand_bq
#define T_bk cand_bk
#define T_bv cand_bv
#define T_bias_k cand_bias_k_row
#define T_bias_v cand_bias_v_row
#else
#define T_Wq cross_Wq
#define T_Wk cross_Wk
#define T_Wv cross_Wv
#define T_bq cross_bq
#define T_bk cross_bk
#define T_bv cross_bv
#define T_bias_k cross_bias_k_row
#define T_bias_v cross_bias_v_row
#endif

// keys/values from C (row-major T_DIM x 16): K (T_KV x d, packed) and V^T (d x T_KV, packed)
static inline void kv_from_c(input_window_int16* __restrict c_in, int16* __restrict Kp, int16* __restrict Vp)
{
    alignas(16) int16 Cr[T_DIM * E_DIM];
    win_read_v<T_DIM * E_DIM>(c_in, Cr);
    alignas(16) int16 CT[16 * 16];
    to_T<T_DIM>(Cr, CT);
    PACKED_WT(Wk_t, D_HEAD, E_DIM, T_Wk);
    PACKED_WT(Wv_t, D_HEAD, E_DIM, T_Wv);
    alignas(16) int16 KT[D_HEAD * 16], VT[D_HEAD * 16];
    gemm_pk<D_HEAD, E_DIM, 16>(Wk_t.p, CT, KT, PIPE_ACC_SHIFT); bias_rows<D_HEAD>(KT, T_bk);
    gemm_pk<D_HEAD, E_DIM, 16>(Wv_t.p, CT, VT, PIPE_ACC_SHIFT); bias_rows<D_HEAD>(VT, T_bv);
    for (int d = 0; d < D_HEAD; d++) {
        KT[d * 16 + T_DIM] = T_bias_k[d]; VT[d * 16 + T_DIM] = T_bias_v[d];
        for (int k = 0; k < T_KV; k++) { Kp[k * D_HEAD + d] = KT[d * 16 + k]; Vp[d * T_KV + k] = VT[d * 16 + k]; }
    }
}

#if defined(HEAD_STAGE_PRE)
#if defined(ATTN_TYPE_CAND)
void HEAD_PRE_FN(input_window_int16* __restrict c_in,
                        output_window_int16* __restrict scores_out,
                        output_window_int16* __restrict v_out)
#else
void HEAD_PRE_FN(input_window_int16* __restrict x_in,
                         input_window_int16* __restrict c_in,
                         output_window_int16* __restrict scores_out,
                         output_window_int16* __restrict v_out)
#endif
{
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    // queries: candidates (cand) or jets (cross)
    alignas(16) int16 QsrcT[16 * 16];
#if defined(ATTN_TYPE_CAND)
    // the candidate window is read twice (queries, then keys/values): copy it
    alignas(16) int16 Cr[T_DIM * E_DIM];
    win_read_v<T_DIM * E_DIM>(c_in, Cr);
    to_T<T_DIM>(Cr, QsrcT);
#else
    alignas(16) int16 Xr[N_MAX * E_DIM];
    win_read_v<N_MAX * E_DIM>(x_in, Xr);
    to_T<N_MAX>(Xr, QsrcT);
#endif
    PACKED_WT(Wq_t, D_HEAD, E_DIM, T_Wq);
    alignas(16) int16 QT[D_HEAD * 16];
    gemm_pk<D_HEAD, E_DIM, 16>(Wq_t.p, QsrcT, QT, PIPE_ACC_SHIFT); bias_rows<D_HEAD>(QT, T_bq);

    alignas(16) int16 Kp[T_KV * D_HEAD], Vp[D_HEAD * T_KV];
#if defined(ATTN_TYPE_CAND)
    {   // keys/values from the same C
        alignas(16) int16 CT2[16 * 16];
        to_T<T_DIM>(Cr, CT2);
        PACKED_WT(Wk_t, D_HEAD, E_DIM, T_Wk);
        PACKED_WT(Wv_t, D_HEAD, E_DIM, T_Wv);
        alignas(16) int16 KT[D_HEAD * 16], VT[D_HEAD * 16];
        gemm_pk<D_HEAD, E_DIM, 16>(Wk_t.p, CT2, KT, PIPE_ACC_SHIFT); bias_rows<D_HEAD>(KT, T_bk);
        gemm_pk<D_HEAD, E_DIM, 16>(Wv_t.p, CT2, VT, PIPE_ACC_SHIFT); bias_rows<D_HEAD>(VT, T_bv);
        for (int d = 0; d < D_HEAD; d++) {
            KT[d * 16 + T_DIM] = T_bias_k[d]; VT[d * 16 + T_DIM] = T_bias_v[d];
            for (int k = 0; k < T_KV; k++) { Kp[k * D_HEAD + d] = KT[d * 16 + k]; Vp[d * T_KV + k] = VT[d * 16 + k]; }
        }
    }
#else
    kv_from_c(c_in, Kp, Vp);
#endif
    // S^T (T_KV keys x 16 lanes) = K (T_KV x d) . Q^T (d x 16)
    alignas(16) int16 ST[T_KV * 16];
    gemm_pk<T_KV, D_HEAD, 16>(Kp, QT, ST, PIPE_QKT_SHIFT);
    scale_rows<T_KV>(ST, (int16)(0.5f * PIPE_SCORE_SCALE), PIPE_SCORE_SHIFT);
    win_write_v<T_KV * 16>(scores_out, ST);
    win_write_v<D_HEAD * T_KV>(v_out, Vp);
    aie::set_saturation(sat_save);
}
#endif // HEAD_STAGE_PRE

#if defined(HEAD_STAGE_POST)
void HEAD_POST_FN(input_window_int16* __restrict scores_in,
                         input_window_int16* __restrict v_in,
                         output_window_int16* __restrict c_out)
{
    const aie::saturation_mode sat_save = aie::swap_saturation(aie::saturation_mode::saturate);
    alignas(16) int16 ST[T_KV * 16];
    win_read_v<T_KV * 16>(scores_in, ST);
    alignas(16) int16 Vp[D_HEAD * T_KV];
    win_read_v<D_HEAD * T_KV>(v_in, Vp);
    alignas(16) int16 PT[T_KV * 16];
    softmax_lanes<T_KV, T_KV>(ST, PT, (float)PIPE_SCORE_SCALE, (float)PIPE_SCALE);
    alignas(16) int16 OT[D_HEAD * 16];
    gemm_pk<D_HEAD, T_KV, 16>(Vp, PT, OT, PIPE_AV_SHIFT);     // V^T (d x keys, packed==row-major) . P^T
    win_write_v<D_HEAD * 16>(c_out, OT);
    aie::set_saturation(sat_save);
}
#endif // HEAD_STAGE_POST
#endif // CAND || CROSS
