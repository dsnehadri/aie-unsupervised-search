// memory budget per tile

// Wq, Wk, Wv: 3 x 64 x 2B = 384B
// bq, bk, bv: 3 x 4 x 2B = 24B
// bias_k, bias_v: 2 x 4 x 2B = 16B
// intermediate Q: 48 x 2B = 96B
// intermediate K: 64 x 2B = 128B
// intermediate V: 64 x 2B = 128B
// scores (float): 192 x 4B = 768B
// attn_weights: 192 x 2B = 384B
// input_window: 192 x 2B = 384B
// output_window: 48 x 2B = 96B
// total: 2.4KB

// GEMM strategy:
// using aie::mmul<4,4,4,int16,int16>tiles
// all dimensions are multiples of 4

// weight ROM: placeholder arrays below, replace with exported weights
// from export_weights_for_aie.py

#include "attn_head_kernel.h"
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <adf.h>


// object self attention weights

#if defined(ATTN_TYPE_OBJ)
    #if ATTN_LAYER == 0
        #if HEAD_IDX == 0
            #include "weights/obj_head0_weights_L0.h"
        #elif HEAD_IDX == 1
            #include "weights/obj_head1_weights_L0.h"
        #elif HEAD_IDX == 2
            #include "weights/obj_head2_weights_L0.h"
        #elif HEAD_IDX == 3
            #include "weights/obj_head3_weights_L0.h"
        #endif
    #elif ATTN_LAYER == 1
        #if HEAD_IDX == 0
            #include "weights/obj_head0_weights_L1.h"
        #elif HEAD_IDX == 1
            #include "weights/obj_head1_weights_L1.h"
        #elif HEAD_IDX == 2
            #include "weights/obj_head2_weights_L1.h"
        #elif HEAD_IDX == 3
            #include "weights/obj_head3_weights_L1.h"
        #endif
    #endif
#endif

#if defined(ATTN_TYPE_CAND)
    #if ATTN_LAYER == 0
        #if HEAD_IDX == 0
            #include "weights/cand_head0_weights_L0.h"
        #elif HEAD_IDX == 1
            #include "weights/cand_head1_weights_L0.h"
        #elif HEAD_IDX == 2
            #include "weights/cand_head2_weights_L0.h"
        #elif HEAD_IDX == 3
            #include "weights/cand_head3_weights_L0.h"
        #endif
    #elif ATTN_LAYER == 1
        #if HEAD_IDX == 0
            #include "weights/cand_head0_weights_L1.h"
        #elif HEAD_IDX == 1
            #include "weights/cand_head1_weights_L1.h"
        #elif HEAD_IDX == 2
            #include "weights/cand_head2_weights_L1.h"
        #elif HEAD_IDX == 3
            #include "weights/cand_head3_weights_L1.h"
        #endif
    #endif
#endif

#if defined(ATTN_TYPE_CROSS)
    #if ATTN_LAYER == 0
        #if HEAD_IDX == 0
            #include "weights/cross_head0_weights_L0.h"
        #elif HEAD_IDX == 1
            #include "weights/cross_head1_weights_L0.h"
        #elif HEAD_IDX == 2
            #include "weights/cross_head2_weights_L0.h"
        #elif HEAD_IDX == 3
            #include "weights/cross_head3_weights_L0.h"
        #endif
    #elif ATTN_LAYER == 1
        #if HEAD_IDX == 0
            #include "weights/cross_head0_weights_L1.h"
        #elif HEAD_IDX == 1
            #include "weights/cross_head1_weights_L1.h"
        #elif HEAD_IDX == 2
            #include "weights/cross_head2_weights_L1.h"
        #elif HEAD_IDX == 3
            #include "weights/cross_head3_weights_L1.h"
        #endif
    #endif
#endif

// titled gmem C[M][N] += A[M][K] x B[K][N]

template <int M, int K, int N>
inline void gemm_tile(const int16* __restrict A, const int16* __restrict B, int16* __restrict C, int shift)
{
    for (int m = 0; m < M; m += 4) {
        for (int n = 0; n < N; n+= 4) {
            aie::mmul<4, 4, 4, int16, int16> acc;
            acc.zero();

            for (int k = 0; k < K; k += 4) {
                aie::vector<int16, 16> va;
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        va[i * 4 + j] = A[(m+i) * K + (k+j)];
                    }
                }
                aie::vector<int16, 16> vb;
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 4; j++)
                        vb[j * 4 + i] = B[(k+i) * N + (n+j)];
                acc.mac(va, vb);
            }
            aie::vector<int16, 16> res = acc.to_vector<int16>(shift);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    C[(m+i) * N + (n+j)] =  res[i*4 + j];
        }
    }
}

// add bias to each row

template<int ROWS, int COLS>
inline void add_bias(int16* __restrict mat, const int16* __restrict bias)
{
    for (int r = 0; r < ROWS; r++) {
        aie::vector<int16, 4> row_vec = aie::load_v<4>(&mat[r * COLS]);
        aie::vector<int16, 4> bias_vec = aie::load_v<4>(bias);
        row_vec = aie::add(row_vec, bias_vec);
        aie::store_v(&mat[r * COLS], row_vec);
    }
}

// rowwise softmax in float
inline void softmax_row(const int16* __restrict scores_in, int16* __restrict attn_out, int n_rows, int n_cols, int n_cols_pad)
{
    float scores_f[N_MAX * N_KV_PAD];

    for (int r = 0; r < n_rows; r++) {
        float row_max = -1e30f;
        for (int c = 0; c < n_cols; c++) {
            float val = (float)scores_in[r * n_cols_pad + c] / DATA_SCALE;
            scores_f[r * n_cols_pad + c] = val;
            if (val > row_max) row_max = val;
        }
        for (int c = n_cols; c < n_cols_pad; c++) {
            scores_f[r * n_cols_pad + c] = -1e30f;
        }
        float sum = 0.0f;
        for (int c = 0; c < n_cols; c++) {
            float e = aie::exp(scores_f[r * n_cols_pad + c] - row_max);
            scores_f[r * n_cols_pad + c] = e;
            sum += e;
        }
        float inv_sum = 1.0f/sum;
        for (int c = 0; c < n_cols_pad; c++) {
            float val = (c < n_cols) ? scores_f[r * n_cols_pad + c]  * inv_sum : 0.0f;
            attn_out[r * n_cols_pad + c] = (int16)(val * DATA_SCALE);
        }
    }
}

// scale scores by 1/sqrt(d_head)
inline void scale_scores(int16* __restrict scores, int n_rows, int n_cols_pad, float inv_sqrt_d)
{
    int16 scale_fixed = (int16)(inv_sqrt_d * DATA_SCALE);
    for (int r = 0; r < n_rows; r++) {
        for (int c = 0; c < n_cols_pad; c++) {
            int32 product = (int32)scores[r * n_cols_pad + c] * (int32)scale_fixed;
            scores[r * n_cols_pad + c] = (int16)(product >> DATA_FRAC_BITS);
        }
    }
}

// object self attention kernel

#if defined(ATTN_TYPE_OBJ)
void obj_attn_head(input_window_int16* __restrict x_in, input_window_int16* __restrict wij_in, output_window_int16* __restrict x_out)
{
    // read input X[N_MAX][E_DIM]
    alignas(16) int16 X[N_MAX * E_DIM];
    for (int i = 0; i <N_MAX * E_DIM; i++)
        X[i] = window_readincr(x_in);

    // read wij_bias[N_MAX][N_KV] for this head
    alignas(16) int16 wij[N_MAX * N_KV];
    for (int i = 0; i < N_MAX * N_KV; i++)
        wij[i] = window_readincr(wij_in);

    // Q projection; Q[12][4] = X[12][16] x Wq[16][4] + bq
    alignas(16) int16 Q[N_MAX * D_HEAD];
    gemm_tile<N_MAX, E_DIM, D_HEAD>(X, Wq, Q, ACC_SHIFT);
    add_bias<N_MAX, D_HEAD>(Q, bq);

    // K projection: first 12 rows from X, then appeal bias_k_row
    // pad to N_KV_PAD=16 rows for aligned mmul
    alignas(16) int16 K[N_KV_PAD * D_HEAD] = {0};
    gemm_tile<N_MAX, E_DIM, D_HEAD>(X, Wk, K, ACC_SHIFT);
    add_bias<N_MAX, D_HEAD>(K, bk);
    for (int j = 0; j < D_HEAD; j++) {
        K[N_MAX * D_HEAD + j] = bias_k_row[j];
    }

    // V projection
    alignas(16) int16 V[N_KV_PAD * D_HEAD] = {0};
    gemm_tile<N_MAX, E_DIM, D_HEAD>(X, Wv, V, ACC_SHIFT);
    add_bias<N_MAX, D_HEAD>(V, bv);
    for (int j = 0; j < D_HEAD; j++) {
        V[N_MAX * D_HEAD + j] = bias_v_row[j];
    }

    // scores: S[12][16] = Q[12][4] x K^T[4][16]
    alignas(16) int16 Kt[D_HEAD * N_KV_PAD];
    for (int i = 0; i < N_KV_PAD; i++) {
        for (int j = 0; j < D_HEAD; j++) {
            Kt[j * N_KV_PAD + i] = K[i * D_HEAD + j];
        }
    }

    alignas(16) int16 scores[N_MAX * N_KV_PAD];
    gemm_tile<N_MAX, D_HEAD, N_KV_PAD>(Q, Kt, scores, ACC_SHIFT);

    // scale by 1/sqrt(D_HEAD) = 1/sqrt(4) = 0.5
    scale_scores(scores, N_MAX, N_KV_PAD, 0.5f);

    // add Wij bias (from pairwise MLP, computed on PL)
    for (int r = 0; r < N_MAX; r++) {
        for (int c = 0; c < N_KV; c++) {
            int32 sum = (int32)scores[r * N_KV_PAD + c] + (int32)wij[r*N_KV + c];
            if (sum > 32767) sum = 32767;
            if (sum < -32768) sum = -32768;
            scores[r * N_KV_PAD + c] = (int16)sum;
        }
    }

    // softmax (row-wise), in float
    alignas(16) int16 attn_weights[N_MAX * N_KV_PAD];
    softmax_row(scores, attn_weights, N_MAX, N_KV, N_KV_PAD);

    alignas(16) int16 head_out[N_MAX * D_HEAD];
    gemm_tile<N_MAX, N_KV_PAD, D_HEAD>(attn_weights, V, head_out, ACC_SHIFT);

    // write output
    for (int i = 0; i < N_MAX * D_HEAD; i++)
        window_writeincr(x_out, head_out[i]);
}
#endif

// candidate self attentiton head kernel

#if defined(ATTN_TYPE_CAND)

void cand_attn_head(input_window_int16* __restrict c_in, output_window_int16* __restrict c_out)
{
    // read C[T_DIM][E_DIM] = [3][16], pad to 4 rows for mmul alignment
    alignas(16) int16 C[4 * E_DIM] = {0};
    for (int r = 0; r <T_DIM; r++)
        for (int c = 0; c <E_DIM; c++)
            C[r * E_DIM + c] = window_readincr(c_in);

    // Q = C[4][16] x Wq[16][4], take first T_DIM=3 rows
    alignas(16) int16 Q[4 * D_HEAD];
    gemm_tile<4, E_DIM, D_HEAD>(C, cand_Wq, Q, ACC_SHIFT);
    add_bias<T_DIM, D_HEAD>(Q, cand_bq);

    // K projection: first 12 rows from X, then appeal bias_k_row
    // pad to N_KV_PAD=16 rows for aligned mmul
    alignas(16) int16 K[T_KV * D_HEAD] = {0};
    gemm_tile<4, E_DIM, D_HEAD>(C, cand_Wk, K, ACC_SHIFT);
    add_bias<T_DIM, D_HEAD>(K, cand_bk);
    for (int j = 0; j < D_HEAD; j++) {
        K[T_DIM * D_HEAD + j] = cand_bias_k_row[j];
    }

    // V projection
    alignas(16) int16 V[T_KV * D_HEAD];
    gemm_tile<4, E_DIM, D_HEAD>(C, cand_Wv, V, ACC_SHIFT);
    add_bias<T_DIM, D_HEAD>(V, cand_bv);
    for (int j = 0; j < D_HEAD; j++) {
        V[T_DIM * D_HEAD + j] = cand_bias_v_row[j];
    }

    // scores: Q[4][4] x K^T[4][4] -> [4][4]
    // T_KV = 4, D_HEAD = 4
    alignas(16) int16 Kt[D_HEAD * T_KV];
    for (int i = 0; i < T_KV; i++) {
        for (int j = 0; j < D_HEAD; j++) {
            Kt[j * T_KV + i] = K[i * D_HEAD + j];
        }
    }

    alignas(16) int16 scores[4 * T_KV];
    gemm_tile<4, D_HEAD, T_KV>(Q, Kt, scores, ACC_SHIFT);

    // scale by 1/sqrt(D_HEAD) = 1/sqrt(4) = 0.5
    scale_scores(scores, T_DIM, T_KV, 0.5f);

    // softmax (row-wise), in float
    alignas(16) int16 attn_w[4 * T_KV];
    softmax_row(scores, attn_w, T_DIM, T_KV, T_KV);

    alignas(16) int16 out[4 * D_HEAD];
    gemm_tile<4, T_KV, D_HEAD>(attn_w, V, out, ACC_SHIFT);

    // write output
    for (int r = 0; r < T_DIM; r++) {
        for (int c = 0; c < D_HEAD; c++) {
            window_writeincr(c_out, out[r * D_HEAD + c]);
        }
    }
}
#endif

// cross attention head kernel

#if defined(ATTN_TYPE_CROSS)

void cross_attn_head(input_window_int16* __restrict x_in, input_window_int16* __restrict c_in, output_window_int16* __restrict x_out)
{

    // read input X[N_MAX][E_DIM]
    alignas(16) int16 X[N_MAX * E_DIM];
    for (int i = 0; i <N_MAX * E_DIM; i++)
        X[i] = window_readincr(x_in);

    // read C[T_DIM][E_DIM] = [3][16], pad to 4 rows for mmul alignment
    alignas(16) int16 C[4 * E_DIM] = {0};
    for (int r = 0; r <T_DIM; r++)
        for (int c = 0; c <E_DIM; c++)
            C[r * E_DIM + c] = window_readincr(c_in);

    // Q = C[4][16] x Wq[16][4], take first T_DIM=3 rows
    alignas(16) int16 Q[N_MAX * D_HEAD];
    gemm_tile<N_MAX, E_DIM, D_HEAD>(X, cross_Wq, Q, ACC_SHIFT);
    add_bias<N_MAX, D_HEAD>(Q, cross_bq);

        // K projection: first 12 rows from X, then appeal bias_k_row
    // pad to N_KV_PAD=16 rows for aligned mmul
    alignas(16) int16 K[T_KV * D_HEAD];
    gemm_tile<4, E_DIM, D_HEAD>(C, cross_Wk, K, ACC_SHIFT);
    add_bias<T_DIM, D_HEAD>(K, cross_bk);
    for (int j = 0; j < D_HEAD; j++) {
        K[T_DIM * D_HEAD + j] = cross_bias_k_row[j];
    }

    // V projection
    alignas(16) int16 V[T_KV * D_HEAD];
    gemm_tile<4, E_DIM, D_HEAD>(C, cross_Wv, V, ACC_SHIFT);
    add_bias<T_DIM, D_HEAD>(V, cross_bv);
    for (int j = 0; j < D_HEAD; j++) {
        V[T_DIM * D_HEAD + j] = cross_bias_v_row[j];
    }

    // scores: Q[4][4] x K^T[4][4] -> [4][4]
    // T_KV = 4, D_HEAD = 4
    alignas(16) int16 Kt[D_HEAD * T_KV];
    for (int i = 0; i < T_KV; i++) {
        for (int j = 0; j < D_HEAD; j++) {
            Kt[j * T_KV + i] = K[i * D_HEAD + j];
        }
    }

    alignas(16) int16 scores[N_MAX * T_KV];
    gemm_tile<N_MAX, D_HEAD, T_KV>(Q, Kt, scores, ACC_SHIFT);

    // scale by 1/sqrt(D_HEAD) = 1/sqrt(4) = 0.5
    scale_scores(scores, N_MAX, T_KV, 0.5f);

    // softmax (row-wise), in float
    alignas(16) int16 attn_w[N_MAX * T_KV];
    softmax_row(scores, attn_w, N_MAX, T_KV, T_KV);

    alignas(16) int16 out[N_MAX * D_HEAD];
    gemm_tile<N_MAX, T_KV, D_HEAD>(attn_w, V, out, ACC_SHIFT);

    // write output
    for (int i = 0; i < N_MAX * D_HEAD; i++)
        window_writeincr(x_out, out[i]);
}
#endif

