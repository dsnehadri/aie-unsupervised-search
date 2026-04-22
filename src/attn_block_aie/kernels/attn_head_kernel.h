//single attention head kernel for aie

// each head runs on its own aie kernel, computing
// Q = X @ Wq + bq [12 x 16] * [16 x 4] = [12 x 4]
// K = [X; bias_k] @ Wk + bk [13 x 16] * [16 x 4] = [13 x 4]
// V = [X; bias_v] @ Wv + bv [13 x 16] * [16 x 4] = [13 x 4]
// scores = Q @ K^T / sqrt(d) [12 x 4] x [4 x 13] = [12  x 13]
// scores += wij_bias
// attn_w = softmax(scores)
// head_out = attn_w @ V [12 x 13] x [13 x 4] = [12  x 4]

// weights are stored as ROM in tile-local data memory
// input: X[N_MAX][E_DIM] as int16 window
// output: head_out[N_MAX][D_HEAD] as int16 window

// wij is streamed in from PL (computed by pairwise MLP on PL)

#ifndef ATTN_HEAD_KERNEL_H
#define ATTN_HEAD_KERNEL_H

#include "attn_aie_types.h"

// kernel function declarations

// object self-attention head (Q=K=V=X, bias_kv)
// x_in: [N_MAX x E_DIM] int16 - embedded jet features
// wij_in: [N_MAX x N_KV] int16 - pairwise MLP bias for this head
// x_out: [N_MAX x D_HEAD] int16 - per-head attention output

void obj_attn_head(input_window_int16* __restrict x_in, input_window_int16* __restrict wij_in, output_window_int16* __restrict x_out);

// candidate self-attention head (Q=K=V=C, no bias_kv for simplicity)
// c_in: [T_DIM x E_DIM] int16
// c_out: [T_DIM x D_HEAD] int16
void cand_attn_head(input_window_int16* __restrict c_in, output_window_int16* __restrict c_out);

// cross attention head (Q=X, K=V=C)
// x_in: [N_MAX x E_DIM] int16 
// c_in: [T_DIM x E_DIM] int16 
// x_out: [N_MAX x D_HEAD] int16

void cross_attn_head(input_window_int16* __restrict x_in, input_window_int16* __restrict c_in, output_window_int16* __restrict x_out);



#endif