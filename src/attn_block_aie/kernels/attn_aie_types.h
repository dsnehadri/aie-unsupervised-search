// attn_aie_types.h

#ifndef ATTN_AIE_TYPES_H
#define ATTN_AIE_TYPES_H

#include <aie_api/aie.hpp>
#include <adf.h>
    
// architectural constants 

constexpr int N_MAX = 12;
constexpr int E_DIM = 16;
constexpr int N_HEADS = 4;
constexpr int D_HEAD = 4;
constexpr int N_KV = 13;
constexpr int T_DIM = 3;
constexpr int T_KV = 4;

// ffn dimensions: 3-layer ffn after attention, all 16-wide

constexpr int FFN_NLAYERS = 3;
constexpr int FFN_DIM = E_DIM;

// fixed point configuration

// datapath: ap_fixed<16, 5> equivalent -> Q4.11 in AIE terms
// weight path: ap_fixed<16, 4> equivalent -> Q3.12
// accumulator 32-bit

// for aie int16 mmul, accumulator is 48 bit internally, then we right shift to extract int16 result.
// slice_weights_for_aie.py exports weights at the SAME scale as data (11 frac bits, 2048),
// so both gemm inputs are at 2^11. Product has 22 frac bits; shift right by 11 to get back
// to 11 frac bits in the output. Previously WEIGHT_FRAC_BITS was 12 here (mismatched the
// export) which halved every gemm's output magnitude and compounded into ~2.4 max_err.

constexpr int DATA_FRAC_BITS = 11;
constexpr int WEIGHT_FRAC_BITS = 11;
constexpr int ACC_SHIFT = WEIGHT_FRAC_BITS;

// scale factor for converting from float to fixed

constexpr float DATA_SCALE = (float)(1 << DATA_FRAC_BITS); // 2048
constexpr float WEIGHT_SCALE = (float)(1 << WEIGHT_FRAC_BITS); // 2048

// Cand pipeline uses a wider integer range (Q6.9, scale 512) because
// cand_build produces unnormalized sums up to ~|19| that overflow Q4.11.
// ALL cand-path data, weights, biases, and LN params are quantized at this
// scale; cand_attn_head_* and cand_post_* use CAND_ACC_SHIFT in their gemms
// and CAND_SCALE in softmax/LN. Output of cand attention is at CAND_SCALE,
// so check_attn_outputs.py reads it at /512.
constexpr int CAND_FRAC_BITS = 9;
constexpr int CAND_ACC_SHIFT = CAND_FRAC_BITS;
constexpr float CAND_SCALE = (float)(1 << CAND_FRAC_BITS); // 512

// Cand Q*K^T can reach magnitudes ~50 (large cand inputs), saturating int16
// at CAND_SCALE=512. Store scores at a smaller scale (Q11.5, scale 32) so
// the post-shift int16 fits (~+/-1024 representable). PL solves the same
// issue by widening score_t to ap_fixed<16,11>.
constexpr int CAND_SCORE_FRAC_BITS = 5;
constexpr int CAND_SCORE_SHIFT = CAND_SCORE_FRAC_BITS; // for scale_scores multiply
constexpr float CAND_SCORE_SCALE = (float)(1 << CAND_SCORE_FRAC_BITS); // 32
// Q*Kt gemm shift: Q,K at CAND_SCALE; want result at CAND_SCORE_SCALE.
// shift = log2(CAND_SCALE * CAND_SCALE / CAND_SCORE_SCALE) = log2(512^2/32) = 13
constexpr int CAND_QKT_SHIFT = CAND_FRAC_BITS + CAND_FRAC_BITS - CAND_SCORE_FRAC_BITS; // 13



// buffer sizes (in int16 elements)

// per head QKV projection weights: [E_DIM x D_HEAD] each
constexpr int QKV_WEIGHT_SIZE = E_DIM * D_HEAD; // 64 per matrix

// per head bias_kv, 2 x D_HEAD (one for K bias row, one for V bias row)
constexpr int BIAS_KV_SIZE = 2 * D_HEAD; // 8

// output projection weights: [E_DIM x E_DIM]
constexpr int OUT_PROJ_WEIGHT_SIZE = E_DIM * E_DIM; // 256

// attn input window: N_MAX x E_DIM
constexpr int ATTN_INPUT_SIZE = N_MAX * E_DIM; // 192

// per head intermediate buffers
constexpr int Q_SIZE = N_MAX * D_HEAD; // 48
constexpr int K_SIZE = N_KV * D_HEAD; // 52
constexpr int V_SIZE = N_KV * D_HEAD; // 52
constexpr int SCORE_SIZE = N_MAX * N_KV; // 156 (padding to 160 for alignment)
constexpr int HEAD_OUT_SIZE = N_MAX * D_HEAD; //48

// full attention output: N_MAX x E_DIM
constexpr int ATTN_OUTPUT_SIZE = N_MAX * E_DIM; // 192

// ffn weight sizes per layer
constexpr int FFN_WEIGHT_SIZE = E_DIM * E_DIM; // 256 per layer
constexpr int FFN_BIAS_SIZE = E_DIM; // 16 per layer

// layernorm params
constexpr int LN_PARAM_SIZE = E_DIM; // 16 (gamma) + 16 (beta)


// padding helpers
// aie mmul type is 4x4x4 for int16, dimensions should be multiples of 4
// n_max = 12, edim = 16, d_head = 4
// n_kv = 13, pad to 16 for k, v matrices

constexpr int N_KV_PAD = 16;

// wij_bias = [N_MAX x N_KV] per head
// in pl this was [N_HEADS x N_MAX x N_KV], in aie each head gets its own slice

constexpr int WIJ_SIZE = N_MAX * N_KV;

#endif