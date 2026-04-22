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

// for aie int16 mmul, accumualtor is 48 bit internally, then we right shfit to extract int16 result
// fractional bits: data = 11, weight = 12 -> product has 23 frac bits
// to get back to 11 frac bits in output, shift right by 12

constexpr int DATA_FRAC_BITS = 11;
constexpr int WEIGHT_FRAC_BITS = 12;
constexpr int ACC_SHIFT = WEIGHT_FRAC_BITS;

// scale factor for converting from float to fixed

constexpr float DATA_SCALE = (float)(1 << DATA_FRAC_BITS); // 2048
constexpr float WEIGHT_SCALE = (float)(1 << WEIGHT_FRAC_BITS); // 4096

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