#ifndef ATTN_BLOCK_TYPES_H
#define ATTN_BLOCK_TYPES_H

#include "ap_fixed.h"
#include "hls_math.h"

// model constants from passwd architecture

static const int N_MAX = 12; // max jets
static const int E_DIM = 16; // embeddings dimension
static const int N_HEADS = 4; // attention heads
static const int D_HEAD = E_DIM / N_HEADS; // per-head dimension
static const int N_KV = N_MAX + 1; // KV length with +1 for bias_kv
static const int T_DIM = 3; // number of categories
static const int T_KV = T_DIM + 1; // to account for bias_kv token
static const int AE_IN_DIM = E_DIM - T_DIM + 1; // 16 - 3 + 1 = 14
static const int AE_DIM = 2;

// encoder cascade_dims(14, 2, 4) =  [14, 11, 8, 5, 2];

static const int AE_D0 = AE_IN_DIM; // 14
static const int AE_D1 = 11; // 
static const int AE_D2 = 8;
static const int AE_D3 = 5;
static const int AE_D4 = AE_DIM; //2


// ffn inside attention block, 3 layers for Linear+LN+ReLu

static const int N_FFN_LAYERS = 3;
static const int FFN_DIM = E_DIM;

// fixed points types

// general data path for embeddings, residuals, FFN activations

// typedef ap_fixed<16, 5> data_t;

// // weights and biases

// typedef ap_fixed<16, 4> weight_t;

// // attention scores

// typedef ap_fixed<16, 6> score_t;

// // post softmax probabilities

// typedef ap_fixed<16, 2> prob_t;

// // layernorm parameters

// typedef ap_fixed<16, 4> ln_param_t;

// // accumulator type for dot products

// typedef ap_fixed<32, 10> acc_t;

// // softmax intermediate

// typedef ap_fixed<32, 10> exp_t;

// typedef ap_fixed<32, 12> data_t;
// typedef ap_fixed<32, 12> weight_t;
// typedef ap_fixed<32, 12> score_t;
// typedef ap_fixed<32, 12> prob_t;
// typedef ap_fixed<32, 12> ln_param_t;
// typedef ap_fixed<64, 20> acc_t;
// typedef ap_fixed<64, 20> exp_t;

#ifdef AIE_FRAC11
// Original 11-frac contract (validated June hybrid config, git d1830be).
// The AIE attention kernels and their sliced weights are defined on this:
// data_t bit-pattern == int16 at DATA_SCALE=2048 (see attn_aie_types.h and
// slice_weights_for_aie.py). Any build whose PL side bridges raw bits to the
// AIE MUST use these types + 11-frac input packing; building the bridge with
// the <16,7> types below skews every obj/cross activation 4x (measured
// median +25% / p90 306% end-to-end error vs the intended function).
typedef ap_fixed<16, 5> data_t;
typedef ap_fixed<16, 4> weight_t;
typedef ap_fixed<16, 6> score_t;
typedef ap_fixed<16, 2> prob_t;
typedef ap_fixed<16, 4> ln_param_t;
typedef ap_fixed<32, 10> acc_t;
typedef ap_fixed<32, 10> exp_t;
#else
// Retrained all-PL contract (wider integer range for the retrained weights).
typedef ap_fixed<16, 7> data_t;
typedef ap_fixed<16, 4> weight_t;
typedef ap_fixed<16, 11> score_t;   // widened integer bits: cand Q*K^T can reach ~320; ±32 was saturating
typedef ap_fixed<16, 2> prob_t;
typedef ap_fixed<16, 4> ln_param_t;
typedef ap_fixed<32, 10> acc_t;
typedef ap_fixed<32, 10> exp_t;
#endif

// scaling constant = 1/sqrt(D_HEAD)

static const score_t SCALE = 0.5;

// large negative values for masked positions (saturates softmax to ~0)

// score_t = ap_fixed<16, 6> has range [-32, 32). Using -64 wraps to 0!
// Use the smallest representable score_t value so exp(NEG_INF - max) underflows
// for all reasonable max scores.
static const score_t NEG_INF = -31.0;

// layer norm epsilon

static const float LN_EPS = 1e-5f;


// for softmax lookup table for exp() in fixed point

static const int EXP_LUT_SIZE = 256;
static const float EXP_MIN = -8.0f; // exp(-8) ~= 0.00034

#endif