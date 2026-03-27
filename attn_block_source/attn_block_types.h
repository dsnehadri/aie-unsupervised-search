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

typedef ap_fixed<32, 12> data_t;
typedef ap_fixed<32, 12> weight_t;
typedef ap_fixed<32, 12> score_t;
typedef ap_fixed<32, 12> prob_t;
typedef ap_fixed<32, 12> ln_param_t;
typedef ap_fixed<64, 20> acc_t;
typedef ap_fixed<64, 20> exp_t;

// scaling constant = 1/sqrt(D_HEAD)

static const score_t SCALE = 0.5;

// large negative values for masked positions (saturates softmax to ~0)

static const score_t NEG_INF = -64.0;

// layer norm epsilon

static const float LN_EPS = 1e-5f;


// for softmax lookup table for exp() in fixed point

static const int EXP_LUT_SIZE = 256;
static const float EXP_MIN = -8.0f; // exp(-8) ~= 0.00034

#endif