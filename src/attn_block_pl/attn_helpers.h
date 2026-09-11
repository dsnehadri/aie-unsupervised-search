#ifndef ATTN_HELPERS_H
#define ATTN_HELPERS_H

#include "attn_block_types.h"

// computes y = x @ W^T + bias

// template allows for compile-time sizing

template <int N_ROWS, int FEAT_DIM = E_DIM>
void relu_2d(data_t x[N_ROWS][FEAT_DIM]) {
    for (int i = 0; i < N_ROWS; i++) {
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < FEAT_DIM; j++) {
            if (x[i][j] < (data_t)0) x[i][j] = (data_t)0;
        } 
    }
}

template <int N_ROWS, int OUT_DIM = E_DIM, int IN_DIM = E_DIM>
void linear(
    const data_t in[N_ROWS][IN_DIM],
    const weight_t W[OUT_DIM][IN_DIM],
    const weight_t bias[OUT_DIM],
    data_t out[N_ROWS][OUT_DIM]
) {

    // Partition along k so the unrolled LIN_K MAC can read operands in
    // parallel. Without this, W/in sit in 2-port BRAM and the written II=1
    // silently degrades to ~IN_DIM/2 (same port-contention failure diagnosed
    // in pairwise_mlp.h — the fix was never applied here). factor=8 (II=2,
    // 8 MACs/cycle) rather than complete (II=1, 16 MACs): complete costs
    // ~16 DSP per linear x 42 linears and blew the csynth DSP estimate to
    // 107% of the device; factor=8 keeps ~4x the old throughput at half
    // the multipliers.
    // LIN_PARTITION lets the k-partitioning be swept without editing here:
    // 8 (default) = 8 MACs/cycle at II=2; 16 = the full row, II=1, ~2x the
    // multipliers. The csynth DSP estimate runs about 1.5x the implemented
    // number, so the "107% of device" that ruled out complete partitioning
    // was an estimate, not a placement result.
// NB: the preprocessor does not expand macros inside #pragma, so the factor
// has to go through _Pragma with stringification or the sweep silently does
// nothing (both settings then synthesize identically).
#ifndef LIN_PARTITION
#define LIN_PARTITION 8
#endif
#define LIN_DO_PRAGMA(x) _Pragma(#x)
#define LIN_PART_K(v, f) LIN_DO_PRAGMA(HLS ARRAY_PARTITION variable=v dim=2 cyclic factor=f)
    LIN_PART_K(W, LIN_PARTITION)
    LIN_PART_K(in, LIN_PARTITION)
#ifdef LIN_J_UNROLL
    // The linear already runs at II=1 over its N_ROWS x OUT_DIM outputs (192
    // iterations -> 195 cycles for 12x16), so widening the k-partition cannot
    // help; the only way down is several outputs per cycle. LIN_J_UNROLL = F
    // computes F outputs per iteration, F x the multipliers. Same two-level
    // _Pragma indirection as LIN_PART_K, or the factor is not expanded.
    #define LIN_PART_J(v, f)   LIN_DO_PRAGMA(HLS ARRAY_PARTITION variable=v dim=1 cyclic factor=f)
    #define LIN_PART_OUT(v, f) LIN_DO_PRAGMA(HLS ARRAY_PARTITION variable=v dim=2 cyclic factor=f)
    #define LIN_PART_B(v, f)   LIN_DO_PRAGMA(HLS ARRAY_PARTITION variable=v cyclic factor=f)
    #define LIN_UNROLL_J(f)    LIN_DO_PRAGMA(HLS UNROLL factor=f)
    LIN_PART_J(W, LIN_J_UNROLL)
    LIN_PART_OUT(out, LIN_J_UNROLL)
    LIN_PART_B(bias, LIN_J_UNROLL)
#endif
    LIN_I:
    for (int i = 0; i < N_ROWS; i++) {
        LIN_J:
        for (int j=0; j < OUT_DIM; j++) {
            #pragma HLS PIPELINE II=1
#ifdef LIN_J_UNROLL
            LIN_UNROLL_J(LIN_J_UNROLL)
#endif
            acc_t sum = (acc_t) bias[j];
            LIN_K:
            for (int k=0; k < IN_DIM; k++) {
                #pragma HLS UNROLL
                // The design is DSP-bound at 93% while LUTs sit near 14%, so the
                // multiplier budget, not logic, is what blocks running the four
                // attention heads concurrently. LIN_FABRIC_MUL moves these
                // products into LUT fabric to trade the plentiful resource for
                // the scarce one.
#ifdef LIN_FABRIC_MUL
                #pragma HLS BIND_OP variable=sum op=mul impl=fabric
#endif
#ifdef NARROW_MUL
                // Multiply at the operands' own widths. Casting both to acc_t
                // first made HLS build a 32x32 multiplier for a 16x16 product:
                // 2 DSPs instead of 1, ~4x the LUTs in fabric. The exact
                // product is the same number and is rounded once, at the add,
                // in both forms, so this is bit-identical (native sim checked).
                sum += in[i][k] * W[j][k];
#else
                sum += (acc_t)in[i][k] * (acc_t)W[j][k];
#endif
            }
            out[i][j] = (data_t)sum;
        }
    }
}

// normalizes each row to have mean 0 and variance 1

// LN_MODE selects the layer-norm implementation, so the three can be compared
// by synthesis rather than by argument:
//   0  as built: float32 statistics, row loop NOT pipelined (660 cycles/12 rows)
//   1  same arithmetic, row loop pipelined
//   2  integer statistics with the reciprocal-square-root table, row pipelined
//   3  integer, row loop pipelined at II=4 so the element multipliers are
//      shared 4 ways (PIPELINE II=1 fully unrolls inner loops whatever the
//      UNROLL factor says, which is why 16 wide multipliers appeared)
//   4  integer, row loop NOT pipelined -- the integer chain is far shorter
//      than the float one, and this matches 2 exactly, so the whole win is the
//      arithmetic, not the pipelining
//   5  as 4 but the element loops unrolled by 4, to share the wide multipliers
//      (only possible without PIPELINE on the row loop, which force-unrolls)
//   6  as 2 (row pipelined at II=1, 15 cycles per layer norm against 372 for
//      mode 5) with the element multipliers bound to LUT fabric. Mode 2 was
//      ruled out for costing +153 DSP per block; the device has 130 spare and
//      1.55M idle LUTs, so this buys mode 2's latency with the plentiful
//      resource instead of the scarce one. A latency lever: 14 layer norms sit
//      on the one-event chain.
// The block spends about 36% of its cycles here, 3.4x the linear layer it
// follows, so this is where the fabric design's headroom is.
#ifndef LN_MODE
#define LN_MODE 0
#endif

// eps at the integer scale: 2^30 * 1e-5, matching var = V / 2^30
#define LN_EPSV 10737
#if LN_MODE == 5
#define LN_UNROLL_PRAGMA LIN_DO_PRAGMA(HLS UNROLL factor=4)
#else
#define LN_UNROLL_PRAGMA _Pragma("HLS UNROLL")
#endif
#if LN_MODE >= 2
typedef int int32;                       // the table is shared with the AI Engine kernel
#include "../attn_block_aie/kernels/ln_rsqrt_lut.h"
#endif

template <int N_ROWS, int FEAT_DIM = E_DIM>
void layernorm(
    data_t x[N_ROWS][FEAT_DIM],
    const ln_param_t gamma[FEAT_DIM],
    const ln_param_t beta[FEAT_DIM]

) {
    // feed the unrolled LN_MEAN/LN_VAR reductions (FEAT_DIM reads/cycle)
    #pragma HLS ARRAY_PARTITION variable=x dim=2 complete
    // Compute mean/var/inv_std in FP32: variance scales as value^2 and can
    // exceed the data_t range for high-magnitude rows (e.g. cand input).
    // Casting through data_t wraps and rsqrt(negative) returns NaN. PyTorch
    // LayerNorm also keeps stats in float32 for the same reason.
    LN_ROW:
    for (int i = 0; i < N_ROWS; i++) {
#if LN_MODE == 1 || LN_MODE == 2 || LN_MODE == 6
        #pragma HLS PIPELINE II=1
#elif LN_MODE == 3
        #pragma HLS PIPELINE II=4
#endif
#if LN_MODE >= 2
        // integer path: exact mean (never rounded), 1/sqrt from a 385-entry
        // Q16 table with linear interpolation. Scales: data 2^9, gamma/beta
        // 2^12, so y_q = (g_q * d16 * R) >> (33 - e/2) + (b_q >> 3).
        ap_int<21> isum = 0;
        ap_int<16> xq[FEAT_DIM];
        LN_SUM: for (int j = 0; j < FEAT_DIM; j++) {
            LN_UNROLL_PRAGMA
            xq[j] = x[i][j].range(15, 0);
            isum += xq[j];
        }
        ap_int<22> d16[FEAT_DIM];
        ap_uint<46> V = 0;
        LN_D: for (int j = 0; j < FEAT_DIM; j++) {
            LN_UNROLL_PRAGMA
            d16[j] = ((ap_int<22>)xq[j] << 4) - isum;
            ap_int<44> sq = (ap_int<44>)d16[j] * d16[j];
#if LN_MODE == 6
            #pragma HLS BIND_OP variable=sq op=mul impl=fabric
#endif
            V += (ap_uint<44>)sq;
        }
        ap_uint<46> Vp = V + (ap_uint<46>)LN_EPSV;
        // normalize Vp to a 32-bit mantissa by an EVEN shift, so the square
        // root splits into sqrt(mantissa) x 2^(shift/2) and the table covers
        // the mantissa exactly. Vp >= LN_EPSV > 2^13, so the shift is bounded.
        int L = 46 - (int)Vp.countLeadingZeros();       // bit length
        int nsh = L - 32; if (nsh & 1) nsh++;           // keep it even
        ap_uint<32> Vn = (nsh >= 0) ? (ap_uint<32>)(Vp >> nsh)
                                    : (ap_uint<32>)(Vp << (-nsh));
        int idx = (int)(Vn >> 23) - 128;                // table index, 0..384
        ap_int<32> frac = (ap_int<32>)((Vn >> 7) & 0xFFFF);
        ap_int<32> l0 = LN_RSQRT_LUT[idx], l1 = LN_RSQRT_LUT[idx + 1];
        ap_int<32> R = l0 + (((l1 - l0) * frac) >> 16);      // Q16 1/sqrt
        int sh = 33 + (nsh >> 1);                        // >= 24 in practice
        LN_OUT: for (int j = 0; j < FEAT_DIM; j++) {
            LN_UNROLL_PRAGMA
            ap_int<16> gq = gamma[j].range(15, 0), bq = beta[j].range(15, 0);
            ap_int<56> num = (ap_int<56>)gq * d16[j] * R;
#if LN_MODE == 6
            #pragma HLS BIND_OP variable=num op=mul impl=fabric
#endif
            ap_int<56> rnd = (ap_int<56>)1 << (sh - 1);   // round to nearest
            ap_int<32> yq = (ap_int<32>)((num + rnd) >> sh) + (ap_int<32>)((bq + 4) >> 3);
            if (yq > 32767) yq = 32767;
            if (yq < -32768) yq = -32768;
            x[i][j].range(15, 0) = yq.range(15, 0);
        }
#else
        float sum_f = 0.0f;
        LN_MEAN:
        for (int j = 0; j < FEAT_DIM; j++) {
            #pragma HLS UNROLL
            sum_f += (float)x[i][j];
        }
        float mean_f = sum_f / (float)FEAT_DIM;

        float var_f = 0.0f;
        LN_VAR:
        for (int j = 0; j < FEAT_DIM; j++) {
            #pragma HLS UNROLL
            float d = (float)x[i][j] - mean_f;
            var_f += d * d;
        }
        var_f /= (float)FEAT_DIM;
        float inv_std_f = hls::rsqrt(var_f + (float)LN_EPS);

        LN_NORM:
        for (int j=0; j < FEAT_DIM; j++) {
            #pragma HLS PIPELINE II=1
            float y_f = ((float)x[i][j] - mean_f) * inv_std_f;
            x[i][j] = (data_t)((float)gamma[j] * y_f + (float)beta[j]);
        }
#endif
    }
}

// exp() via compile-time ROM (see gen_exp_lut.cpp). score_t has 5 fractional
// bits, so (x - EXP_MIN)*32 is an exact integer and the 257-entry table covers
// EVERY representable input in [-8, 0] exactly -- this is not an approximation
// relative to the datapath, it is (exp_t)expf(x) precomputed. Replaces a full
// float expf core per softmax element (the previous LUT was dead code: built,
// runtime-initialized, and never read).
#ifdef FLOAT_DATAPATH
static exp_t exp_fixed(score_t x) { return expf(x); }
#else
#include "exp_lut_rom.h"

static exp_t exp_fixed(score_t x) {
    #pragma HLS INLINE
    if (x >= (score_t)0) return (exp_t)1.0;
    if (x <= (score_t)EXP_MIN) return (exp_t)0.0; // exp(<=-8) < 1 prob_t LSB
    score_t d = x - (score_t)EXP_MIN;                       // (0, 8)
    int idx = (int)(d * (score_t)(EXP_LUT_SIZE / 8));       // *32: exact shift
    return exp_lut_rom[idx];
}
#endif

// converts into probabilities with partition fn over a row of length LEN

template <int LEN> // actual key length, (N_KV for obj/cross, T for cand)
void softmax_row(
    score_t row[LEN],
    prob_t out[LEN]
) {
    // find max
    score_t max_val = row[0];
    SM_MAX:
    for (int j = 0; j < LEN; j++) {
        #pragma HLS PIPELINE II=1
        if (row[j] > max_val) max_val = row[j];
    }

    // exp and sum

    exp_t exp_vals[LEN];
    exp_t exp_sum = 0;
    SM_EXP:
    for (int j = 0; j < LEN; j++) {
        #pragma HLS PIPELINE II=1
        exp_vals[j] = exp_fixed(row[j] - max_val);
        exp_sum += exp_vals[j];
    }

    // normalize

    exp_t inv_sum = (exp_t)(1.0f / (float)exp_sum);
    SM_NORM:
    for (int j = 0; j < LEN; j++) {
        #pragma HLS PIPELINE II=1
        out[j] = (prob_t)(exp_vals[j] * inv_sum);
    }
}

// FFN layer (linear, layernorm, relu) then skip + layernorm

template <int N_ROWS>
void ffn_block(
    data_t x[N_ROWS][E_DIM],
    const weight_t ffn_w[N_FFN_LAYERS][E_DIM][E_DIM],
    const weight_t ffn_b[N_FFN_LAYERS][E_DIM],
    const ln_param_t ffn_ln_g[N_FFN_LAYERS][E_DIM],
    const ln_param_t ffn_ln_b[N_FFN_LAYERS][E_DIM],
    const ln_param_t post_ffn_ln_g[E_DIM],
    const ln_param_t post_ffn_ln_b[E_DIM]
) {
    // save residuals

    data_t residual[N_ROWS][E_DIM]; 
    for (int i = 0; i < N_ROWS; i++) {
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < E_DIM; j++) {
            residual[i][j] = x[i][j];
        }
    }

    // layers
    for (int l = 0; l < N_FFN_LAYERS; l++) {
        data_t tmp[N_ROWS][E_DIM];
        linear<N_ROWS>(x, ffn_w[l], ffn_b[l], tmp);
        layernorm<N_ROWS>(tmp, ffn_ln_g[l], ffn_ln_b[l]);

        // relu

        for (int i = 0; i < N_ROWS; i++) {
            #pragma HLS PIPELINE II=1
            for (int j = 0; j < E_DIM; j++) {
                x[i][j] = (tmp[i][j] > (data_t)0? tmp[i][j] : (data_t) 0);
            }
        }

    }

    // skip connections and layernorm

    for (int i = 0; i < N_ROWS; i++) {
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < E_DIM; j++) {
            x[i][j] = x[i][j] + residual[i][j];
        }
    }

    layernorm<N_ROWS>(x, post_ffn_ln_g, post_ffn_ln_b);
}

// reshape QKV into per head arrays and append bias_kv
template <int N_Q, int N_KEY>
void reshape_and_append_bias_kv(
    const data_t Q_full[N_Q][E_DIM],
    const data_t K_full[N_KEY][E_DIM],
    const data_t V_full[N_KEY][E_DIM],
    const weight_t bias_k[E_DIM],
    const weight_t bias_v[E_DIM],
    data_t Q_h[N_HEADS][N_Q][D_HEAD],
    data_t K_h[N_HEADS][N_KEY+1][D_HEAD],
    data_t V_h[N_HEADS][N_KEY+1][D_HEAD]
) {
#ifdef RESHAPE_FAST
    // Two pipelined passes over the rows, writing every head and dim of a row
    // in one cycle: N_Q + N_KEY + 1 cycles plus latency (~30 for 12 rows).
    // The original pipelines only the innermost 4-iteration loop and pays
    // pipeline fill on every (h, i): 255 cycles for the same 25 rows.
    #pragma HLS ARRAY_PARTITION variable=Q_full dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=K_full dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=V_full dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=bias_k complete
    #pragma HLS ARRAY_PARTITION variable=bias_v complete
    #pragma HLS ARRAY_PARTITION variable=Q_h dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=Q_h dim=3 complete
    #pragma HLS ARRAY_PARTITION variable=K_h dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=K_h dim=3 complete
    #pragma HLS ARRAY_PARTITION variable=V_h dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=V_h dim=3 complete
    RS_Q: for (int i = 0; i < N_Q; i++) {
        #pragma HLS PIPELINE II=1
        for (int h = 0; h < N_HEADS; h++) {
            #pragma HLS UNROLL
            for (int d = 0; d < D_HEAD; d++) {
                #pragma HLS UNROLL
                Q_h[h][i][d] = Q_full[i][h * D_HEAD + d];
            }
        }
    }
    RS_KV: for (int i = 0; i <= N_KEY; i++) {
        #pragma HLS PIPELINE II=1
        for (int h = 0; h < N_HEADS; h++) {
            #pragma HLS UNROLL
            for (int d = 0; d < D_HEAD; d++) {
                #pragma HLS UNROLL
                const int e = h * D_HEAD + d;
                K_h[h][i][d] = (i < N_KEY) ? K_full[i][e] : (data_t)bias_k[e];
                V_h[h][i][d] = (i < N_KEY) ? V_full[i][e] : (data_t)bias_v[e];
            }
        }
    }
#else
    RESHAPE:
    for (int h = 0; h <N_HEADS; h++) {
        #pragma HLS UNROLL
        for (int i = 0; i < N_Q; i++) {
            for (int d = 0; d < D_HEAD; d++) {
                #pragma HLS PIPELINE II=1
                Q_h[h][i][d] = Q_full[i][h*D_HEAD+d];
            }
        }
        // separate from above because for cross attention N_KEY != N_Q
        for (int i = 0; i < N_KEY; i++) {
            for (int d = 0; d < D_HEAD; d++) {
                #pragma HLS PIPELINE II = 1
                int e = h * D_HEAD + d;
                K_h[h][i][d] = K_full[i][e];
                V_h[h][i][d] = V_full[i][e];
            }
        }
        
        for (int d = 0; d < D_HEAD; d++) {
            #pragma HLS PIPELINE II=1
            int e = h * D_HEAD + d;
            K_h[h][N_KEY][d] = (data_t)bias_k[e];
            V_h[h][N_KEY][d] = (data_t)bias_v[e];
        }
    }
#endif
}

// scaled dot product attention for one head

template <int N_Q, int N_KEY_TOT>
void compute_scores(
    const data_t Q[N_Q][D_HEAD],
    const data_t K[N_KEY_TOT][D_HEAD],
    score_t scores[N_Q][N_KEY_TOT]
) {
    // feed the unrolled QK_D dot product (D_HEAD reads/cycle from Q and K)
    #pragma HLS ARRAY_PARTITION variable=Q dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=K dim=2 complete
    QK_I:
    for (int i = 0; i < N_Q; i++) {
        QJ_I:
        for (int j = 0; j < N_KEY_TOT; j++) {
            #pragma HLS PIPELINE II=1
            acc_t sum = 0;
            QK_D:
            for (int d = 0; d < D_HEAD; d++) {
                #pragma HLS UNROLL
#ifdef NARROW_MUL
                sum += Q[i][d] * K[j][d];
#else
                sum += (acc_t)Q[i][d] * (acc_t)K[j][d];
#endif
            }
            scores[i][j] = (score_t)(sum * (acc_t)SCALE);
        }
    }
}


// softmax scores and then context = attn_weights @ V
template <int N_Q, int N_KEY_TOT>
void softmax_and_context(
    score_t scores[N_Q][N_KEY_TOT],
    const data_t V[N_KEY_TOT][D_HEAD],
    data_t context[N_Q][D_HEAD]
) {
    // feed the unrolled AV_J reduction (N_KEY_TOT reads/cycle)
    #pragma HLS ARRAY_PARTITION variable=V dim=1 complete
    prob_t attn_w[N_Q][N_KEY_TOT];
    #pragma HLS ARRAY_PARTITION variable=attn_w dim=2 complete
#ifdef SOFTMAX_PIPE
    // Overlap the twelve row softmaxes. One row is a serial chain of ~70
    // cycles (max, exp and sum, a float reciprocal, normalise) and without
    // this the rows run one after another: ~840 of the ~1200 cycles a head
    // costs, the largest single item in the block's latency. Pipelining the
    // row loop force-unrolls the row's inner loops 13 wide; that is the price.
    #pragma HLS ARRAY_PARTITION variable=scores dim=2 complete
#endif
    SM_ROWS:
    for (int i = 0; i < N_Q; i++) {
#ifdef SOFTMAX_PIPE
        #pragma HLS PIPELINE II=1
#endif
        softmax_row<N_KEY_TOT>(scores[i], attn_w[i]);
    }

    AV_I:
    for (int i = 0; i < N_Q; i++) {
        AV_D:
        for (int d = 0; d < D_HEAD; d++) {
            #pragma HLS PIPELINE II=1
            acc_t sum = 0;
            AV_J:
            for (int j=0; j < N_KEY_TOT; j++) {
                #pragma HLS UNROLL
#ifdef NARROW_MUL
                sum += attn_w[i][j] * V[j][d];
#else
                sum += (acc_t)attn_w[i][j] * (acc_t)V[j][d];
#endif
            }
            context[i][d] = (data_t)sum;
        }
    }
}

// All heads at once, for the blocks with no w_ij bias and no padding mask
// (candidate and cross attention). The per-head loop in those blocks runs the
// heads back to back; here the head index is the innermost unrolled dimension
// of each pipelined loop, so the four heads share every iteration. Same
// arithmetic and order as compute_scores + softmax_and_context.
template <int N_Q, int N_KEY_TOT>
void heads_batched(
    const data_t Q_h[N_HEADS][N_Q][D_HEAD],
    const data_t K_h[N_HEADS][N_KEY_TOT][D_HEAD],
    const data_t V_h[N_HEADS][N_KEY_TOT][D_HEAD],
    data_t context[N_HEADS][N_Q][D_HEAD])
{
    #pragma HLS ARRAY_PARTITION variable=Q_h dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=Q_h dim=3 complete
    #pragma HLS ARRAY_PARTITION variable=K_h dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=K_h dim=3 complete
    #pragma HLS ARRAY_PARTITION variable=V_h dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=V_h dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=context dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=context dim=3 complete
    score_t scores[N_HEADS][N_Q][N_KEY_TOT];
    #pragma HLS ARRAY_PARTITION variable=scores dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=scores dim=3 complete
    HBG_SC_I: for (int i = 0; i < N_Q; i++) {
        HBG_SC_J: for (int j = 0; j < N_KEY_TOT; j++) {
            #pragma HLS PIPELINE II=1
            HBG_SC_H: for (int h = 0; h < N_HEADS; h++) {
                #pragma HLS UNROLL
                acc_t sum = 0;
                HBG_SC_D: for (int d = 0; d < D_HEAD; d++) {
                    #pragma HLS UNROLL
#ifdef NARROW_MUL
                    sum += Q_h[h][i][d] * K_h[h][j][d];
#else
                    sum += (acc_t)Q_h[h][i][d] * (acc_t)K_h[h][j][d];
#endif
                }
                scores[h][i][j] = (score_t)(sum * (acc_t)SCALE);
            }
        }
    }
    prob_t attn_w[N_HEADS][N_Q][N_KEY_TOT];
    #pragma HLS ARRAY_PARTITION variable=attn_w dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=attn_w dim=3 complete
    HBG_SM_H: for (int h = 0; h < N_HEADS; h++) {
        HBG_SM_I: for (int i = 0; i < N_Q; i++) {
            #pragma HLS PIPELINE II=1
            softmax_row<N_KEY_TOT>(scores[h][i], attn_w[h][i]);
        }
    }
    HBG_AV_I: for (int i = 0; i < N_Q; i++) {
        HBG_AV_D: for (int d = 0; d < D_HEAD; d++) {
            #pragma HLS PIPELINE II=1
            HBG_AV_H: for (int h = 0; h < N_HEADS; h++) {
                #pragma HLS UNROLL
                acc_t sum = 0;
                HBG_AV_J: for (int j = 0; j < N_KEY_TOT; j++) {
                    #pragma HLS UNROLL
#ifdef NARROW_MUL
                    sum += attn_w[h][i][j] * V_h[h][j][d];
#else
                    sum += (acc_t)attn_w[h][i][j] * (acc_t)V_h[h][j][d];
#endif
                }
                context[h][i][d] = (data_t)sum;
            }
        }
    }
}

// concatenate heads and output projection

template<int N_Q>
void concat_and_project(
    const data_t context[N_HEADS][N_Q][D_HEAD],
    const weight_t Wo[E_DIM][E_DIM],
    const weight_t bo[E_DIM],
    data_t out[N_Q][E_DIM]
) {
    data_t concat_out[N_Q][E_DIM];
    CONCAT:
    for (int i = 0; i < N_Q; i++) {
        #pragma HLS PIPELINE II=1
        for (int h = 0; h < N_HEADS; h++) {
            for (int d = 0; d < D_HEAD; d++) {
                concat_out[i][h * D_HEAD + d] = context[h][i][d];
            }
        }
    }
    linear<N_Q>(concat_out, Wo, bo, out);
}

template<int N_ROWS>
void skip_and_norm(
    data_t x[N_ROWS][E_DIM],
    const data_t residual[N_ROWS][E_DIM],
    const ln_param_t ln_g[E_DIM],
    const ln_param_t ln_b[E_DIM]
) {
    for (int i = 0; i < N_ROWS; i++) {
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < E_DIM; j++)
            x[i][j] = x[i][j] + residual[i][j];
    }
    layernorm<N_ROWS>(x, ln_g, ln_b);
}

// zero out padded jets after attn blocks
inline void remask(data_t x[N_MAX][E_DIM], const bool mask[N_MAX]) {
    REMASK_J: for (int j = 0; j < N_MAX; j++) {
        if (mask[j]) {
            REMASK_E: for (int e = 0; e < E_DIM; e++) {
                #pragma HLS PIPELINE II=1
                x[j][e] = (data_t)0;
            }
        }
    }
}

// expand wij[12x12] to wij_bias[48x13] for multi-head attention
// replicates the same 12x12 wij for each of N_HEADS heads
// column 12 (bias_kv position) stays zero

inline void expand_wij(
    const data_t wij[N_MAX][N_MAX],
    score_t wij_bias[N_HEADS * N_MAX][N_KV]
) {
    // single pass: copy per head, column N_MAX (bias_kv token) stays zero
    EXPAND: for (int h = 0; h < N_HEADS; h++) {
        for (int i = 0; i < N_MAX; i++) {
            #pragma HLS PIPELINE II=1
            for (int j = 0; j < N_KV; j++) {
                wij_bias[h * N_MAX + i][j] = (j < N_MAX) ? (score_t)wij[i][j] : (score_t)0;
            }
        }
    }
}


struct AttnWeights {
    weight_t Wq[E_DIM][E_DIM], bq[E_DIM];   
    weight_t Wk[E_DIM][E_DIM], bk[E_DIM];   
    weight_t Wv[E_DIM][E_DIM], bv[E_DIM];
    weight_t bias_k[E_DIM], bias_v[E_DIM];
    weight_t Wo[E_DIM][E_DIM], bo[E_DIM];
    ln_param_t attn_ln_g[E_DIM], attn_ln_b[E_DIM];
    weight_t ffn_w[N_FFN_LAYERS][E_DIM][E_DIM];
    weight_t ffn_b[N_FFN_LAYERS][E_DIM];
    ln_param_t ffn_ln_g[N_FFN_LAYERS][E_DIM];
    ln_param_t ffn_ln_b[N_FFN_LAYERS][E_DIM];
    ln_param_t post_ffn_g[E_DIM], post_ffn_b[E_DIM];
};  

#endif