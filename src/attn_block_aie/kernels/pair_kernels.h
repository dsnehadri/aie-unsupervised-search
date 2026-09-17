// Pairwise bias MLP on the array (PAIRWISE_ON_AIE).
//
// For every pair of jets (i, j) the bias network is
//   Linear(3 -> 16) + LN + ReLU -> Linear(16 -> 16) + LN + ReLU
//   -> Linear(16 -> 16) + LN + ReLU -> Linear(16 -> 1)
// on (eta_i - eta_j, cos(phi_i - phi_j), sin(phi_i - phi_j)). On the fabric this
// was the slowest stage of the hybrid (653 cycles, 5.2 us) and its result
// reached the array about 8 us after the fork, later than the embedding path.
//
// Layout: one feature kernel computes the 144 x 3 features from the raw jets
// window (the same one the embedding reads) and streams them out, chains 0-5
// on one stream, 6-11 on the other. Twelve three-tile chains (layer 0 / layer 1
// / layers 2 + 3) each take one jet i against every j, four rows at a time, the
// way the embedding pipeline does. A tree of eleven two-input merge kernels puts
// the twelve bias rows back in order; the last one feeds the layer-0 head posts'
// bias windows as a stream.
// Rows are emitted at the score scale (Q8.7) exactly as the fabric's wij_send
// did (Q6.9 -> Q10.5 by truncation, then << 2), padded to 16 lanes.
#ifndef PAIR_KERNELS_H
#define PAIR_KERNELS_H

#include "attn_aie_types.h"

constexpr int PAIR_K          = 4;                    // 3 features padded to 4
constexpr int PAIR_ROWS       = N_MAX;                // pairs per chain: jet i against every j
constexpr int PAIR_CHAINS     = N_MAX;                // one chain per jet i
constexpr int PAIR_PER_OUT    = PAIR_CHAINS / 2;      // chains fed by each feature stream
constexpr int PAIR_FEAT_WORDS = PAIR_ROWS * PAIR_K;   // 48 words per chain

void pair_feat(input_window_int16* __restrict jets_in,
               output_stream_int16* __restrict out_a,
               output_stream_int16* __restrict out_b);

#if defined(PAIR_L0_WINDOW)
#define DECL_PAIR_L0(k) void pair_l0_c##k(input_window_int16* __restrict jets_in, output_stream_int16* __restrict out)
#else
#define DECL_PAIR_L0(k) void pair_l0_c##k(input_stream_int16* __restrict in, output_stream_int16* __restrict out)
#endif
#define DECL_PAIR_CHAIN(k) \
    DECL_PAIR_L0(k); \
    void pair_l1_c##k(input_stream_int16* __restrict in, output_stream_int16* __restrict out); \
    void pair_l2_c##k(input_stream_int16* __restrict in, output_stream_int16* __restrict out)
DECL_PAIR_CHAIN(0);  DECL_PAIR_CHAIN(1);  DECL_PAIR_CHAIN(2);  DECL_PAIR_CHAIN(3);
DECL_PAIR_CHAIN(4);  DECL_PAIR_CHAIN(5);  DECL_PAIR_CHAIN(6);  DECL_PAIR_CHAIN(7);
DECL_PAIR_CHAIN(8);  DECL_PAIR_CHAIN(9);  DECL_PAIR_CHAIN(10); DECL_PAIR_CHAIN(11);

// merge tree: level 1 pairs chains (1 + 1 rows), level 2 (2 + 2), level 3 (4 + 4),
// level 4 (8 + 4) -> the 12 rows in order
#define DECL_PAIR_MERGE(n) void pair_merge_##n(input_stream_int16* __restrict a_in, \
    input_stream_int16* __restrict b_in, output_stream_int16* __restrict out)
DECL_PAIR_MERGE(1_0); DECL_PAIR_MERGE(1_1); DECL_PAIR_MERGE(1_2);
DECL_PAIR_MERGE(1_3); DECL_PAIR_MERGE(1_4); DECL_PAIR_MERGE(1_5);
DECL_PAIR_MERGE(2_0); DECL_PAIR_MERGE(2_1); DECL_PAIR_MERGE(2_2);
DECL_PAIR_MERGE(3_0); DECL_PAIR_MERGE(4_0);

#endif // PAIR_KERNELS_H
