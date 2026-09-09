// Jet embedding MLP on one AI Engine tile.
//   Linear(5 -> 16) + LN + ReLU -> Linear(16 -> 16) + LN + ReLU -> Linear(16 -> 16)
// applied to the N_MAX jets of an event. In the hybrid pipeline this stage runs
// on the fabric and is the slowest stage there (59 us/event), which sets the
// whole pipeline's rate now that the attention blocks are fast.
#ifndef EMBED_KERNEL_H
#define EMBED_KERNEL_H

#include "attn_aie_types.h"

constexpr int EMBED_IN     = 5;     // log pT, eta, cos phi, sin phi, log E
constexpr int EMBED_IN_PAD = 8;     // gemm_pk needs K a multiple of 4
constexpr int EMBED_ROWS   = N_MAX;
// The input window must be a multiple of 32 bytes or the compiler rounds it up
// and the kernel then eats the start of the next event: 12x5 int16 = 120 B was
// silently mapped to 128 B, which corrupted every event (AUC 0.9818 -> 0.8597).
// Send a padded 64-int16 window instead and drop the tail.
constexpr int EMBED_IN_WORDS = 64;  // 128 bytes

// in:  EMBED_IN_WORDS int16: N_MAX x EMBED_IN raw features then padding
// out: N_MAX x E_DIM embedded jets (masking stays on the fabric)
void embed_mlp(input_window_int16* __restrict jets_in,
               output_window_int16* __restrict embed_out);

#endif // EMBED_KERNEL_H
