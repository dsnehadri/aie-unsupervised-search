// post-attention processing on aie

// concatenate 4 heads outputs -> [12][16]
// output projection: [12][16] x Wout[16][16] -> [12][16]
// skip connection + layernorm
// ffn linear->ln->relu x 3 layers
// skip connection + layernorm

// runs one one aie tile
// memory Wout (512B) + 3xFFN_W(1536B) + LN params (~200B) + buffers = 3.5KB

#ifndef ATTN_POST_KERNEL_H
#define ATTN_POST_KERNEL_H

#include "attn_aie_types.h"

void attn_post(input_window_int16* __restrict head0_in,
                input_window_int16* __restrict head1_in,
                input_window_int16* __restrict head2_in,
                input_window_int16* __restrict head3_in,
                input_window_int16* __restrict residual_in,
                output_window_int16* __restrict x_out)
#endif