// Tile kernels that replace the fabric stages between attention blocks when
// the whole ABC layer runs on the array (aie_chain_graph.h). Their outputs are
// STREAMS: a window fanned out to 4-5 consumers mis-delivered in aiesimulator
// and one window per consumer (or trees of copy kernels) would not place, while
// a stream multicasts through the switch to any number of window inputs --
// the mechanism a PLIO uses, which feeds five kernels in the deployed graphs.
//   chain_assemble_zero : x (N_MAX x E_DIM) + mask (16 words, 1 = padded jet)
//                         -> x with padded rows zeroed plus the mask row
//                         (the object block's N_MAX+1 row window).
//   chain_assemble      : same without the zeroing (obj_attn_send_nowij).
//   chain_post_obj      : remask + candidate build (remask_stage and
//                         candidate_build_stage): zero padded rows, isr bias
//                         x[i][2] -= 1, per-jet argmax over the first T_DIM
//                         features, c[t] = sum of the jets in category t;
//                         out: x (biased, remasked) for the cross block, c for
//                         the candidate block.
//   chain_w2s_48        : c window -> c stream (the candidate block's output
//                         feeds the cross block's four heads).
#ifndef CHAIN_KERNELS_H
#define CHAIN_KERNELS_H
#include "attn_aie_types.h"
void chain_assemble_zero(input_window_int16* __restrict x_in, input_window_int16* __restrict mask_in,
                         output_stream_int16* __restrict x_out);
void chain_assemble(input_window_int16* __restrict x_in, input_window_int16* __restrict mask_in,
                    output_stream_int16* __restrict x_out);
void chain_post_obj(input_window_int16* __restrict x_in, input_window_int16* __restrict mask_in,
                    output_stream_int16* __restrict x_out, output_stream_int16* __restrict c_out);
void chain_w2s_48(input_window_int16* __restrict c_in, output_stream_int16* __restrict c_out);
#endif
