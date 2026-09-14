// Tile kernels that replace the fabric stages between attention blocks when
// the whole ABC layer runs on the array (aie_chain_graph.h).
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
//   chain_dup2_<N>      : one window in, two identical windows out. A tile
//                         has two MM2S DMA channels, so a window consumed by
//                         more than two remote kernels goes through a tree of
//                         these (a 4-5 way fan-out from one kernel placed for
//                         a one-layer slice but not for the whole stack, and
//                         a plain multi-consumer window mis-delivered in
//                         aiesimulator).
#ifndef CHAIN_KERNELS_H
#define CHAIN_KERNELS_H
#include "attn_aie_types.h"
void chain_assemble_zero(input_window_int16* __restrict x_in, input_window_int16* __restrict mask_in,
                         output_window_int16* __restrict x_out);
void chain_assemble(input_window_int16* __restrict x_in, input_window_int16* __restrict mask_in,
                    output_window_int16* __restrict x_out);
void chain_post_obj(input_window_int16* __restrict x_in, input_window_int16* __restrict mask_in,
                    output_window_int16* __restrict x_out, output_window_int16* __restrict c_out);
void chain_dup2_208(input_window_int16* __restrict in, output_window_int16* __restrict o0, output_window_int16* __restrict o1);
void chain_dup2_192(input_window_int16* __restrict in, output_window_int16* __restrict o0, output_window_int16* __restrict o1);
void chain_dup2_48(input_window_int16* __restrict in, output_window_int16* __restrict o0, output_window_int16* __restrict o1);
#endif
