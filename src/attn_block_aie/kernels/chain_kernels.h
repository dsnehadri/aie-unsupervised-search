// Tile kernels that replace the fabric stages between attention blocks when
// the whole ABC layer runs on the array (aie_chain_graph.h). Every consumer
// gets its own output window: a kernel window fanned out to 4-5 consumers
// mis-delivered in aiesimulator (consumers ran more or fewer times than the
// producer), while the existing graphs never fan a kernel output out more
// than 2 ways; PLIO fan-out is fine.
//   chain_assemble_zero : x (N_MAX x E_DIM) + mask (16 words, 1 = padded jet)
//                         -> x with padded rows zeroed plus the mask row
//                         (the object block's N_MAX+1 row window), 5 copies:
//                         4 head kernels + post_a_proj's residual.
//   chain_assemble      : same without the zeroing (obj_attn_send_nowij).
//   chain_post_obj      : remask + candidate build (remask_stage and
//                         candidate_build_stage): zero padded rows, isr bias
//                         x[i][2] -= 1, per-jet argmax over the first T_DIM
//                         features, c[t] = sum of the jets in category t;
//                         x (biased, remasked) x5 for the cross block, c x5
//                         for the candidate block.
//   chain_fanout_c      : c -> 5 copies (cross block's 4 heads + one spare,
//                         used for the c PLIO out in layer 1).
#ifndef CHAIN_KERNELS_H
#define CHAIN_KERNELS_H
#include "attn_aie_types.h"
void chain_assemble_zero(input_window_int16* __restrict x_in, input_window_int16* __restrict mask_in,
    output_window_int16* __restrict o0, output_window_int16* __restrict o1, output_window_int16* __restrict o2,
    output_window_int16* __restrict o3, output_window_int16* __restrict o4);
void chain_assemble(input_window_int16* __restrict x_in, input_window_int16* __restrict mask_in,
    output_window_int16* __restrict o0, output_window_int16* __restrict o1, output_window_int16* __restrict o2,
    output_window_int16* __restrict o3, output_window_int16* __restrict o4);
void chain_post_obj(input_window_int16* __restrict x_in, input_window_int16* __restrict mask_in,
    output_window_int16* __restrict x0, output_window_int16* __restrict x1, output_window_int16* __restrict x2,
    output_window_int16* __restrict x3, output_window_int16* __restrict x4,
    output_window_int16* __restrict c0, output_window_int16* __restrict c1, output_window_int16* __restrict c2,
    output_window_int16* __restrict c3, output_window_int16* __restrict c4);
void chain_fanout_c4(input_window_int16* __restrict c_in,
    output_window_int16* __restrict o0, output_window_int16* __restrict o1, output_window_int16* __restrict o2,
    output_window_int16* __restrict o3);
void chain_fanout_c(input_window_int16* __restrict c_in,
    output_window_int16* __restrict o0, output_window_int16* __restrict o1, output_window_int16* __restrict o2,
    output_window_int16* __restrict o3, output_window_int16* __restrict o4);
#endif
