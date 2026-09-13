// Tile kernels that replace the fabric stages between attention blocks when
// the whole ABC layer runs on the array (aie_chain_graph.h):
//   chain_assemble_zero : x (N_MAX x E_DIM) + mask (16 words, 1 = padded jet)
//                         -> x with padded rows zeroed, plus the mask row
//                         appended (the object block's N_MAX+1 row window);
//                         what embed_recv + obj_attn_send did on the fabric.
//   chain_assemble      : same without the zeroing (obj_attn_send_nowij).
//   chain_post_obj      : remask + candidate build (what remask_stage and
//                         candidate_build_stage did): zero padded rows, isr
//                         bias x[i][2] -= 1, per-jet argmax over the first
//                         T_DIM features, c[t] = sum of the jets in category t;
//                         emits x (biased, remasked) for the cross block and c.
#ifndef CHAIN_KERNELS_H
#define CHAIN_KERNELS_H
#include "attn_aie_types.h"
void chain_assemble_zero(input_window_int16* __restrict x_in, input_window_int16* __restrict mask_in,
                         output_window_int16* __restrict x_out);
void chain_assemble(input_window_int16* __restrict x_in, input_window_int16* __restrict mask_in,
                    output_window_int16* __restrict x_out);
void chain_post_obj(input_window_int16* __restrict x_in, input_window_int16* __restrict mask_in,
                    output_window_int16* __restrict x_out, output_window_int16* __restrict c_out);
#endif
