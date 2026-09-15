// Post-attention split across 4 tiles per subgraph (per (type, layer)):
//   post_a_proj:   4 head outputs + residual -> gemm + skip + LN -> proj_out
//                  (the former post_a_concat tile was pure data movement
//                  and has been merged in)
//   post_b1:       FFN layer 0
//   post_b2:       FFN layer 1
//   post_c:        FFN layer 2 + skip with broadcast proj_out + LN
//
// Forward declarations of every (type, layer) variant so the graph can
// reference them by unique symbol. The aiecompiler dedups wrappers by
// function symbol, so per-instance unique names are required.

#ifndef ATTN_POST_KERNEL_H
#define ATTN_POST_KERNEL_H

#include "attn_aie_types.h"

#ifdef FLOAT_AIE
#define AIE_IW input_window_float
#define AIE_OW output_window_float
#else
#define AIE_IW input_window_int16
#define AIE_OW output_window_int16
#endif

#define DECL_POST_PROJ(t, l) void t##_post_a_proj_L##l( \
    AIE_IW* __restrict head0_in, \
    AIE_IW* __restrict head1_in, \
    AIE_IW* __restrict head2_in, \
    AIE_IW* __restrict head3_in, \
    AIE_IW* __restrict residual_in, \
    AIE_OW* __restrict proj_out)
#define DECL_POST_B1(t, l) void t##_post_b1_L##l( \
    AIE_IW* __restrict proj_in, \
    AIE_OW* __restrict ffn0_out)
#define DECL_POST_B2(t, l) void t##_post_b2_L##l( \
    AIE_IW* __restrict ffn0_in, \
    AIE_OW* __restrict ffn1_out)
#define DECL_POST_C(t, l) void t##_post_c_L##l( \
    AIE_IW* __restrict ffn_in, \
    AIE_IW* __restrict residual_b_in, \
    AIE_OW* __restrict x_out)
// POST_MERGED: b1 + b2 + c in one kernel (proj in, x out; the FFN residual is
// the same proj window, so post_a's broadcast and two window hops are gone)
#define DECL_POST_BC(t, l) void t##_post_bc_L##l( \
    AIE_IW* __restrict proj_in, \
    AIE_OW* __restrict x_out)
DECL_POST_BC(obj, 0);  DECL_POST_BC(obj, 1);
DECL_POST_BC(cand, 0); DECL_POST_BC(cand, 1);
DECL_POST_BC(cross, 0);DECL_POST_BC(cross, 1);

#ifdef POST_STREAM
// POST_STREAM: rows stream from a_proj through b1, b2 and c (row-level
// pipelining; a_proj still reads the head and residual windows). The block
// output is a stream too; consumers connect with connect<stream, window<>>.
#undef DECL_POST_PROJ
#undef DECL_POST_B1
#undef DECL_POST_B2
#undef DECL_POST_C
#define DECL_POST_PROJ(t, l) void t##_post_a_proj_L##l( \
    AIE_IW* __restrict head0_in, AIE_IW* __restrict head1_in, AIE_IW* __restrict head2_in, \
    AIE_IW* __restrict head3_in, AIE_IW* __restrict residual_in, output_stream_int16* __restrict proj_out)
#if defined(HEAD_STREAM)
// HEAD_STREAM: the object and cross projections read rows from the two merge
// kernels instead of gathering four head windows. The candidate block keeps the
// windows: three rows do not divide into groups of four.
#define DECL_POST_PROJ_S(t, l) void t##_post_a_proj_L##l( \
    input_stream_int16* __restrict h01_in, input_stream_int16* __restrict h23_in, \
    AIE_IW* __restrict residual_in, output_stream_int16* __restrict proj_out)
#endif
#define DECL_POST_B1(t, l) void t##_post_b1_L##l(input_stream_int16* __restrict proj_in, output_stream_int16* __restrict ffn0_out)
#define DECL_POST_B2(t, l) void t##_post_b2_L##l(input_stream_int16* __restrict ffn0_in, output_stream_int16* __restrict ffn1_out)
#define DECL_POST_C(t, l) void t##_post_c_L##l(input_stream_int16* __restrict ffn_in, \
    input_stream_int16* __restrict residual_b_in, output_stream_int16* __restrict x_out)
#if defined(POST_SPLIT_C)
// POST_SPLIT_C: c1 takes the FFN's last linear layer, its norm and the ReLU;
// c keeps the residual add and the final norm. c's first input is c1's output,
// so the block still ends at post_c and the graph above it is unchanged.
#define DECL_POST_C1(t, l) void t##_post_c1_L##l(input_stream_int16* __restrict ffn_in, \
    output_stream_int16* __restrict ffn_out)
DECL_POST_C1(obj, 0);  DECL_POST_C1(obj, 1);
DECL_POST_C1(cand, 0); DECL_POST_C1(cand, 1);
DECL_POST_C1(cross, 0);DECL_POST_C1(cross, 1);
#endif
#endif

#if defined(HEAD_STREAM) && defined(POST_STREAM)
DECL_POST_PROJ_S(obj, 0);  DECL_POST_PROJ_S(obj, 1);
DECL_POST_PROJ(cand, 0);   DECL_POST_PROJ(cand, 1);
DECL_POST_PROJ_S(cross, 0);DECL_POST_PROJ_S(cross, 1);
#else
DECL_POST_PROJ(obj, 0);  DECL_POST_PROJ(obj, 1);
DECL_POST_PROJ(cand, 0); DECL_POST_PROJ(cand, 1);
DECL_POST_PROJ(cross, 0);DECL_POST_PROJ(cross, 1);
#endif

DECL_POST_B1(obj, 0);  DECL_POST_B1(obj, 1);
DECL_POST_B1(cand, 0); DECL_POST_B1(cand, 1);
DECL_POST_B1(cross, 0);DECL_POST_B1(cross, 1);

DECL_POST_B2(obj, 0);  DECL_POST_B2(obj, 1);
DECL_POST_B2(cand, 0); DECL_POST_B2(cand, 1);
DECL_POST_B2(cross, 0);DECL_POST_B2(cross, 1);

DECL_POST_C(obj, 0);  DECL_POST_C(obj, 1);
DECL_POST_C(cand, 0); DECL_POST_C(cand, 1);
DECL_POST_C(cross, 0);DECL_POST_C(cross, 1);

#endif
