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

DECL_POST_PROJ(obj, 0);  DECL_POST_PROJ(obj, 1);
DECL_POST_PROJ(cand, 0); DECL_POST_PROJ(cand, 1);
DECL_POST_PROJ(cross, 0);DECL_POST_PROJ(cross, 1);

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
