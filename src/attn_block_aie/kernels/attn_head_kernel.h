// Attention head kernels for AIE, split into pre + post stages.
// pre:  read inputs -> Q/K/V proj -> Q*K^T scaled -> emit scores+V
// post: read scores+V (+wij for obj) -> softmax -> AV -> emit head_out
//
// The aiecompiler dedups generated tile wrappers by function-pointer
// identity, so each (type, stage, head, layer) instance must declare a
// distinct symbol. Generated via token-paste macros in attn_head_kernel.cc.
// This header has to forward-declare all 24 variants (3 types * 4 heads * 2
// stages) per layer.

#ifndef ATTN_HEAD_KERNEL_H
#define ATTN_HEAD_KERNEL_H

#include "attn_aie_types.h"

#ifdef FLOAT_AIE
#define AIE_IW input_window_float
#define AIE_OW output_window_float
#else
#define AIE_IW input_window_int16
#define AIE_OW output_window_int16
#endif

// pre signatures
#define DECLARE_OBJ_PRE(h, l)   void obj_attn_head_pre_h##h##_L##l ( \
    AIE_IW* __restrict x_in, \
    AIE_OW* __restrict scores_out, \
    AIE_OW* __restrict v_out)
#define DECLARE_CAND_PRE(h, l)  void cand_attn_head_pre_h##h##_L##l ( \
    AIE_IW* __restrict c_in, \
    AIE_OW* __restrict scores_out, \
    AIE_OW* __restrict v_out)
#define DECLARE_CROSS_PRE(h, l) void cross_attn_head_pre_h##h##_L##l ( \
    AIE_IW* __restrict x_in, \
    AIE_IW* __restrict c_in, \
    AIE_OW* __restrict scores_out, \
    AIE_OW* __restrict v_out)

// post signatures (obj layer 1 has no wij port -- the bias only exists in
// layer 0; streaming zeros to a dummy port wasted 624 words/event of NoC
// traffic plus 4 PLIOs)
#define DECLARE_OBJ_POST_L0(h)  void obj_attn_head_post_h##h##_L0 ( \
    AIE_IW* __restrict scores_in, \
    AIE_IW* __restrict v_in, \
    AIE_IW* __restrict wij_in, \
    AIE_OW* __restrict x_out)
#define DECLARE_OBJ_POST_L1(h)  void obj_attn_head_post_h##h##_L1 ( \
    AIE_IW* __restrict scores_in, \
    AIE_IW* __restrict v_in, \
    AIE_OW* __restrict x_out)
#define DECLARE_CAND_POST(h, l) void cand_attn_head_post_h##h##_L##l ( \
    AIE_IW* __restrict scores_in, \
    AIE_IW* __restrict v_in, \
    AIE_OW* __restrict c_out)
#define DECLARE_CROSS_POST(h, l) void cross_attn_head_post_h##h##_L##l ( \
    AIE_IW* __restrict scores_in, \
    AIE_IW* __restrict v_in, \
    AIE_OW* __restrict x_out)

DECLARE_OBJ_PRE(0, 0);  DECLARE_OBJ_PRE(1, 0);  DECLARE_OBJ_PRE(2, 0);  DECLARE_OBJ_PRE(3, 0);
DECLARE_OBJ_PRE(0, 1);  DECLARE_OBJ_PRE(1, 1);  DECLARE_OBJ_PRE(2, 1);  DECLARE_OBJ_PRE(3, 1);
DECLARE_OBJ_POST_L0(0); DECLARE_OBJ_POST_L0(1); DECLARE_OBJ_POST_L0(2); DECLARE_OBJ_POST_L0(3);
DECLARE_OBJ_POST_L1(0); DECLARE_OBJ_POST_L1(1); DECLARE_OBJ_POST_L1(2); DECLARE_OBJ_POST_L1(3);

DECLARE_CAND_PRE(0, 0);  DECLARE_CAND_PRE(1, 0);  DECLARE_CAND_PRE(2, 0);  DECLARE_CAND_PRE(3, 0);
DECLARE_CAND_PRE(0, 1);  DECLARE_CAND_PRE(1, 1);  DECLARE_CAND_PRE(2, 1);  DECLARE_CAND_PRE(3, 1);
DECLARE_CAND_POST(0, 0); DECLARE_CAND_POST(1, 0); DECLARE_CAND_POST(2, 0); DECLARE_CAND_POST(3, 0);
DECLARE_CAND_POST(0, 1); DECLARE_CAND_POST(1, 1); DECLARE_CAND_POST(2, 1); DECLARE_CAND_POST(3, 1);

DECLARE_CROSS_PRE(0, 0);  DECLARE_CROSS_PRE(1, 0);  DECLARE_CROSS_PRE(2, 0);  DECLARE_CROSS_PRE(3, 0);
DECLARE_CROSS_PRE(0, 1);  DECLARE_CROSS_PRE(1, 1);  DECLARE_CROSS_PRE(2, 1);  DECLARE_CROSS_PRE(3, 1);
DECLARE_CROSS_POST(0, 0); DECLARE_CROSS_POST(1, 0); DECLARE_CROSS_POST(2, 0); DECLARE_CROSS_POST(3, 0);
DECLARE_CROSS_POST(0, 1); DECLARE_CROSS_POST(1, 1); DECLARE_CROSS_POST(2, 1); DECLARE_CROSS_POST(3, 1);

#endif
