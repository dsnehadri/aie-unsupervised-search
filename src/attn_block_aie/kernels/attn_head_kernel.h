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

// pre signatures. PRE_STREAM: x arrives as a row stream (the object and cross
// blocks); the candidate block's c stays a window.
#if defined(PRE_STREAM)
#define DECLARE_OBJ_PRE(h, l)   void obj_attn_head_pre_h##h##_L##l ( \
    input_stream_int16* __restrict x_in, \
    AIE_OW* __restrict scores_out, \
    AIE_OW* __restrict v_out)
#else
#define DECLARE_OBJ_PRE(h, l)   void obj_attn_head_pre_h##h##_L##l ( \
    AIE_IW* __restrict x_in, \
    AIE_OW* __restrict scores_out, \
    AIE_OW* __restrict v_out)
#endif
#define DECLARE_CAND_PRE(h, l)  void cand_attn_head_pre_h##h##_L##l ( \
    AIE_IW* __restrict c_in, \
    AIE_OW* __restrict scores_out, \
    AIE_OW* __restrict v_out)
#if defined(PRE_STREAM) && defined(PRE_STREAM_CROSS)
#define DECLARE_CROSS_PRE(h, l) void cross_attn_head_pre_h##h##_L##l ( \
    input_stream_int16* __restrict x_in, \
    input_stream_int16* __restrict c_in, \
    AIE_OW* __restrict scores_out, \
    AIE_OW* __restrict v_out)
#else
#define DECLARE_CROSS_PRE(h, l) void cross_attn_head_pre_h##h##_L##l ( \
    AIE_IW* __restrict x_in, \
    AIE_IW* __restrict c_in, \
    AIE_OW* __restrict scores_out, \
    AIE_OW* __restrict v_out)
#endif

// post signatures (obj layer 1 has no wij port -- the bias only exists in
// layer 0; streaming zeros to a dummy port wasted 624 words/event of NoC
// traffic plus 4 PLIOs)
// Object and cross head posts emit their rows on a stream when their block type
// streams; two merge kernels then pair the four heads for the projection.
#if defined(HEAD_STREAM_OBJ)
#define DECLARE_OBJ_POST_L0(h)  void obj_attn_head_post_h##h##_L0 ( \
    AIE_IW* __restrict scores_in, \
    AIE_IW* __restrict v_in, \
    AIE_IW* __restrict wij_in, \
    output_stream_int16* __restrict x_out)
#define DECLARE_OBJ_POST_L1(h)  void obj_attn_head_post_h##h##_L1 ( \
    AIE_IW* __restrict scores_in, \
    AIE_IW* __restrict v_in, \
    output_stream_int16* __restrict x_out)
#else
#define DECLARE_OBJ_POST_L0(h)  void obj_attn_head_post_h##h##_L0 ( \
    AIE_IW* __restrict scores_in, \
    AIE_IW* __restrict v_in, \
    AIE_IW* __restrict wij_in, \
    AIE_OW* __restrict x_out)
#define DECLARE_OBJ_POST_L1(h)  void obj_attn_head_post_h##h##_L1 ( \
    AIE_IW* __restrict scores_in, \
    AIE_IW* __restrict v_in, \
    AIE_OW* __restrict x_out)
#endif

#if defined(HEAD_STREAM_OBJ) || defined(HEAD_STREAM_CROSS)
#define DECLARE_HEAD_MERGE(t, i, l) void t##_head_merge##i##_L##l ( \
    input_stream_int16* __restrict a_in, \
    input_stream_int16* __restrict b_in, \
    output_stream_int16* __restrict out)
#endif
#if defined(HEAD_STREAM_OBJ)
DECLARE_HEAD_MERGE(obj, 0, 0);  DECLARE_HEAD_MERGE(obj, 1, 0);
DECLARE_HEAD_MERGE(obj, 0, 1);  DECLARE_HEAD_MERGE(obj, 1, 1);
#endif
#if defined(HEAD_STREAM_CROSS)
DECLARE_HEAD_MERGE(cross, 0, 0);DECLARE_HEAD_MERGE(cross, 1, 0);
DECLARE_HEAD_MERGE(cross, 0, 1);DECLARE_HEAD_MERGE(cross, 1, 1);
#endif

#define DECLARE_CAND_POST(h, l) void cand_attn_head_post_h##h##_L##l ( \
    AIE_IW* __restrict scores_in, \
    AIE_IW* __restrict v_in, \
    AIE_OW* __restrict c_out)
#if defined(HEAD_STREAM_CROSS)
#define DECLARE_CROSS_POST(h, l) void cross_attn_head_post_h##h##_L##l ( \
    AIE_IW* __restrict scores_in, \
    AIE_IW* __restrict v_in, \
    output_stream_int16* __restrict x_out)
#else
#define DECLARE_CROSS_POST(h, l) void cross_attn_head_post_h##h##_L##l ( \
    AIE_IW* __restrict scores_in, \
    AIE_IW* __restrict v_in, \
    AIE_OW* __restrict x_out)
#endif

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
