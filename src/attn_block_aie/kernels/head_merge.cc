// HEAD_STREAM: pair two heads' output rows into one stream.
//
// The projection kernel needs a full row, [h0 h1 h2 h3] of D_HEAD words each,
// but a tile has only two stream inputs, so the four head-post kernels cannot
// feed it directly. Two of these merge the heads pairwise and the projection
// reads one half-row from each.
//
// A head emits four rows at a time, D_HEAD words per row:
//   a = [a_r0 a_r1 a_r2 a_r3], b = [b_r0 b_r1 b_r2 b_r3]
//   zip(a, b, 4) -> [a_r0 b_r0 a_r1 b_r1] , [a_r2 b_r2 a_r3 b_r3]
// which is exactly rows 0..3 as [heads paired], eight words each.
#include "attn_aie_types.h"
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <adf.h>
#include "win_vec.h"

// one distinct symbol per (type, pair, layer): the compiler dedups tile
// wrappers by function identity
#define _MRG_3(t, i, l) t##_head_merge##i##_L##l
#define _MRG_2(t, i, l) _MRG_3(t, i, l)
#define HEAD_MERGE_FN   _MRG_2(ATTN_TYPE_TAG, MERGE_IDX, ATTN_LAYER)

static inline void head_merge_body(input_stream_int16* __restrict a_in,
                                   input_stream_int16* __restrict b_in,
                                   output_stream_int16* __restrict out)
{
    static_assert(D_HEAD * 4 == 16, "a group of four rows is one 16-lane vector per head");
    for (int g = 0; g < N_MAX / 4; g++) {
        const v16_t a = stream_read16(a_in);
        const v16_t b = stream_read16(b_in);
        const auto z = aie::interleave_zip(a, b, 4);
        stream_write16(out, z.first);    // rows 4g+0, 4g+1
        stream_write16(out, z.second);   // rows 4g+2, 4g+3
    }
}

void HEAD_MERGE_FN(input_stream_int16* __restrict a_in,
                   input_stream_int16* __restrict b_in,
                   output_stream_int16* __restrict out)
{
    head_merge_body(a_in, b_in, out);
}
