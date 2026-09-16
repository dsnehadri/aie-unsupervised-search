#define ATTN_TYPE_CROSS
#define ATTN_TYPE_TAG cross
#define ATTN_LAYER 0
#define POST_STAGE_C
#define ROW_HALF_B
#include "../attn_block_aie/kernels/attn_post_kernel.cc"
