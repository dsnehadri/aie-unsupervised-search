# creates 30 .cc wrapper files that set ATTN_TYPE/ATTN_LAYER/HEAD_IDX
# defines and include the main kernel source

# each AIE compiles to its own binary with distinct weight ROM. using -D
# defines on a single source file lets one codebase specialize for 30 difference
# tiles

# wrappers needed:
# 3 attn types x 2 layers x 4 heads = 24 head wrappers
# 3 attn types x 2 layers = 6 post wrappers

# total : 30

# usage: bash generated_kernel_wrappers.sh

set -e

KERNEL_DIR="${1:-.}"
cd "$KERNEL_DIR"

for TYPE in obj cand cross; do
    TYPE_UPPER=$(echo "$TYPE" | tr 'a-z' 'A-Z')
    for LAYER in 0 1; do
        for HEAD in 0 1 2 3; do
            FNAME="${TYPE}_head${HEAD}_L${LAYER}.cc"
            cat > "$FNAME" <<EOF

// Auto-generated wrapper for ${TYPE} attention, layer ${LAYER}, head ${HEAD}
#define ATTN_TYPE_${TYPE_UPPER}
#define ATTN_LAYER_${LAYER}
#define ATTN_HEAD_${HEAD}
#include "../attn_block_aie/kernels/attn_head_kernel.cc"
EOF
        echo "Generated $FNAME"
    done

    # post wrapper (1 per type/layer)
    FNAME="${TYPE}_post_L${LAYER}.cc"
    cat > "$FNAME" << EOF
//Auto generated wrapper for ${TYPE} attention post kernel, layer ${LAYER}
#define ATTN_TYPE_${TYPE_UPPER}
#define ATTN_LAYER_${LAYER}
#include "../attn_block_aie/kernels/attn_post_kernel.cc"
EOF
        echo "Generated $FNAME"
    done
done

echo ""
echo "Done. 30 wrapper files created in $(pwd)  "