// Embedding-only graph: x86sim numerics check and aiesim cycle profile.
#include <adf.h>
#include "../../attn_block_aie/graph/embed_graph.h"

EmbedGraph embed_graph;

#ifndef AIE_NUM_EVENTS
#define AIE_NUM_EVENTS 4
#endif

int main(void) {
    embed_graph.init();
    embed_graph.run(AIE_NUM_EVENTS);
    embed_graph.end();
    return 0;
}
