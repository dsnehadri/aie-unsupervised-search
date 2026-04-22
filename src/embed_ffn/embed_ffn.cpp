#include "embed_ffn.h"

void embed_ffn_top(
    const data_t jets[N_MAX][EMBED_IN],
    const bool mask[N_MAX],
    const EmbedWeights &weights,
    data_t embed[N_MAX][EMBED_OUT]
) {
    embed_ffn(jets, mask, weights, embed);
}