#include "pairwise_mlp.h"

void pairwise_mlp_top(
    const data_t w[N_MAX][3],
    const MLPWeights & weights,
    data_t wij[N_MAX][N_MAX]
) {
    pairwise_mlp(w, weights, wij);
}