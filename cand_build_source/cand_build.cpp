#include "attn_helpers.h"
#include "candidate_build.h"

void candidate_build_top(
    data_t x[N_MAX][E_DIM],
    data_t c[T_DIM][E_DIM],
    int jet_assessment[N_MAX]
) {
    build_candidates<N_MAX>(x, c, jet_assessment);
}