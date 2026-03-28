#include "embed_ffn.h"

void embed_ffn_top(
    const data_t jets[N_MAX][IN_DIM],
    const bool mask[N_MAX],

    // layer 0 linear (5->16) with bn fused

    const weight_t lin0_w[E_DIM][IN_DIM],
    const weight_t lin0_b[E_DIM],
    const ln_param_t ln0_g[E_DIM],
    const ln_param_t ln0_b[E_DIM],

    // layer 1 linear (16->16) with bn fused

    const weight_t lin1_w[E_DIM][E_DIM],
    const weight_t lin1_b[E_DIM],
    const ln_param_t ln1_g[E_DIM],
    const ln_param_t ln1_b[E_DIM],

    // layer 2 linear (16->16) no norm

    const weight_t lin2_w[E_DIM][E_DIM],
    const weight_t lin2_b[E_DIM],

    // output 
    data_t embed[N_MAX][E_DIM]
) {
    embed_ffn(jets, mask,
    lin0_w, lin0_b, ln0_g, ln0_b,
    lin1_w, lin1_b, ln1_g, ln1_b,
    lin2_w, lin2_b, embed);
}