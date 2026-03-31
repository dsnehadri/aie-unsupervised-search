#include "cand_lorentz.h"

void cand_lorentz_top(
    const data_t raw_x[N_MAX][RAW_DIM], // raw jet features
    const data_t attn_x[N_MAX][E_DIM], // last cross-attn output
    const data_t cand_embed[T_DIM][E_DIM], // last cand_blocks output
    const bool mask[N_MAX], //padding mask

    float jp4[N_MAX][P4_DIM], // jet 4 momenta
    int jet_assign[N_MAX],  // jet -> cand assignment
    float cand_p4[T_DIM][P4_DIM], // candidate 4 momenta
    float cand_mass_scaled[T_DIM], // scaled invariant mass 
    data_t ae_input[T_DIM][AE_IN_DIM] // assembed ae input features
) {
    cand_lorentz(raw_x, attn_x, cand_embed, mask, jp4, jet_assign, cand_p4, cand_mass_scaled, ae_input);
}