#include "passwd.h"

void passwd_pipeline_top(
    const data_t raw_jets[N_MAX][RAW_DIM],
    const bool mask[N_MAX],

    const EmbedWeights &embed_w,
    const MLPWeights &mlp_w,

    const AttnWeights &obj0_w, const AttnWeights &cand0_w, const AttnWeights &cross0_w,
    const AttnWeights &obj1_w, const AttnWeights &cand1_w, const AttnWeights &cross1_w,

    const AEEncoderWeights &ae_enc_w,
    const AEDecoderWeights & ae_dec_w,

    float &mse_loss,
    float &mse_crossed_loss,
    float &latent_dist
) {
    passwd_pipeline(
        raw_jets, mask,
        embed_w, mlp_w,
        obj0_w, cand0_w, cross0_w,
        obj1_w, cand1_w, cross1_w,
        ae_enc_w, ae_dec_w, 
        mse_loss, mse_crossed_loss, latent_dist
    );
}