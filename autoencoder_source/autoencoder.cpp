#include "autoencoder.h"

void dual_autoencoder_top(
    const data_t c0_in[1][AE_IN_DIM],
    const data_t c1_in[1][AE_IN_DIM],

    const AEEncoderWeights &enc_w,
    const AEDecoderWeights &dec_w,

    data_t c0_latent[1][AE_DIM],
    data_t c1_latent[1][AE_DIM],

    data_t c0_decoded[1][AE_IN_DIM],
    data_t c1_decoded[1][AE_IN_DIM],

    float &mse_loss,
    float &mse_crossed_loss,
    float &latent_dist_l2sq
) {
    dual_autoencoder(c0_in, c1_in, enc_w, dec_w, c0_latent, c1_latent, c0_decoded, c1_decoded, mse_loss, mse_crossed_loss, latent_dist_l2sq);
}