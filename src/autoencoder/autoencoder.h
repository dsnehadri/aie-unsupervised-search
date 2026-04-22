#ifndef AUTOENCODER_H
#define AUTOENCODER_H

#include "../attn_block_pl/attn_block_types.h"
#include "../attn_block_pl/attn_helpers.h"

// dnn block doesnt supprot hidden layers of different widths, so we'll just do it manually here

struct AEEncoderWeights {
    // layer 0: 14 -> 11

    weight_t w0[AE_D1][AE_D0];
    weight_t b0[AE_D1];
    ln_param_t ln0_g[AE_D1];
    ln_param_t ln0_b[AE_D1];

    // layer 1: 11 -> 8 

    weight_t w1[AE_D2][AE_D1];
    weight_t b1[AE_D2];
    ln_param_t ln1_g[AE_D2];
    ln_param_t ln1_b[AE_D2];

    // layer 2: 8 -> 5

    weight_t w2[AE_D3][AE_D2];
    weight_t b2[AE_D3];
    ln_param_t ln2_g[AE_D3];
    ln_param_t ln2_b[AE_D3];

    // layer 3: 5 -> 2

    weight_t w3[AE_D4][AE_D3];
    weight_t b3[AE_D4];

};

struct AEDecoderWeights {
    // layer 0: 14 -> 11

    weight_t w0[AE_D3][AE_D4];
    weight_t b0[AE_D3];
    ln_param_t ln0_g[AE_D3];
    ln_param_t ln0_b[AE_D3];

    // layer 1: 11 -> 8 

    weight_t w1[AE_D2][AE_D3];
    weight_t b1[AE_D2];
    ln_param_t ln1_g[AE_D2];
    ln_param_t ln1_b[AE_D2];

    // layer 2: 8 -> 5

    weight_t w2[AE_D1][AE_D2];
    weight_t b2[AE_D1];
    ln_param_t ln2_g[AE_D1];
    ln_param_t ln2_b[AE_D1];

    // layer 3: 5 -> 2

    weight_t w3[AE_D0][AE_D1];
    weight_t b3[AE_D0];

};


inline void ae_encode(
    const data_t x[1][AE_IN_DIM],
    const AEEncoderWeights &enc_w,
    data_t latent[1][AE_DIM]
) {
    // intermediate buffers at cascade width
    data_t buf_d1[1][AE_D1]; // after layer 0: dim 11
    data_t buf_d2[1][AE_D2]; // after layer 1: dim 8
    data_t buf_d3[1][AE_D3]; // after layer 2: dim 5

    // layer 0 14 -> 11 + LN + ReLU
    linear<1, AE_D1, AE_D0>(x, enc_w.w0, enc_w.b0, buf_d1);
    layernorm<1, AE_D1>(buf_d1, enc_w.ln0_g, enc_w.ln0_b);
    relu_2d<1, AE_D1>(buf_d1);

    // layer 1 11 -> 8 + LN + ReLU
    linear<1, AE_D2, AE_D1>(buf_d1, enc_w.w1, enc_w.b1, buf_d2);
    layernorm<1, AE_D2>(buf_d2, enc_w.ln1_g, enc_w.ln1_b);
    relu_2d<1, AE_D2>(buf_d2);

    // layer 2 8 -> 5 + LN + ReLU
    linear<1, AE_D3, AE_D2>(buf_d2, enc_w.w2, enc_w.b2, buf_d3);
    layernorm<1, AE_D3>(buf_d3, enc_w.ln2_g, enc_w.ln2_b);
    relu_2d<1, AE_D3>(buf_d3);

    // layer 3: 5 -> 2
    linear<1, AE_D4, AE_D3>(buf_d3, enc_w.w3, enc_w.b3, latent);
}

inline void ae_decode(
    const data_t latent[1][AE_DIM],
    const AEDecoderWeights &dec_w,
    data_t recon[1][AE_IN_DIM]
) {
    // intermediate buffers at cascade width
    
    data_t buf_d3[1][AE_D3]; // after layer 0: dim 5
    data_t buf_d2[1][AE_D2]; // after layer 1: dim 8
    data_t buf_d1[1][AE_D1]; // after layer 2: dim 11

    // layer 0 2 -> 5 + LN + ReLU
    linear<1, AE_D3, AE_D4>(latent, dec_w.w0, dec_w.b0, buf_d3);
    layernorm<1, AE_D3>(buf_d3, dec_w.ln0_g, dec_w.ln0_b);
    relu_2d<1, AE_D3>(buf_d3);

    // layer 1 5 -> 8 + LN + ReLU
    linear<1, AE_D2, AE_D3>(buf_d3, dec_w.w1, dec_w.b1, buf_d2);
    layernorm<1, AE_D2>(buf_d2, dec_w.ln1_g, dec_w.ln1_b);
    relu_2d<1, AE_D2>(buf_d2);

    // layer 2 8 -> 11 + LN + ReLU
    linear<1, AE_D1, AE_D2>(buf_d2, dec_w.w2, dec_w.b2, buf_d1);
    layernorm<1, AE_D1>(buf_d1, dec_w.ln2_g, dec_w.ln2_b);
    relu_2d<1, AE_D1>(buf_d1);

    // layer 3: 11 -> 14
    linear<1, AE_D0, AE_D1>(buf_d1, dec_w.w3, dec_w.b3, recon);
}


// dual auto encoder, processes both candidates and computes losses

inline void dual_autoencoder(
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
    ae_encode(c0_in, enc_w, c0_latent);
    ae_encode(c1_in, enc_w, c1_latent);

    ae_decode(c0_latent, dec_w, c0_decoded);
    ae_decode(c1_latent, dec_w, c1_decoded);

    // mse loss

    float mse_c0_c0 = 0.0f, mse_c1_c1 = 0.0f;
    float mse_c0_c1 = 0.0f, mse_c1_c0 = 0.0f;
    for (int i = 0; i < AE_IN_DIM; i++) {
        #pragma HLS PIPELINE II=1
        float d;
        // reconstruction c0 vs c0_decoded
        d = (float)c0_in[0][i] - (float)c0_decoded[0][i];
        mse_c0_c0 += d * d;
        // reconstruction c1 vs c1_decoded
        d = (float)c1_in[0][i] - (float)c1_decoded[0][i];
        mse_c1_c1 += d * d;
        // crossed: c0 vs c1 decoded
        d = (float)c0_in[0][i] - (float)c1_decoded[0][i];
        mse_c0_c1 += d * d;

        // crossed: c1 vs c0 decoded
        d = (float)c1_in[0][i] - (float)c0_decoded[0][i];
        mse_c1_c0 += d * d;
    }

    float inv_dim = 1.0f / AE_IN_DIM;
    mse_loss = (mse_c0_c0 + mse_c1_c1) * inv_dim;
    mse_crossed_loss = (mse_c0_c1 + mse_c1_c0) * inv_dim;

    // latent distance

    float dist = 0.0f;
    for (int i = 0; i < AE_DIM; i++) {
        #pragma HLS PIPELINE II=1
        float d = (float)c0_latent[0][i] - (float)c1_latent[0][i];
        dist += d*d;
    }
    latent_dist_l2sq = dist;
}

#endif