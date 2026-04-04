#ifndef PASSWD_H
#define PASSWD_H

#include "../embed_ffn_source/embed_ffn.h"
#include "../pairwise_mlp_source/pairwise_mlp.h"
#include "../cand_lorentz_source/cand_lorentz.h"
#include "../autoencoder_source/autoencoder.h"
#include "../cand_build_source/candidate_build.h"

#include "../attn_block_source/attn_block_obj.h"
#include "../attn_block_source/attn_block_cand.h"
#include "../attn_block_source/attn_block_cross.h"

inline void passwd_pipeline(
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

    // stage 1: embedded ffn - raws jets (12x5) -> embeddings (12x16)
    data_t x[N_MAX][E_DIM];
    embed_ffn(raw_jets, mask, embed_w, x);

    // stage 2: pairwise mlp - angular features -> Wij bias(12x12)

    data_t w_ang[N_MAX][3];
    for (int j = 0; j < N_MAX; j++) {
        #pragma HLS PIPELINE II=1
        w_ang[j][0] = raw_jets[j][1]; // eta
        w_ang[j][1] = raw_jets[j][2]; // cos(phi)
        w_ang[j][2] = raw_jets[j][3]; // sin(phi)
    }

    data_t wij[N_MAX][N_MAX];
    pairwise_mlp(w_ang, mlp_w, wij);

    // expand to multi-head format

    score_t wij_bias[N_HEADS * N_MAX][N_KV];
    expand_wij(wij, wij_bias);

    // stage 3: layer 0: obj -> build_cand -> cand -> cross

    attn_block_obj(x, mask, wij_bias, true,
        obj0_w.Wq, obj0_w.bq, obj0_w.Wk, obj0_w.bk, obj0_w.Wv, obj0_w.bv,
        obj0_w.bias_k, obj0_w.bias_v, obj0_w.Wo, obj0_w.bo, 
        obj0_w.attn_ln_g, obj0_w.attn_ln_b, 
        obj0_w.ffn_w, obj0_w.ffn_b, obj0_w.ffn_ln_g, obj0_w.ffn_ln_b,
        obj0_w.post_ffn_g, obj0_w.post_ffn_b);
    remask(x, mask);

    // build embedded candidates (ISR shift + argmax + matmul)
    data_t c[T_DIM][E_DIM];
    int jet_assign_tmp[N_MAX];
    build_candidates<N_MAX>(x, c, jet_assign_tmp);

    // candidate self attention

    attn_block_cand(c,
        cand0_w.Wq, cand0_w.bq, cand0_w.Wk, cand0_w.bk, cand0_w.Wv, cand0_w.bv,
        cand0_w.bias_k, cand0_w.bias_v, cand0_w.Wo, cand0_w.bo, 
        cand0_w.attn_ln_g, cand0_w.attn_ln_b, 
        cand0_w.ffn_w, cand0_w.ffn_b, cand0_w.ffn_ln_g, cand0_w.ffn_ln_b,
        cand0_w.post_ffn_g, cand0_w.post_ffn_b);

    // cross attention

    attn_block_cross(x, c,
        cross0_w.Wq, cross0_w.bq, cross0_w.Wk, cross0_w.bk, cross0_w.Wv, cross0_w.bv,
        cross0_w.bias_k, cross0_w.bias_v, cross0_w.Wo, cross0_w.bo, 
        cross0_w.attn_ln_g, cross0_w.attn_ln_b, 
        cross0_w.ffn_w, cross0_w.ffn_b, cross0_w.ffn_ln_g, cross0_w.ffn_ln_b,
        cross0_w.post_ffn_g, cross0_w.post_ffn_b);
    remask(x, mask);

    // stage 3, layer 1: obj -> build_cand -> cand -> cross
    
    attn_block_obj(x, mask, wij_bias, true,
        obj1_w.Wq, obj1_w.bq, obj1_w.Wk, obj1_w.bk, obj1_w.Wv, obj1_w.bv,
        obj1_w.bias_k, obj1_w.bias_v, obj1_w.Wo, obj1_w.bo, 
        obj1_w.attn_ln_g, obj1_w.attn_ln_b, 
        obj1_w.ffn_w, obj1_w.ffn_b, obj1_w.ffn_ln_g, obj1_w.ffn_ln_b,
        obj1_w.post_ffn_g, obj1_w.post_ffn_b);
    remask(x, mask);

    // build embedded candidates (ISR shift + argmax + matmul)
    build_candidates<N_MAX>(x, c, jet_assign_tmp);

    // candidate self attention

    attn_block_cand(c,
        cand1_w.Wq, cand1_w.bq, cand1_w.Wk, cand1_w.bk, cand1_w.Wv, cand1_w.bv,
        cand1_w.bias_k, cand1_w.bias_v, cand1_w.Wo, cand1_w.bo, 
        cand1_w.attn_ln_g, cand1_w.attn_ln_b, 
        cand1_w.ffn_w, cand1_w.ffn_b, cand1_w.ffn_ln_g, cand1_w.ffn_ln_b,
        cand1_w.post_ffn_g, cand1_w.post_ffn_b);

    // cross attention

    attn_block_cross(x, c,
        cross1_w.Wq, cross1_w.bq, cross1_w.Wk, cross1_w.bk, cross1_w.Wv, cross1_w.bv,
        cross1_w.bias_k, cross1_w.bias_v, cross1_w.Wo, cross1_w.bo, 
        cross1_w.attn_ln_g, cross1_w.attn_ln_b, 
        cross1_w.ffn_w, cross1_w.ffn_b, cross1_w.ffn_ln_g, cross1_w.ffn_ln_b,
        cross1_w.post_ffn_g, cross1_w.post_ffn_b);
    remask(x, mask);

    // stage 4 candidate building in lorentz space

    float jp4[N_MAX][P4_DIM];
    int jet_assign[N_MAX];
    float cand_p4[T_DIM][P4_DIM];
    float cand_mass_scaled[T_DIM];
    data_t ae_input[T_DIM][AE_IN_DIM];

    cand_lorentz(raw_jets, x, c, mask, jp4, jet_assign, cand_p4, cand_mass_scaled, ae_input);

    // stage 5-6 dual autoencoder + loss scalars

    // extract cand 0 and 1 as row vectors [1][AE_IN_DIM]

    data_t c0_in[1][AE_IN_DIM], c1_in[1][AE_IN_DIM];
    for (int i = 0; i < AE_IN_DIM; i++) {
        #pragma HLS PIPELINE II=1
        c0_in[0][i] = ae_input[0][i];
        c1_in[0][i] = ae_input[1][i];
    }

    data_t c0_latent[1][AE_DIM], c1_latent[1][AE_DIM];
    data_t c0_decoded[1][AE_IN_DIM], c1_decoded[1][AE_IN_DIM];

    dual_autoencoder(c0_in, c1_in, ae_enc_w, ae_dec_w,
                        c0_latent, c1_latent, c0_decoded, c1_decoded,
                        mse_loss, mse_crossed_loss, latent_dist);
}


#endif