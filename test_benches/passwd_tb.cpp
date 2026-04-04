#include <cstdio>
#include <cmath>
#include <string>
#include "../passwd_source/passwd.h"
#include "tb_helpers.h"

std::string wt_dir = "/home/snehadri/repos/unsupervised-search/phase3_export/weights/";
std::string tv_dir = "/home/snehadri/repos/unsupervised-search/phase3_export/test_vectors/";
static const int EVENT_IDX = 0;

int main() {
    int failures = 0;
    printf("passwd abc testbench\n");

    printf("loading weights...\n");

    // embedding ffn

    EmbedWeights embed_w;
    load_dnn_block_weights<EMBED_IN, EMBED_HIDDEN, EMBED_OUT, EMBED_N_MID>(
        wt_dir, "embed_net_", embed_w);

    // pairwise mlp
    MLPWeights mlp_w;
    load_dnn_block_weights<MLP_IN, MLP_HIDDEN, MLP_OUT, MLP_N_MID>(
        wt_dir, "mlp_net_", mlp_w);

    // attention blocks

    AttnWeights obj0_w, cand0_w, cross0_w;
    AttnWeights obj1_w, cand1_w, cross1_w;

    load_attn_weights("obj_blocks_0", obj0_w);
    load_attn_weights("cand_blocks_0", cand0_w);
    load_attn_weights("cross_blocks_0", cross0_w);
    load_attn_weights("obj_blocks_1", obj1_w);
    load_attn_weights("cand_blocks_1", cand1_w);
    load_attn_weights("cross_blocks_1", cross1_w);

    // autoencoder

    AEEncoderWeights ae_enc_w;
    AEDecoderWeights ae_dec_w;
    load_ae_encoder_weights(wt_dir, ae_enc_w);
    load_ae_decoder_weights(wt_dir, ae_dec_w);

    printf("all weights loaded\n");

    // load input test vectors

    printf("loading test vectors (event %d) \n", EVENT_IDX);

    data_t raw_jets[N_MAX][RAW_DIM];
    load_2d<data_t, N_MAX, RAW_DIM>(tv_dir + "stage0_input_raw.npy", raw_jets, EVENT_IDX);

    bool mask[N_MAX];
    load_padding_mask(tv_dir + "stage0_padding_mask.npy", mask, EVENT_IDX);

    // load golden outputs

    float golden_mse, golden_xloss, golden_ldist;
    {
        cnpy::NpyArray a1 = cnpy::npy_load(tv_dir + "stage6_mse_loss.npy");
        cnpy::NpyArray a2 = cnpy::npy_load(tv_dir + "stage6_mse_crossed_loss.npy");
        cnpy::NpyArray a3 = cnpy::npy_load(tv_dir + "stage6_latent_distance_l2sq.npy");

        golden_mse = a1.data<float>()[EVENT_IDX];
        golden_xloss = a2.data<float>()[EVENT_IDX];
        golden_ldist = a3.data<float>()[EVENT_IDX];
    }

    // run full pipeline

    printf("running passwd pipeline \n");

    float mse_loss, mse_xloss, latent_dist;
    passwd_pipeline(
        raw_jets, mask,
        embed_w, mlp_w,
        obj0_w, cand0_w, cross0_w,
        obj1_w, cand1_w, cross1_w,
        ae_enc_w, ae_dec_w, 
        mse_loss, mse_xloss, latent_dist
    );

    // check final outputs

    printf("final loss scalars\n");

    if (!compare_scalar("mse_loss", mse_loss, golden_mse)) failures++;
    if (!compare_scalar("mse_xloss", mse_xloss, golden_xloss)) failures++;
    if (!compare_scalar("latent_dist", latent_dist, golden_ldist)) failures++;

    // summary

    printf("\n%s: %d failure(s)\n", failures == 0 ? "ALL PASSED" : "FAILED", failures);

    return failures;
}