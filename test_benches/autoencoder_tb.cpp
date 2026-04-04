#include <cstdio>
#include <cmath>
#include <string>
#include "../autoencoder_source/autoencoder.h"
#include "tb_helpers.h"


// 1d comparison helper

template <int DIM>
bool compare_1d(const char* name, const data_t out[DIM], const data_t golden[DIM], float tol = 0.1f) {
    float max_err = 0.0f;
    int worst_idx = 0;
    for (int i = 0; i < DIM; i++) {
        float err = std::fabs((float)out[i] - (float)golden[i]);
        if (err > max_err) {max_err = err; worst_idx = i; }
    }

    bool pass = (max_err < tol);
    printf(" %-30s max_abs_err = %.6f @ [%d] %s\n", name, max_err, worst_idx, pass ? "PASS" : "FAIL");
    return pass;
}



int main() {
    int failures = 0;
    int event_idx = 0;

    std::string wt_dir = "/home/snehadri/repos/unsupervised-search/phase3_export/weights/";
    std::string tv_dir = "/home/snehadri/repos/unsupervised-search/phase3_export/test_vectors/";

    printf("loading weights...\n");
    AEEncoderWeights enc_w;
    AEDecoderWeights dec_w;
    load_ae_encoder_weights(wt_dir, enc_w);
    load_ae_decoder_weights(wt_dir, dec_w);

    printf("loading test vestors...\n");

    // test vector saved as [batch, 14], extract with event_idx
    data_t c0_in[1][AE_IN_DIM], c1_in[1][AE_IN_DIM];
    {
        cnpy::NpyArray arr0 = cnpy::npy_load(tv_dir + "stage5_ae_in_cand0.npy");
        cnpy::NpyArray arr1 = cnpy::npy_load(tv_dir + "stage5_ae_in_cand1.npy");
        float* d0 = arr0.data<float>();
        float* d1 = arr1.data<float>();
        for (int i = 0; i < AE_IN_DIM; i++) {
            c0_in[0][i] = (data_t)d0[event_idx*AE_IN_DIM + i];
            c1_in[0][i] = (data_t)d1[event_idx*AE_IN_DIM + i];
        }
    }

    data_t golden_c0_latent[1][AE_DIM], golden_c1_latent[1][AE_DIM];
    {
        cnpy::NpyArray arr0 = cnpy::npy_load(tv_dir + "stage5_latent_cand0.npy");
        cnpy::NpyArray arr1 = cnpy::npy_load(tv_dir + "stage5_latent_cand1.npy");
        float* d0 = arr0.data<float>();
        float* d1 = arr1.data<float>();
        for (int i = 0; i < AE_DIM; i++) {
            golden_c0_latent[0][i] = (data_t)d0[event_idx*AE_DIM + i];
            golden_c1_latent[0][i] = (data_t)d1[event_idx*AE_DIM + i];
        }
    }

    data_t golden_c0_decoded[1][AE_IN_DIM], golden_c1_decoded[1][AE_IN_DIM];
    {
        cnpy::NpyArray arr0 = cnpy::npy_load(tv_dir + "stage5_ae_decoded_cand0.npy");
        cnpy::NpyArray arr1 = cnpy::npy_load(tv_dir + "stage5_ae_decoded_cand1.npy");
        float* d0 = arr0.data<float>();
        float* d1 = arr1.data<float>();
        for (int i = 0; i < AE_IN_DIM; i++) {
            golden_c0_decoded[0][i] = (data_t)d0[event_idx*AE_IN_DIM + i];
            golden_c1_decoded[0][i] = (data_t)d1[event_idx*AE_IN_DIM + i];
        }
    }

    // golden scalar losses
    float golden_mse_loss, golden_xloss, golden_latent_dist;
    {
        cnpy::NpyArray arr_loss = cnpy::npy_load(tv_dir + "stage6_mse_loss.npy");
        cnpy::NpyArray arr_xloss = cnpy::npy_load(tv_dir + "stage6_mse_crossed_loss.npy");
        cnpy::NpyArray arr_ldist = cnpy::npy_load(tv_dir + "stage6_latent_distance_l2sq.npy");
        golden_mse_loss = arr_loss.data<float>()[event_idx];
        golden_xloss = arr_xloss.data<float>()[event_idx];
        golden_latent_dist = arr_ldist.data<float>()[event_idx];
    }

    // run dual autoencoder

    printf("running dual autoencoder...\n");

    data_t c0_latent[1][AE_DIM], c1_latent[1][AE_DIM];
    data_t c0_decoded[1][AE_IN_DIM], c1_decoded[1][AE_IN_DIM];
    float mse_loss, mse_xloss, latent_dist;

    dual_autoencoder(c0_in, c1_in, enc_w, dec_w, c0_latent, c1_latent,
                        c0_decoded, c1_decoded, mse_loss, mse_xloss, latent_dist);

    printf("\n encoder (latent) \n");

    if(!compare_1d<AE_DIM>("c0_latent", c0_latent[0], golden_c0_latent[0])) failures++;
    if(!compare_1d<AE_DIM>("c1_latent", c1_latent[0], golden_c1_latent[0])) failures++;

    printf("\n decoder (reconstruction) \n");

    if(!compare_1d<AE_IN_DIM>("c0_decoded", c0_decoded[0], golden_c0_decoded[0])) failures++;
    if(!compare_1d<AE_IN_DIM>("c1_decoded", c1_decoded[0], golden_c1_decoded[0])) failures++;

    printf("\n losses \n");

    if(!compare_scalar("mse_loss", mse_loss, golden_mse_loss)) failures++;
    if(!compare_scalar("mse_xloss", mse_xloss, golden_xloss)) failures++;
    if(!compare_scalar("latent_dist", latent_dist, golden_latent_dist)) failures++;

    printf("\n%s: %d failure(s) \n", failures == 0 ? "ALL PASSED" : "FAILED", failures);
    return failures;

}