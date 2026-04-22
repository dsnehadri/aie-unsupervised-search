#ifndef TB_HELPERS_H
#define TB_HELPERS_H

// attention block testbench
// loads exported weights and test vectors, runs hls attention blocks and compares against pytorch reference

#include <cmath>
#include <string>
#include "../cnpy/cnpy.h"

#include "../attn_block_pl/attn_block_types.h"
#include "../dnn_block/dnn_block.h"
#include "../autoencoder/autoencoder.h"

// helpers to load npy files

template <typename T, int ROWS, int COLS>
void load_2d(const std::string& path, T arr[ROWS][COLS], int event_idx = 0) {
    cnpy::NpyArray npy = cnpy::npy_load(path);
    float* data = npy.data<float>();
    int offset = event_idx * ROWS * COLS;
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            arr[i][j] = (T)data[offset + i * COLS +  j];
        }
    }
}

template <typename T, int LEN>
void load_1d(const std::string& path, T arr[LEN]) {
    cnpy::NpyArray npy = cnpy::npy_load(path);
    float* data = npy.data<float>();
    for (int i = 0; i < LEN; i++) {
        arr[i] = (T)data[i];
    }
}

void load_padding_mask(const std::string& path, bool mask[N_MAX], int event_idx = 0) {
    cnpy::NpyArray npy = cnpy::npy_load(path);
    unsigned char* raw = npy.data<unsigned char>();
    int offset = event_idx * N_MAX;
    for (int i = 0; i < N_MAX; i++) {
        mask[i] = (raw[offset + i] != 0);
    }
}



template <int ROWS, int COLS = E_DIM, typename T = data_t>
bool compare(const char* name, T out[ROWS][COLS], T golden[ROWS][COLS], const bool* skip_mask = nullptr, float tol = 0.1f) {
    // compare output vs reference

    printf("testing %s\n", name);

    float max_err = 0;
    float sum_sq_err = 0;
    int count = 0;

    for (int i = 0; i < ROWS; i++) {
        if (skip_mask && skip_mask[i]) continue;
        for (int j = 0; j < COLS; j++) {
            float hls_val = (float)out[i][j];
            float ref_val = (float)golden[i][j];
            float err = fabsf(hls_val - ref_val);
            if (err > max_err) max_err = err;
            sum_sq_err += err * err;
            count++;
        }
    }

    float rmse = sqrtf(sum_sq_err / count);
    printf("comparison between hls value and reference");
    printf("max absolute err: %.6f\n", max_err);
    printf("rmse: %.6f\n", rmse);
    printf(" samples compared: %d\n", count);


    // pass fail

    if (max_err < tol) {
        printf("pass: max error %.6f < tolerance %.4f\n", max_err, tol);
        return true;
    } else {
        printf("fail: max error %.6f >= tolerance %.4f\n", max_err, tol);
        return false;
    }
}

bool compare_scalar(const char* name, float computed, float golden, float tol = 0.01f) {
    float err = std::fabs(computed - golden);
    bool pass = (err < tol);
    printf(" %-30s computed = %.6f golden = %.6f err %.6f %s\n", name, computed, golden, err, pass ? "PASS" : "FAIL");
    return pass;
}


void load_attn_weights(const std::string& block, AttnWeights& w) {
    std::string dir = "/home/snehadri/repos/unsupervised-search/phase3_export/";
    std::string weights_suffix = "weights/";
    std::string mha_suffix = "mha_decomposed/";
    load_2d<weight_t, E_DIM, E_DIM>(dir + weights_suffix + mha_suffix + block + "_Wq.npy", w.Wq);
    load_1d<weight_t, E_DIM> (dir + weights_suffix + mha_suffix + block + "_bq.npy", w.bq);
    load_2d<weight_t, E_DIM, E_DIM>(dir + weights_suffix + mha_suffix + block + "_Wk.npy", w.Wk);
    load_1d<weight_t, E_DIM> (dir + weights_suffix + mha_suffix + block + "_bk.npy", w.bk);
    load_2d<weight_t, E_DIM, E_DIM>(dir + weights_suffix + mha_suffix + block + "_Wv.npy", w.Wv);
    load_1d<weight_t, E_DIM> (dir + weights_suffix + mha_suffix + block + "_bv.npy", w.bv);
    load_2d<weight_t, E_DIM, E_DIM>(dir + weights_suffix + mha_suffix + block + "_Wo.npy", w.Wo);
    load_1d<weight_t, E_DIM> (dir + weights_suffix + mha_suffix + block + "_bo.npy", w.bo);
    load_1d<weight_t, E_DIM> (dir + weights_suffix + mha_suffix + block + "_bias_k.npy", w.bias_k);
    load_1d<weight_t, E_DIM> (dir + weights_suffix + mha_suffix + block + "_bias_v.npy", w.bias_v);

    // load post-attention layernorm

    ln_param_t attn_ln_g[E_DIM], attn_ln_b[E_DIM];
    load_1d<ln_param_t, E_DIM>(dir + weights_suffix + block + "_post_attn_norm_weight.npy", w.attn_ln_g);
    load_1d<ln_param_t, E_DIM>(dir + weights_suffix + block + "_post_attn_norm_bias.npy", w.attn_ln_b);

    for (int i = 0; i < N_FFN_LAYERS; i++) {
        int lin_idx = i*3;
        int ln_idx = i*3+1;

        load_2d<weight_t, E_DIM, E_DIM> (
            dir + weights_suffix + block + "_ffwd_" + std::to_string(lin_idx) + "_weight.npy", w.ffn_w[i]);
        load_1d<weight_t, E_DIM> (
            dir + weights_suffix + block + "_ffwd_" + std::to_string(lin_idx) + "_bias.npy", w.ffn_b[i]);
        load_1d<ln_param_t, E_DIM> (
            dir + weights_suffix + block + "_ffwd_" + std::to_string(ln_idx) + "_weight.npy", w.ffn_ln_g[i]);
        load_1d<ln_param_t, E_DIM> (
            dir + weights_suffix + block + "_ffwd_" + std::to_string(ln_idx) + "_bias.npy", w.ffn_ln_b[i]);
    }


    // load post ffn layernorm

    ln_param_t post_ffn_ln_g[E_DIM], post_ffn_ln_b[E_DIM];
    load_1d<ln_param_t, E_DIM>(dir + weights_suffix + block + "_post_ffwd_norm_weight.npy", w.post_ffn_g);
    load_1d<ln_param_t, E_DIM>(dir + weights_suffix + block + "_post_ffwd_norm_bias.npy", w.post_ffn_b);
}


template <int IN_DIM, int HIDDEN, int OUT_DIM, int N_MID>
void load_dnn_block_weights(
    const std::string &wt_dir, 
    const std::string &prefix,
    DNNBlockWeights<IN_DIM, HIDDEN, OUT_DIM, N_MID> &weights
) {

    load_2d<weight_t, HIDDEN, IN_DIM> (wt_dir + prefix + "0_weight.npy", weights.first_w);
    load_1d<weight_t, HIDDEN> (wt_dir + prefix + "0_bias.npy", weights.first_b);
    load_1d<ln_param_t, HIDDEN> (wt_dir + prefix + "1_weight.npy", weights.first_ln_g);
    load_1d<ln_param_t, HIDDEN> (wt_dir + prefix + "1_bias.npy", weights.first_ln_b);

    for (int l = 0; l < N_MID; l++) {
        int lin_idx = (l+1)*3;
        int ln_idx = (l+1)*3+1;

        load_2d<weight_t, HIDDEN, HIDDEN> (wt_dir + prefix + std::to_string(lin_idx) + "_weight.npy", weights.mid_w[l]);
        load_1d<weight_t, HIDDEN> (wt_dir + prefix + std::to_string(lin_idx) + "_bias.npy", weights.mid_b[l]);
        load_1d<ln_param_t, HIDDEN> (wt_dir + prefix + std::to_string(ln_idx) + "_weight.npy", weights.mid_ln_g[l]);
        load_1d<ln_param_t, HIDDEN> (wt_dir + prefix + std::to_string(ln_idx) + "_bias.npy", weights.mid_ln_b[l]);
    }

    // load post ffn layernorm

    int last_idx = (N_MID+1)*3;

    load_2d<weight_t, OUT_DIM, HIDDEN> (wt_dir + prefix + std::to_string(last_idx) + "_weight.npy", weights.last_w);
    load_1d<weight_t, OUT_DIM> (wt_dir + prefix + std::to_string(last_idx) + "_bias.npy", weights.last_b);
}


void load_ae_encoder_weights(const std::string &wt_dir, AEEncoderWeights &w) {
    std::string p = wt_dir + "ae_in_net_";

    // layer 0: index 0 (linear) 1 (LN)
    load_2d<weight_t, AE_D1, AE_D0>(p + "0_weight.npy", w.w0);
    load_1d<weight_t, AE_D1>(p + "0_bias.npy", w.b0);
    load_1d<ln_param_t, AE_D1>(p + "1_weight.npy", w.ln0_g);
    load_1d<ln_param_t, AE_D1>(p + "1_bias.npy", w.ln0_b);

    // layer 1: index 3 (linear) 4 (LN)
    load_2d<weight_t, AE_D2, AE_D1>(p + "3_weight.npy", w.w1);
    load_1d<weight_t, AE_D2>(p + "3_bias.npy", w.b1);
    load_1d<ln_param_t, AE_D2>(p + "4_weight.npy", w.ln1_g);
    load_1d<ln_param_t, AE_D2>(p + "4_bias.npy", w.ln1_b);

    // layer 2: index 6 (linear) 7 (LN)
    load_2d<weight_t, AE_D3, AE_D2>(p + "6_weight.npy", w.w2);
    load_1d<weight_t, AE_D3>(p + "6_bias.npy", w.b2);
    load_1d<ln_param_t, AE_D3>(p + "7_weight.npy", w.ln2_g);
    load_1d<ln_param_t, AE_D3>(p + "7_bias.npy", w.ln2_b);

    // layer 3: index 9 (linear, bare)

    load_2d<weight_t, AE_D4, AE_D3>(p + "9_weight.npy", w.w3);
    load_1d<weight_t, AE_D4>(p + "9_bias.npy", w.b3);

}

void load_ae_decoder_weights(const std::string &wt_dir, AEDecoderWeights &w) {
    std::string p = wt_dir + "ae_out_net_";

    // layer 0: index 0 (linear) 1 (LN)
    load_2d<weight_t, AE_D3, AE_D4>(p + "0_weight.npy", w.w0);
    load_1d<weight_t, AE_D3>(p + "0_bias.npy", w.b0);
    load_1d<ln_param_t, AE_D3>(p + "1_weight.npy", w.ln0_g);
    load_1d<ln_param_t, AE_D3>(p + "1_bias.npy", w.ln0_b);

    // layer 1: index 3 (linear) 4 (LN)
    load_2d<weight_t, AE_D2, AE_D3>(p + "3_weight.npy", w.w1);
    load_1d<weight_t, AE_D2>(p + "3_bias.npy", w.b1);
    load_1d<ln_param_t, AE_D2>(p + "4_weight.npy", w.ln1_g);
    load_1d<ln_param_t, AE_D2>(p + "4_bias.npy", w.ln1_b);

    // layer 2: index 6 (linear) 7 (LN)
    load_2d<weight_t, AE_D1, AE_D2>(p + "6_weight.npy", w.w2);
    load_1d<weight_t, AE_D1>(p + "6_bias.npy", w.b2);
    load_1d<ln_param_t, AE_D1>(p + "7_weight.npy", w.ln2_g);
    load_1d<ln_param_t, AE_D1>(p + "7_bias.npy", w.ln2_b);

    // layer 3: index 9 (linear, bare)

    load_2d<weight_t, AE_D0, AE_D1>(p + "9_weight.npy", w.w3);
    load_1d<weight_t, AE_D0>(p + "9_bias.npy", w.b3);

}

#endif