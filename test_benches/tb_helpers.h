#ifndef TB_HELPERS_H
#define TB_HELPERS_H

// attention block testbench
// loads exported weights and test vectors, runs hls attention blocks and compares against pytorch reference

#include <cmath>
#include <string>
#include "../attn_block_source/attn_block_types.h"
#include "../cnpy/cnpy.h"

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


template <int ROWS>
bool compare(const char* name, data_t out[ROWS][E_DIM], data_t golden[ROWS][E_DIM], const bool* skip_mask = nullptr, float tol = 0.1f) {
    // compare output vs reference

    float max_err = 0;
    float sum_sq_err = 0;
    int count = 0;

    for (int i = 0; i < ROWS; i++) {
        if (skip_mask && skip_mask[i]) continue;
        for (int j = 0; j < E_DIM; j++) {
            float hls_val = (float)out[i][j];
            float ref_val = (float)golden[i][j];
            float err = fabsf(hls_val - ref_val);
            if (err > max_err) max_err = err;
            sum_sq_err += err * err;
            count++;
        }
    }
    // After attn_block_obj returns:
    printf("HLS out[0][0..3]: %f %f %f %f\n",
    (float)out[0][0], (float)out[0][1], (float)out[0][2], (float)out[0][3]);

    float rmse = sqrtf(sum_sq_err / count);
    printf("comparison between hls value and reference");
    printf("max absolute err: %.6f\n", max_err);
    printf("rmse: %.6f\n", rmse);
    printf(" samples compared: %d\n", count);


    // pass fail

    float TOLERANCE = 0.1;
    if (max_err < TOLERANCE) {
        printf("pass: max error %.6f < tolerance %.4f\n", max_err, TOLERANCE);
        return true;
    } else {
        printf("fail: max error %.6f >= tolerance %.4f\n", max_err, TOLERANCE);
        return false;
    }
}



struct attn_weights {
    weight_t Wq[E_DIM][E_DIM], Wk[E_DIM][E_DIM], Wv[E_DIM][E_DIM], Wo[E_DIM][E_DIM];
    weight_t bq[E_DIM], bk[E_DIM], bv[E_DIM], bo[E_DIM];
    weight_t bias_k[E_DIM], bias_v[E_DIM];
    ln_param_t attn_ln_g[E_DIM], attn_ln_b[E_DIM];
    weight_t ffn_w[N_FFN_LAYERS][E_DIM][E_DIM];
    weight_t ffn_b[N_FFN_LAYERS][E_DIM];
    ln_param_t ffn_ln_g[N_FFN_LAYERS][E_DIM];
    ln_param_t ffn_ln_b[N_FFN_LAYERS][E_DIM];
    ln_param_t post_ffn_g[E_DIM], post_ffn_b[E_DIM];

};

void load_attn_weights(const std::string& block, attn_weights& w) {
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

    printf("Loading Wq from: %s\n", (dir + weights_suffix + mha_suffix + block + "_Wq.npy").c_str());
    printf("bq[0..3]: %f %f %f %f\n", (float)w.bq[0], (float)w.bq[1], (float)w.bq[2], (float)w.bq[3]);

    // load post-attention layernorm

    ln_param_t attn_ln_g[E_DIM], attn_ln_b[E_DIM];
    load_1d<ln_param_t, E_DIM>(dir + weights_suffix + block + "_post_attn_norm_weight.npy", w.attn_ln_g);
    load_1d<ln_param_t, E_DIM>(dir + weights_suffix + block + "_post_attn_norm_bias.npy", w.attn_ln_b);

    for (int i = 0; i < N_FFN_LAYERS; i++) {
        int lin_idx = i*3;
        int ln_idx = i*3+1;

        printf("Loading FFN layer %d: lin_idx=%d, ln_idx=%d\n", i, lin_idx, ln_idx);
        printf("  %s\n", (dir + weights_suffix + block + "_ffwd_" + std::to_string(lin_idx) + "_weight.npy").c_str());

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

#endif