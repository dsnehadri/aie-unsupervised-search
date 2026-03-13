// attention block testbench
// loads exported weights and test vectors, runs hls attention blocks and compares against pytorch reference

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <string>
#include "attn_block_types.h"

#include "cnpy.h"

// forward declaration of the dut

extern void attn_block_obj(
    data_t x[N_MAX][E_DIM], // input embeddings, which are modified in place

    // masks

    const bool padding_mask[N_MAX],
    const score_t wij_bias[N_MAX * N_HEADS][N_KV],
    const bool use_wij,

    // MHA weights

    const weight_t Wq[E_DIM][E_DIM], const weight_t bq[E_DIM],
    const weight_t Wk[E_DIM][E_DIM], const weight_t bk[E_DIM],
    const weight_t Wv[E_DIM][E_DIM], const weight_t bv[E_DIM],
    const weight_t bias_k[E_DIM], const weight_t bias_v[E_DIM],
    const weight_t Wo[E_DIM][E_DIM], const weight_t bo[E_DIM], // output projections

    // post attention layer norm
    const ln_param_t attn_ln_g[E_DIM], const ln_param_t attn_ln_b[E_DIM],

    // ffn weights: n_ffn_layers 
    const weight_t ffn_w[N_FFN_LAYERS][E_DIM][E_DIM],
    const weight_t ffn_b[N_FFN_LAYERS][E_DIM],
    const ln_param_t ffn_ln_g[N_FFN_LAYERS][E_DIM],
    const ln_param_t ffn_ln_b[N_FFN_LAYERS][E_DIM],

    // post ffn layernorm after skip connection

    const ln_param_t post_ffn_g[E_DIM],
    const ln_param_t post_ffn_b[E_DIM]

);

// helpers to load npy files

template <typename T, int ROWS, int COLS>
void load_2d(const std::string& path, T arr[ROWS][COLS]) {
    cnpy::NpyArray npy = cnpy::npy_load(path);
    float* data = npy.data<float>();
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            arr[i][j] = (T)data[i * COLS + j];
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

int main() {
    std::string dir = "/home/snehadri/repos/unsupervised-search/phase3_export/";
    std::string block = "obj_blocks.0";


    // mha projection weights

    weight_t Wq[E_DIM][E_DIM], Wk[E_DIM][E_DIM], Wv[E_DIM][E_DIM], Wo[E_DIM][E_DIM];
    weight_t bq[E_DIM], bk[E_DIM], bv[E_DIM], bo[E_DIM];
    weight_t bias_k[E_DIM], bias_v[E_DIM];

    load_2d<weight_t, E_DIM, E_DIM>(dir + block + ".attn.Wq.npy", Wq);
    load_1d<weight_t, E_DIM> (dir + block + ".attn.bq.npy", bq);
    load_2d<weight_t, E_DIM, E_DIM>(dir + block + ".attn.Wk.npy", Wk);
    load_1d<weight_t, E_DIM> (dir + block + ".attn.bk.npy", bk);
    load_2d<weight_t, E_DIM, E_DIM>(dir + block + ".attn.Wv.npy", Wv);
    load_1d<weight_t, E_DIM> (dir + block + ".attn.bv.npy", bv);
    load_2d<weight_t, E_DIM, E_DIM>(dir + block + ".attn.Wo.npy", Wo);
    load_1d<weight_t, E_DIM> (dir + block + ".attn.bo.npy", bo);
    load_1d<weight_t, E_DIM> (dir + block + ".attn.bias_k.npy", bias_k);
    load_1d<weight_t, E_DIM> (dir + block + ".attn.bias_v.npy", bias_v);

    // load post-attention layernorm

    ln_param_t attn_ln_g[E_DIM], attn_ln_b[E_DIM];
    load_1d<ln_param_t, E_DIM>(dir + block + ".post_attn_norm.weight.npy", attn_ln_g);
    load_1d<ln_param_t, E_DIM>(dir + block + ".post_attn_norm.bias.npy", attn_ln_b);

    // load ffn weights, pytorch sequential indices (0 = linear, 1 = layer norm, 2 = relu, 3 = linear, etc.)

    weight_t ffn_w[N_FFN_LAYERS][E_DIM][E_DIM];
    weight_t ffn_b[N_FFN_LAYERS][E_DIM];
    ln_param_t ffn_ln_g[N_FFN_LAYERS][E_DIM];
    ln_param_t ffn_ln_b[N_FFN_LAYERS][E_DIM];

    for (int i = 0; i < N_FFN_LAYERS; i++) {
        int lin_idx = i*3;
        int ln_idx = i*3+1;

        load_2d<weight_t, E_DIM, E_DIM> (
            dir + block + ".ffwd." + std::to_string(lin_idx) + ".weight.npy", ffn_w[i]);
        load_1d<weight_t, E_DIM> (
            dir + block + ".ffwd." + std::to_string(lin_idx) + ".bias.npy", ffn_b[i]);
        load_1d<ln_param_t, E_DIM> (
            dir + block + ".ffwd." + std::to_string(ln_idx) + ".weight.npy", ffn_ln_g[i]);
        load_1d<ln_param_t, E_DIM> (
            dir + block + ".ffwd." + std::to_string(ln_idx) + ".bias.npy", ffn_ln_b[i]);
    }

    // load post ffn layernorm

    ln_param_t post_ffn_ln_g[E_DIM], post_ffn_ln_b[E_DIM];
    load_1d<ln_param_t, E_DIM>(dir + block + ".post_ffwd_norm.weight.npy", post_ffn_ln_g);
    load_1d<ln_param_t, E_DIM>(dir + block + ".post_ffwd_norm.bias.npy", post_ffn_ln_b);

    // load test vectors

    data_t x[N_MAX][E_DIM];
    data_t golden[N_MAX][E_DIM];
    load_2d<data_t, N_MAX, E_DIM>(dir + "test_vec_obj_block_0_in.npy", x);
    load_2d<data_t, N_MAX, E_DIM>(dir + "test_vec_obj_block_0_out.npy", golden);

    // build padding mask from test input

    bool padding_mask[N_MAX];
    {
        cnpy::NpyArray npy = cnpy::npy_load(dir + "test_vec_obj_block_0_in.npy");
        float* raw = npy.data<float>();
        for (int i = 0; i < N_MAX; i++) {
            padding_mask[i] = (raw[i * E_DIM] == 0.0f);
        }
    }

    // load wij bias

    score_t wij_bias[N_HEADS * N_MAX][N_KV] = {0};
    bool use_wij = false;

    // if testing with wij enables
    // load_2d<score_t, N_HEADS*N_MAX, N_KV>(dir + "test_vec_wij_block_0.npy", wij_bias)
    // use_wij = true;

    // run the device under test

    attn_block_obj(
        x, 
        padding_mask,
        wij_bias, use_wij,
        Wq, bq, Wk, bk, Wv, bv,
        bias_k, bias_v,
        Wo, bo,
        attn_ln_g, attn_ln_b,
        ffn_w, ffn_b, ffn_ln_g, ffn_ln_b,
        post_ffn_ln_g, post_ffn_ln_b
    );

    // compare output vs reference

    float max_err = 0;
    float sum_sq_err = 0;
    int count = 0;

    for (int i = 0; i < N_MAX; i++) {
        if (padding_mask[i]) continue;
        for (int j = 0; j < E_DIM; j++) {
            float hls_val = (float)x[i][j];
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

    float TOLERANCE = 0.1;
    if (max_err < TOLERANCE) {
        printf("pass: max error %.6f < tolerance %.4f\n", max_err, TOLERANCE);
        return 0;
    } else {
        printf("fail: max error %.6f >= tolerance %.4f\n", max_err, TOLERANCE);
        return 1;
    }

    





}