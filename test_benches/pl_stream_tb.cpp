// testbench for DATAFLOW stream-based pipeline

// same weight-laoding and golden reference checks as passwd_tb.cpp
// but input/output go through hls::stream<axi_word_t> instead of direct
// array arguments

#include "../pl_stream_source/pl_stream.h"
#include "tb_helpers.h"
#include <cstdio>
#include <cmath>
#include <string>

// forward declaration of DUT

void pl_stream_top(
    ap_uint<32>* in_buf,
    ap_uint<32>* out_buf,
    int n_events
);

int main() {
    int failures = 0;
    const int EVENT_IDX = 0;

    std::string tv_dir = "/home/snehadri/repos/unsupervised-search/phase3_export/test_vectors/";

    printf("passwd abc stream pipeline test\n");

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

    // pack input into memory mapped buffer
    ap_uint<32> in_buf[72];
    ap_uint<32> out_buf[3];

    // write raw jets

    for (int i = 0; i < N_MAX; i++) {
        for (int j = 0; j < RAW_DIM; j++) {
            data_t val = raw_jets[i][j];
            ap_uint<16> bits = val.range(15, 0);
            in_buf[i * RAW_DIM + j] = (ap_uint<32>)bits;
        }
    }

    // write mask

    for (int i = 0; i <N_MAX; i++) {
        in_buf[60 + i] = mask[i] ? 1 : 0;
    }

    // run DUT

    pl_stream_top(in_buf, out_buf, 1);

    // unpack output

    float hw_mse = *(float*)&out_buf[0];
    float hw_xloss = *(float*)&out_buf[1];
    float hw_ldist = *(float*)&out_buf[2];

    // const float TOL = 1e-3f; //tolerance for wide ap_fixed<32,12>
    const float TOL = 0.5f; //tolerance for ap_fixed<16,5>

    printf("\n%-20s %12s %12s %12s %s\n",
        "Metric", "HW", "Golden", "AbsErr", "Status");

    printf("\n%-20s %12s %12s %12s %s\n",
        "------", "------", "------", "------", "------");

    auto check = [&](const char* name, float hw, float gold) {
        float err = fabsf(hw - gold);
        bool pass = (err < TOL) || (gold != 0 && fabsf(err / gold) < TOL);
        printf("\n%-20s %12f %12f %12f %s\n",
            name, hw,  gold,  err,  pass? "PASS" : "FAIL");
        if (!pass) failures++;
    };

    check("mse_loss", hw_mse, golden_mse);
    check("mse_crossed", hw_xloss, golden_xloss);
    check("latent_dist", hw_ldist, golden_ldist);

    // summary

    printf("\n ==== %s ====\n", failures == 0? "ALL TESTS PASSED" : "FAILURES DETECTED");

    return failures;

}