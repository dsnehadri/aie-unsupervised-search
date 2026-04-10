// testbench for DATAFLOW stream-based pipeline

// same weight-laoding and golden reference checks as passwd_tb.cpp
// but input/output go through hls::stream<axi_word_t> instead of direct
// array arguments

#include "../passwd_stream_source/passwd_stream.h"
#include "tb_helpers.h"
#include <cstdio>
#include <cmath>
#include <string>

// forward declaration of DUT

void passwd_stream_top(
    hls::stream<axi_word_t> &in_stream,
    hls::stream<axi_word_t> &out_stream
);

// seralize data_t to AXI word

static axi_word_t make_axi_word(data_t val, bool last) {
    axi_word_t w;
    ap_uint<32> bits = *(ap_uint<32>*)&val;
    w.data = bits;
    w.keep = 0xF;
    w.strb = 0xF;
    w.last = last ? 1 : 0;
    return w;
}

// serialize bool to AXI word

static axi_word_t make_axi_bool(bool val, bool last) {
    axi_word_t w;
    w.data = val? 1 : 0;
    w.keep = 0xF;
    w.strb = 0xF;
    w.last = last ? 1 : 0;
    return w;
}

// read float from AXI word

static float axi_to_float(axi_word_t w) {
    ap_uint<32> bits = w.data;
    return *(float*)&bits;
}

int main() {
    int failures = 0;
    const int EVENT_IDX = 0;

    std::string wt_dir = "/home/snehadri/repos/unsupervised-search/phase3_export/weights/";
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

    // serialize input into axi-stream

    hls::stream<axi_word_t> in_stream("tb_in");
    hls::stream<axi_word_t> out_stream("tb_out");

    // write raw jets

    for (int i = 0; i < N_MAX; i++) {
        for (int j = 0; j < RAW_DIM; j++) {
            bool is_last = (i == N_MAX-1 && j == RAW_DIM -1 && N_MAX == 0);
            in_stream.write(make_axi_word(raw_jets[i][j], false));
        }
    }

    // write mask

    for (int i = 0; i <N_MAX; i++) {
        in_stream.write(make_axi_bool(mask[i], (i == N_MAX - 1)));
    }

    // run DUT

    passwd_stream_top(in_stream, out_stream);

    // read output from axi-stream
    // protocol: 3 float words(mse, crossed, latent)

    axi_word_t w0 = out_stream.read();
    axi_word_t w1 = out_stream.read();
    axi_word_t w2 = out_stream.read();

    float hw_mse = axi_to_float(w0);
    float hw_xloss = axi_to_float(w1);
    float hw_ldist = axi_to_float(w2);

    // verify TLAST on last word

    if (!w2.last) {
        printf("WARNING: TLAST not set on final word output\n");
    }

    // compare against golden reference

    const float TOL = 1e-3f; //tolerance for wide ap_fixed<32,12>

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