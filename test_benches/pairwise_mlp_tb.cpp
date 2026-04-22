#include "../pairwise_mlp/pairwise_mlp.h"
#include "tb_helpers.h"


int main() {
    std::string dir = "/home/snehadri/repos/unsupervised-search/phase3_export/";
    std::string wt = dir + "weights/";
    std::string tv = dir + "test_vectors/";
    int event_idx = 0;

    // load weights

    MLPWeights mlp_wts;
    load_dnn_block_weights<MLP_IN, MLP_HIDDEN, MLP_OUT, MLP_N_MID>(
        wt, "mlp_net_", mlp_wts);

    std::cout << "weights loaded." << std::endl;

    // extract w = (eta, cos_phi, sin_phi) from raw jets

    cnpy::NpyArray jets_npy = cnpy::npy_load(tv + "stage0_input_raw.npy");
    float *jets_data = jets_npy.data<float>();

    data_t w[N_MAX][3];
    for (int j = 0; j < N_MAX; j++) {
        int base = event_idx * N_MAX * 5 + j * 5;
        w[j][0] = (data_t)jets_data[base + 1]; // eta
        w[j][1] = (data_t)jets_data[base + 2]; // eta
        w[j][2] = (data_t)jets_data[base + 3]; // eta

    }

    printf("w[0]: %f %f %f\n", (float)w[0][0], (float)w[0][1], (float)w[0][2]);
    printf("w[6]: %f %f %f\n", (float)w[6][0], (float)w[6][1], (float)w[6][2]);
    fflush(stdout);

    data_t golden_wij[N_MAX][N_MAX];
    load_2d<data_t, N_MAX, N_MAX>(tv + "stage2_wij_post_mlp.npy", golden_wij, event_idx);

    // run kernel

    data_t wij[N_MAX][N_MAX];
    pairwise_mlp(w, mlp_wts, wij);

    bool pass = compare<N_MAX, N_MAX>("pairwise_mlp", wij, golden_wij);

    return pass ? 0 : 1;

}
