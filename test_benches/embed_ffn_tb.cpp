#include <iostream>
#include <cmath>
#include <string>
#include "../embed_ffn/embed_ffn.h"
#include "tb_helpers.h"

int main() {
    std::string dir = "/home/snehadri/repos/unsupervised-search/phase3_export/";
    std::string wt = dir + "weights/";
    std::string tv = dir + "test_vectors/";
    int event_idx = 0;

    // load weights

    EmbedWeights embed_wts;
    load_dnn_block_weights<EMBED_IN, EMBED_HIDDEN, EMBED_OUT, EMBED_N_MID>(
        wt, "embed_net_", embed_wts);

    std::cout << "weights loaded." << std::endl;

    // load inputs

    data_t jets[N_MAX][EMBED_IN];
    load_2d<data_t, N_MAX, EMBED_IN>(tv + "stage0_input_raw.npy", jets, event_idx);


    bool mask[N_MAX];
    load_padding_mask(tv + "stage0_padding_mask.npy", mask, event_idx);

    data_t golden[N_MAX][EMBED_OUT];
    load_2d<data_t, N_MAX, EMBED_OUT>(tv + "stage1_post_embedding.npy", golden, event_idx);

    // run kernel

    data_t embed[N_MAX][E_DIM];
    embed_ffn(jets, mask, embed_wts, embed);

    bool pass = compare<N_MAX>("embed_ffn", embed, golden, mask);

    return pass ? 0 : 1;

}
