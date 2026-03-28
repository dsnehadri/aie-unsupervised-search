#include <iostream>
#include <cmath>
#include <string>
#include "embed_ffn.h"
#include "../test_benches/tb_helpers.h"

int main() {
    std::string dir = "/home/snehadri/repos/unsupervised-search/phase3_export/";
    std::string wt = dir + "weights/";
    std::string tv = dir + "test_vectors/";
    int event_idx = 0;

    // 1. load weights

    // 0 = linear (5->16)
    // 1 = layernorm(16)
    // 2 = relu
    // 3 = linear (16->16)
    // 4 = layernorm
    // 5 = relu
    // 6 = linear

    weight_t lin0_w[E_DIM][IN_DIM];
    weight_t lin0_b[E_DIM];
    ln_param_t ln0_g[E_DIM], ln0_b[E_DIM];

    weight_t lin1_w[E_DIM][E_DIM];
    weight_t lin1_b[E_DIM];
    ln_param_t ln1_g[E_DIM], ln1_b[E_DIM];

    weight_t lin2_w[E_DIM][E_DIM];
    weight_t lin2_b[E_DIM];

    load_2d<weight_t, E_DIM, IN_DIM>(wt + "embed_net_0_weight.npy", lin0_w);
    load_1d<weight_t, E_DIM>(wt + "embed_net_0_bias.npy", lin0_b);
    load_1d<ln_param_t, E_DIM>(wt + "embed_net_1_weight.npy", ln0_g);
    load_1d<ln_param_t, E_DIM>(wt + "embed_net_1_bias.npy", ln0_b);

    load_2d<weight_t, E_DIM, E_DIM>(wt + "embed_net_3_weight.npy", lin1_w);
    load_1d<weight_t, E_DIM>(wt + "embed_net_3_bias.npy", lin1_b);
    load_1d<ln_param_t, E_DIM>(wt + "embed_net_4_weight.npy", ln1_g);
    load_1d<ln_param_t, E_DIM>(wt + "embed_net_4_bias.npy", ln1_b);

    load_2d<weight_t, E_DIM, E_DIM>(wt + "embed_net_6_weight.npy", lin2_w);
    load_1d<weight_t, E_DIM>(wt + "embed_net_6_bias.npy", lin2_b);

    printf("weights loaded \n");

    // load test vectors

    data_t jets[N_MAX][IN_DIM];
    load_2d<data_t, N_MAX, IN_DIM>(tv + "stage0_input_raw.npy", jets, event_idx);

    bool mask[N_MAX];
    load_padding_mask(tv + "stage0_padding_mask.npy", mask, event_idx);

    data_t golden[N_MAX][E_DIM];
    load_2d<data_t, N_MAX, E_DIM>(tv + "stage1_post_embedding.npy", golden, event_idx);

    // run kernel

    data_t embed[N_MAX][E_DIM];
    embed_ffn(jets, mask,
    lin0_w, lin0_b, ln0_g, ln0_b,
    lin1_w, lin1_b, ln1_g, ln1_b,
    lin2_w, lin2_b, embed);

    bool pass = compare<N_MAX>("embed_ffn", embed, golden, mask);

    return pass ? 0 : 1;

}
