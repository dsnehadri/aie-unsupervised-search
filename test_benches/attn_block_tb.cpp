// attention block testbench
// loads exported weights and test vectors, runs hls attention blocks and compares against pytorch reference

#include "tb_helpers.h"
#include <cstdio>
#include <cstdlib>

// usage: /home/snehadri/Vitis_HLS/2022.2/bin/vitis_hls -f run_csim.tcl

// forward declaration of the dut

#include "../attn_block_pl/attn_block_obj.h"
#include "../attn_block_pl/attn_block_cand.h"
#include "../attn_block_pl/attn_block_cross.h"

int main() {
    std::string dir = "/home/snehadri/repos/unsupervised-search/phase3_export/";
    std::string weights_suffix = "weights/";
    std::string tests_suffix = "test_vectors/";

    int event_idx = 0;
    int failures = 0;

    {
        std::string block = "obj_blocks_0";
        printf("test 1: OBJ\n");
        AttnWeights w;
        load_attn_weights(block, w);

        data_t x[N_MAX][E_DIM];
        data_t golden[N_MAX][E_DIM];

        load_2d<data_t, N_MAX, E_DIM>(dir + tests_suffix + "stage1_post_embedding.npy", x, event_idx);
        load_2d<data_t, N_MAX, E_DIM>(dir + tests_suffix + "stage3_layer0_post_obj_selfattn.npy", golden, event_idx);

        bool padding_mask[N_MAX];
        load_padding_mask(dir + tests_suffix + "stage0_padding_mask.npy", padding_mask, event_idx);

        // load wij bias

        score_t wij_bias[N_HEADS * N_MAX][N_KV] = {0};
        bool use_wij = true;

        {
            // load one events Wij from post mlp output
            data_t wij_single[N_MAX][N_MAX];
            load_2d<data_t, N_MAX, N_MAX>(dir + tests_suffix + "stage2_wij_post_mlp.npy", wij_single, event_idx);

            // replicate accross heads, column N_MAX (bias_kv) stays zero

            for (int h = 0; h <N_HEADS; h++) {
                for (int i = 0; i < N_MAX; i++) {
                    for (int j = 0; j<N_MAX; j++) {
                        wij_bias[h * N_MAX + i][j] = (score_t)wij_single[i][j];
                    }
                }
            }
        }

        printf("running obj_blocks[0] on event%d...\n", event_idx);
        
        attn_block_obj(
            x, 
            padding_mask,
            wij_bias, use_wij,
            w.Wq, w.bq, w.Wk, w.bk, w.Wv, w.bv,
            w.bias_k, w.bias_v, w.Wo, w.bo,
            w.attn_ln_g, w.attn_ln_b,
            w.ffn_w, w.ffn_b, w.ffn_ln_g, w.ffn_ln_b,
            w.post_ffn_g, w.post_ffn_b);    

        if (!compare<N_MAX>("obj_blocks", x, golden, padding_mask)) failures++;
    }

    {
        std::string block = "cand_blocks_0";
        printf("test 2: CAND\n");
        AttnWeights w;
        load_attn_weights(block, w);

        data_t c[T_DIM][E_DIM];
        data_t golden[T_DIM][E_DIM];

        load_2d<data_t, T_DIM, E_DIM>(dir + tests_suffix + "stage3_layer0_candidates_embedded.npy", c, event_idx);
        load_2d<data_t, T_DIM, E_DIM>(dir + tests_suffix + "stage3_layer0_post_cand_selfattn.npy", golden, event_idx);


        attn_block_cand(c, 
            w.Wq, w.bq, w.Wk, w.bk, w.Wv, w.bv,
            w.bias_k, w.bias_v, w.Wo, w.bo,
            w.attn_ln_g, w.attn_ln_b,
            w.ffn_w, w.ffn_b, w.ffn_ln_g, w.ffn_ln_b,
            w.post_ffn_g, w.post_ffn_b);    

        if (!compare<T_DIM>("cand_blocks", c, golden)) failures++;
    }

    {
        std::string block = "cross_blocks_0";
        printf("test 3: CROSS\n");
        AttnWeights w;
        load_attn_weights(block, w);

        data_t x[N_MAX][E_DIM];
        data_t c[T_DIM][E_DIM];
        data_t golden[N_MAX][E_DIM];

        load_2d<data_t, N_MAX, E_DIM>(dir + tests_suffix + "stage3_layer0_post_obj_selfattn.npy", x, event_idx);
        load_2d<data_t, T_DIM, E_DIM>(dir + tests_suffix + "stage3_layer0_post_cand_selfattn.npy", c, event_idx);
        load_2d<data_t, N_MAX, E_DIM>(dir + tests_suffix + "stage3_layer0_post_cross_attn.npy", golden, event_idx);

        bool padding_mask[N_MAX];
        load_padding_mask(dir + tests_suffix + "stage0_padding_mask.npy", padding_mask, event_idx);

        printf("running cross_blocks[0] on event%d...\n", event_idx);
        
        attn_block_cross(x, c, 
            w.Wq, w.bq, w.Wk, w.bk, w.Wv, w.bv,
            w.bias_k, w.bias_v, w.Wo, w.bo,
            w.attn_ln_g, w.attn_ln_b,
            w.ffn_w, w.ffn_b, w.ffn_ln_g, w.ffn_ln_b,
            w.post_ffn_g, w.post_ffn_b);    

        // remask padded positions
        for (int i = 0; i < N_MAX; i++) {
            if (padding_mask[i]) {
                for (int j = 0; j < E_DIM; j++) {
                    x[i][j] = 0;
                }
            }
        }

        if (!compare<N_MAX>("cross_blocks", x, golden)) failures++;

        for (int i = 0; i < N_MAX; i++) {
            float row_max = 0;
            for (int j = 0; j < E_DIM; j++) {
                float err = fabsf((float)x[i][j] - (float)golden[i][j]);
                if (err > row_max) row_max = err;
            }
        }
    }

}
