// attention block testbench
// loads exported weights and test vectors, runs hls attention blocks and compares against pytorch reference

#include "tb_helpers.h"
#include <cstdio>
#include <cstdlib>

// usage: /home/snehadri/Vitis_HLS/2022.2/bin/vitis_hls -f run_csim.tcl

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

extern void attn_block_cand(
    data_t c[T_DIM][E_DIM], // input embeddings, which are modified in place

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


extern void attn_block_cross(
    data_t x[N_MAX][E_DIM],
    const data_t c[T_DIM][E_DIM], // input embeddings, which are modified in place

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

int main() {
    std::string dir = "/home/snehadri/repos/unsupervised-search/phase3_export/";
    std::string weights_suffix = "weights/";
    std::string tests_suffix = "test_vectors/";

    int event_idx = 0;
    int failures = 0;

    {
        std::string block = "obj_blocks_0";
        printf("test 1: OBJ");
        attn_weights w;
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


        // Check a few input values
        printf("Input x[0][0..3]: %f %f %f %f\n",
            (float)x[0][0], (float)x[0][1], (float)x[0][2], (float)x[0][3]);

        // Check padding mask
        printf("Padding mask: ");
        for (int i = 0; i < N_MAX; i++) printf("%d ", padding_mask[i]);
        printf("\n");

        // Check a few weight values
        printf("Wq[0][0..3]: %f %f %f %f\n",
            (float)w.Wq[0][0], (float)w.Wq[0][1], (float)w.Wq[0][2], (float)w.Wq[0][3]);

        // Check golden output
        printf("Golden[0][0..3]: %f %f %f %f\n",
            (float)golden[0][0], (float)golden[0][1], (float)golden[0][2], (float)golden[0][3]);

        // Check Wij
        printf("Wij[0][0..3]: %f %f %f %f\n",
            (float)wij_bias[0][0], (float)wij_bias[0][1], (float)wij_bias[0][2], (float)wij_bias[0][3]);

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
        printf("test 2: CAND");
        attn_weights w;
        load_attn_weights(block, w);

        data_t c[T_DIM][E_DIM];
        data_t golden[T_DIM][E_DIM];

        load_2d<data_t, T_DIM, E_DIM>(dir + tests_suffix + "stage3_layer0_candidates_embedded.npy", c, event_idx);
        load_2d<data_t, T_DIM, E_DIM>(dir + tests_suffix + "stage3_layer0_post_cand_selfattn.npy", golden, event_idx);


        // Check a few input values
        printf("Input c[0][0..3]: %f %f %f %f\n",
            (float)c[0][0], (float)c[0][1], (float)c[0][2], (float)c[0][3]);

        // Check golden output
        printf("Golden[0][0..3]: %f %f %f %f\n",
            (float)golden[0][0], (float)golden[0][1], (float)golden[0][2], (float)golden[0][3]);

        printf("running cand_blocks[0] on event%d...\n", event_idx);
        
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
        printf("test 2: CROSS");
        attn_weights w;
        load_attn_weights(block, w);

        data_t x[N_MAX][E_DIM];
        data_t c[T_DIM][E_DIM];
        data_t golden[N_MAX][E_DIM];

        load_2d<data_t, N_MAX, E_DIM>(dir + tests_suffix + "stage3_layer0_post_obj_selfattn.npy", x, event_idx);
        load_2d<data_t, T_DIM, E_DIM>(dir + tests_suffix + "stage3_layer0_post_cand_selfattn.npy", c, event_idx);
        load_2d<data_t, N_MAX, E_DIM>(dir + tests_suffix + "stage3_layer0_post_cross_attn.npy", golden, event_idx);

        bool padding_mask[N_MAX];
        load_padding_mask(dir + tests_suffix + "stage0_padding_mask.npy", padding_mask, event_idx);

        // Check a few input values
        printf("Input x[0][0..3]: %f %f %f %f\n",
            (float)x[0][0], (float)x[0][1], (float)x[0][2], (float)x[0][3]);

        // Check a few input values
        printf("Input c[0][0..3]: %f %f %f %f\n",
            (float)c[0][0], (float)c[0][1], (float)c[0][2], (float)c[0][3]);

        // Check golden output
        printf("Golden[0][0..3]: %f %f %f %f\n",
            (float)golden[0][0], (float)golden[0][1], (float)golden[0][2], (float)golden[0][3]);

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
            printf("row %d max_err=%.6f  padded=%d\n", i, row_max, padding_mask[i]);
        }
    }

}
