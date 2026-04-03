#include <iostream>
#include <cmath>
#include <string>
#include "../cand_lorentz_source/cand_lorentz.h"
#include "../test_benches/tb_helpers.h"

int main() {
    std::string dir = "/home/snehadri/repos/unsupervised-search/phase3_export/";
    std::string tv = dir + "test_vectors/";

    int test_events = 3;
    int failures = 0;
    bool all_pass = true;

    for (int ev = 0; ev < test_events; ev++) {
        printf("event %d \n", ev);

        data_t raw_x[N_MAX][RAW_DIM];
        load_2d<data_t, N_MAX, RAW_DIM>(tv + "stage0_input_raw.npy", raw_x, ev);

        bool mask[N_MAX];
        load_padding_mask(tv + "stage0_padding_mask.npy", mask, ev);

        data_t attn_x[N_MAX][E_DIM];
        load_2d<data_t, N_MAX, E_DIM>(tv + "stage3_layer1_post_cross_attn.npy", attn_x, ev);

        data_t cand_embed[T_DIM][E_DIM];
        load_2d<data_t, T_DIM, E_DIM>(tv + "stage3_layer1_post_cand_selfattn.npy", cand_embed, ev);

        // load golden references

        float golden_jp4[N_MAX][P4_DIM];
        load_2d<float, N_MAX, P4_DIM>(tv + "stage4_jet_p4.npy", golden_jp4, ev);

        float golden_jc[N_MAX][T_DIM];
        load_2d<float, N_MAX, T_DIM>(tv + "stage4_final_jet_choice.npy", golden_jc, ev);

        float golden_cand_p4[T_DIM][P4_DIM];
        load_2d<float, T_DIM, P4_DIM>(tv + "stage4_candidates_p4.npy", golden_cand_p4, ev);

        float golden_mass_2d[T_DIM][1];
        {
            cnpy::NpyArray arr = cnpy::npy_load(tv + "stage4_candidate_mass_scale.npy");
            float *d = arr.data<float>();
            for (int t = 0; t < T_DIM; t++) {
                golden_mass_2d[t][0] = d[ev * T_DIM + t];
            }
            
        }
        data_t golden_ae[T_DIM][AE_IN_DIM];
        load_2d<data_t, T_DIM, AE_IN_DIM>(tv + "stage4_ae_input_features.npy", golden_ae, ev);

        // zero out padded jets in attn_x

        for (int i = 0; i < N_MAX; i++) {
            if (mask[i]) {
                for (int e = 0; e < E_DIM; e ++) {
                    attn_x[i][e] = 0;
                }
            }
        }

        // // run kernel
        
        // float cand_p4[T_DIM][P4_DIM];
        // float cand_mass_scaled[T_DIM];
        // data_t ae_input[T_DIM][AE_IN_DIM];

        // cand_lorentz(raw_x, attn_x, cand_embed, mask, jp4, jet_assign, cand_p4, cand_mass_scaled, ae_input);

        // validate

        bool ev_pass = true;

        // jet assignments vs golden one-hot
        int jet_assign[N_MAX] = {};
        float jet_choice_float[N_MAX][T_DIM];
        get_jet_choice(attn_x, mask, jet_assign, jet_choice_float);

        int assign_mismatches = 0;
        for (int i = 0; i <N_MAX; i++) {
            if (mask[i]) continue;
            int golden_cat = -1;
            for (int t = 0; t < T_DIM; t++) {
                if (golden_jc[i][t]  > 0.5f) {
                    golden_cat = t;
                    break;
                }
            }
            if (golden_cat == -1) continue;
            if (jet_assign[i] != golden_cat) {
                printf(" jet note jet %d: HLS: %d golden=%d "
                "(scores: %.4f %.4f %.4f)\n",
                    i, jet_assign[i], golden_cat,
                    (float)attn_x[i][0], (float)attn_x[i][1], (float)attn_x[i][2]);
                assign_mismatches++;
            }
        }

        if (assign_mismatches > 0) {
            printf(" jet assignment mismatches: %d\n", assign_mismatches);
            ev_pass = false;
        }

        // per jet p4

        float jp4[N_MAX][P4_DIM];
        x_to_p4_hw(raw_x, mask, jp4);

        if (!compare<N_MAX, P4_DIM, float>("x_to_p4", jp4, golden_jp4, mask, 1.0f))
            ev_pass = false;

        // candidate p4

        float cand_p4[T_DIM][P4_DIM];
        build_candidates_p4(golden_jc, jp4, cand_p4);

        if (!compare<T_DIM, P4_DIM, float>("cand_p4", cand_p4, golden_cand_p4, nullptr, 5.0f))
            ev_pass = false;

        // mass

        float cand_mass_scaled[T_DIM];
        compute_mass(cand_p4, cand_mass_scaled);

        float mass_2d[T_DIM][1];
        for (int t = 0; t < T_DIM; t++) {
            mass_2d[t][0] =  cand_mass_scaled[t];
        }
        if(!compare<T_DIM, 1, float>("mass_scaled", mass_2d, golden_mass_2d, nullptr, 0.1f))
            ev_pass = false;

        // ae input features

        data_t ae_input[T_DIM][AE_IN_DIM];
        assemble_ae_input(cand_embed, cand_mass_scaled, ae_input);

        if(!compare<T_DIM, AE_IN_DIM, data_t>("ae_input", ae_input, golden_ae, nullptr, 0.1f))
            ev_pass = false;

        if (ev_pass) {
            printf("event %d PASS \n", ev);
        } else {
            printf(" Event %d: FAIL\n", ev);
            all_pass = false;
            failures++;
        }
    }

    printf("tested %d events, failrues %d\n", test_events, failures);
    printf("%s\n", all_pass? "ALL PASS" : "FAIL");
    return all_pass ? 0 : 1;
}