#include "tb_helpers.h"
#include "attn_helpers.h"
#include "candidate_build.h"

int main() {
    std::string dir = "/home/snehadri/repos/unsupervised-search/phase3_export/";
    std::string tv = dir + "test_vectors/";
    int event_idx = 0;
    int failures = 0;

    bool padding_mask[N_MAX];
    load_padding_mask(tv + "stage0_padding_mask.npy", padding_mask, event_idx);

    // test candidate building

    {
        printf("candidate building test");

        data_t x[N_MAX][E_DIM];
        load_2d<data_t, N_MAX, E_DIM>(tv + "stage3_layer0_post_obj_selfattn.npy", x, event_idx);

        data_t golden_c[T_DIM][E_DIM];
        load_2d<data_t, T_DIM, E_DIM>(tv + "stage3_layer0_candidates_embedded.npy", golden_c, event_idx);

        data_t golden_jc[N_MAX][T_DIM];
        load_2d<data_t, N_MAX, T_DIM>(tv + "stage3_layer0_jet_choice.npy", golden_jc, event_idx);

        // zero out padded jets

        for (int i = 0; i < N_MAX; i++) {
            if (padding_mask[i])
                for (int j = 0; j <  E_DIM; j++) 
                    x[i][j] = 0;
        }

        printf("input x[0][0..3]: %f %f %f %f\n", (float)x[0][0], (float)x[0][1], (float)x[0][2], (float)x[0][3]);
        printf("input golden_c[0][0..3]: %f %f %f %f\n", (float)golden_c[0][0], (float)golden_c[0][1], (float)golden_c[0][2], (float)golden_c[0][3]);

        // print golden jet assignments

        for (int i = 0; i < N_MAX; i++) {
            if(padding_mask[i]) continue;
            int gt = 0;
            for (int t = 0; t < T_DIM; t++) {
                if ((float)golden_jc[i][t] > (float)golden_jc[i][gt]) gt = t;
            }
            printf("jet %d -> cat %d\n", i, gt);
        }

        printf("running candidate_build on event %d...\n", event_idx);

        data_t c[T_DIM][E_DIM];
        int jet_assessment[N_MAX];
        build_candidates<N_MAX>(x, c, jet_assessment);

        printf("HLS jet assignments: \n");
        for (int i = 0; i < N_MAX; i++) {
            if(padding_mask[i]) continue;
            printf(" jet %d -> cat %d\n", i, jet_assessment[i]);
        }

        int mismatches = 0;
        for (int i = 0; i< N_MAX; i++) {
            if(padding_mask[i]) continue;
            int gt = 0;
            for (int t = 1; t < T_DIM; t++) {
                if ((float)golden_jc[i][t] > (float)golden_jc[i][gt]) gt = t;
            }
            if (jet_assessment[i] != gt) {
                printf("mismatch jet %d: HLS = %d, golden = %d\n",i, jet_assessment[i], gt);
                mismatches++;
            }
        }

        printf("jet assignment mismatches: %d\n", mismatches);
        if (mismatches > 0) failures++;

        if(!compare<T_DIM>("candidate_build_0", c, golden_c)) failures++;
    }
    printf("=== %d failures ===", failures);
    return failures;
}