// FLOAT_DATAPATH per-stage check: run embed + obj0 block (the shared
// attention algorithm) in float and dump outputs for comparison vs PyTorch.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <cmath>
namespace hls {
    float  rsqrt(float x)  { return 1.0f / std::sqrt(x); }
    double rsqrt(double x) { return 1.0  / std::sqrt(x); }
}
#include "../attn_block_pl/attn_block_types.h"
#include "../embed_ffn/embed_ffn.h"
#include "../pairwise_mlp/pairwise_mlp.h"
#include "../cand_lorentz/cand_lorentz.h"  // RAW_DIM
#include "../attn_block_pl/attn_block_obj.h"
#include "weights_rom.h"

int main(int argc, char** argv) {
    int nev = argc > 1 ? atoi(argv[1]) : 20;
    static EmbedWeights embed_w; static MLPWeights mlp_w;
    static AttnWeights obj0_w, cand0_w, cross0_w, obj1_w, cand1_w, cross1_w;
    static AEEncoderWeights ae_enc_w; static AEDecoderWeights ae_dec_w;
    init_all_weights(embed_w, mlp_w, obj0_w, cand0_w, cross0_w,
                     obj1_w, cand1_w, cross1_w, ae_enc_w, ae_dec_w);

    std::ifstream fin("evalfloat_bkg.bin", std::ios::binary);
    FILE* f1 = fopen("cpp_stage1.bin", "wb");   // embed out
    FILE* f3 = fopen("cpp_stage3.bin", "wb");   // obj block out

    for (int ev = 0; ev < nev; ev++) {
        uint32_t words[72];
        fin.read((char*)words, 72 * 4);
        data_t raw[N_MAX][RAW_DIM]; bool mask[N_MAX];
        for (int i = 0; i < N_MAX; i++)
            for (int j = 0; j < RAW_DIM; j++)
                memcpy(&raw[i][j], &words[i * RAW_DIM + j], 4);
        for (int i = 0; i < N_MAX; i++) mask[i] = (words[60 + i] != 0);

        data_t x[N_MAX][E_DIM];
        embed_ffn(raw, mask, embed_w, x);
        fwrite(x, sizeof(float), N_MAX * E_DIM, f1);

        data_t w_ang[N_MAX][3];
        for (int j = 0; j < N_MAX; j++) { w_ang[j][0] = raw[j][1]; w_ang[j][1] = raw[j][2]; w_ang[j][2] = raw[j][3]; }
        data_t wij[N_MAX][N_MAX];
        pairwise_mlp(w_ang, mlp_w, wij);
        score_t wij_bias[N_MAX * N_HEADS][N_KV];
        expand_wij(wij, wij_bias);

        attn_block_obj(x, mask, wij_bias, true,
            obj0_w.Wq, obj0_w.bq, obj0_w.Wk, obj0_w.bk, obj0_w.Wv, obj0_w.bv,
            obj0_w.bias_k, obj0_w.bias_v, obj0_w.Wo, obj0_w.bo,
            obj0_w.attn_ln_g, obj0_w.attn_ln_b,
            obj0_w.ffn_w, obj0_w.ffn_b, obj0_w.ffn_ln_g, obj0_w.ffn_ln_b,
            obj0_w.post_ffn_g, obj0_w.post_ffn_b);
        remask(x, mask);
        fwrite(x, sizeof(float), N_MAX * E_DIM, f3);
    }
    fclose(f1); fclose(f3);
    printf("dumped %d events\n", nev);
    return 0;
}
