// Stepped hybrid golden: bit-accurate prediction of the aie_stream hardware
// output. Runs the PL stages natively (same headers the hardware PL was built
// from, -DAIE_FRAC11) and hands each attention block to x86sim via PLIO text
// files, mirroring the bridge's bit-reinterpretation and packing exactly.
//
// Phases (driven by an outer script that runs x86simulator between them):
//   p1 <input.bin> <ev>   : read+fork, embed, pairwise -> obj0 x + 4 wij files
//   p2                    : obj0 out -> remask -> cand_build -> cand0 c file
//   p3                    : cand0 out + saved x -> cross0 x/c files
//   p4                    : cross0 out -> obj1 x file
//   p5                    : obj1 out -> remask -> cand_build -> cand1 c file
//   p6                    : cand1 out + saved x -> cross1 x/c files
//   p7                    : cross1 out + saved raw/mask/c1 -> lorentz+AE
//                           -> prints the exact 3 output words (float + hex)
// State between phases lives in ./golden_state/*.bin (raw bits).
//
// Build: g++ -O2 -std=c++14 -DAIE_FRAC11 -I$VITIS/include hybrid_golden_steps.cpp

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <cmath>
namespace hls {
    float  rsqrt(float x)  { return 1.0f / std::sqrt(x); }
    double rsqrt(double x) { return 1.0  / std::sqrt(x); }
}
#include "../../attn_block_pl/attn_block_types.h"
#include "../../embed_ffn/embed_ffn.h"
#include "../../pairwise_mlp/pairwise_mlp.h"
#include "../../cand_lorentz/cand_lorentz.h"
#include "../../autoencoder/autoencoder.h"
#include "../../cand_build/candidate_build.h"
// RETRAINED weights (repo weights_rom.h): matches the retrained AIE weight
// headers sliced 2026-08-10. Default types (<16,7> data / <16,11> score).
#include "../../pl_stream/weights_rom.h"

static const char* ST = "golden_state";

// ---- PLIO text io: 4 int16 per line (one 64-bit beat) ----
static void write_plio(const char* path, const int16_t* v, int n) {
    FILE* f = fopen(path, "w");
    for (int i = 0; i < n; i += 4) {
        for (int j = 0; j < 4; j++)
            fprintf(f, "%d%s", (i + j < n) ? v[i + j] : 0, j < 3 ? " " : "");
        fprintf(f, "\n");
    }
    fclose(f);
}
static int read_plio(const char* path, int16_t* v, int n) {
    std::ifstream f(path);
    int i = 0; long x;
    while (i < n && (f >> x)) v[i++] = (int16_t)x;
    return i;
}
static void save_bin(const char* name, const void* p, size_t n) {
    char b[256]; snprintf(b, 256, "%s/%s", ST, name);
    FILE* f = fopen(b, "wb"); fwrite(p, 1, n, f); fclose(f);
}
static void load_bin(const char* name, void* p, size_t n) {
    char b[256]; snprintf(b, 256, "%s/%s", ST, name);
    FILE* f = fopen(b, "rb");
    if (!f || fread(p, 1, n, f) != n) { fprintf(stderr, "state %s missing\n", name); exit(2); }
    fclose(f);
}

// bit converters (mirror the bridge exactly: raw range copies)
static int16_t d2i(data_t v)  { ap_uint<16> b = v.range(15, 0); return (int16_t)(uint16_t)b; }
static data_t  i2d(int16_t x) { data_t v; v.range(15, 0) = (uint16_t)x; return v; }

static void x_to_file(const data_t x[N_MAX][E_DIM], const char* path) {
    int16_t b[N_MAX * E_DIM];
    for (int i = 0; i < N_MAX; i++) for (int j = 0; j < E_DIM; j++) b[i * E_DIM + j] = d2i(x[i][j]);
    write_plio(path, b, N_MAX * E_DIM);
}
static void file_to_x(const char* path, data_t x[N_MAX][E_DIM]) {
    int16_t b[N_MAX * E_DIM];
    if (read_plio(path, b, N_MAX * E_DIM) != N_MAX * E_DIM) { fprintf(stderr, "short read %s\n", path); exit(2); }
    for (int i = 0; i < N_MAX; i++) for (int j = 0; j < E_DIM; j++) x[i][j] = i2d(b[i * E_DIM + j]);
}
static void c_to_file(const data_t c[T_DIM][E_DIM], const char* path) {
    int16_t b[T_DIM * E_DIM];
    for (int i = 0; i < T_DIM; i++) for (int j = 0; j < E_DIM; j++) b[i * E_DIM + j] = d2i(c[i][j]);
    write_plio(path, b, T_DIM * E_DIM);
}
static void file_to_c(const char* path, data_t c[T_DIM][E_DIM]) {
    int16_t b[T_DIM * E_DIM];
    if (read_plio(path, b, T_DIM * E_DIM) != T_DIM * E_DIM) { fprintf(stderr, "short read %s\n", path); exit(2); }
    for (int i = 0; i < T_DIM; i++) for (int j = 0; j < E_DIM; j++) c[i][j] = i2d(b[i * E_DIM + j]);
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s p1..p7 [input.bin ev]\n", argv[0]); return 1; }
    std::string ph = argv[1];

    static EmbedWeights embed_w; static MLPWeights mlp_w;
    static AEEncoderWeights ae_enc_w; static AEDecoderWeights ae_dec_w;
    init_pl_only_weights(embed_w, mlp_w, ae_enc_w, ae_dec_w);

    if (ph == "p1") {
        const char* fin = argv[2]; int ev = atoi(argv[3]);
        uint32_t words[72];
        std::ifstream f(fin, std::ios::binary);
        f.seekg((long)ev * 72 * 4); f.read((char*)words, 72 * 4);
        // read_and_fork bit pattern: low 16 bits -> data_t; mask word != 0
        data_t raw[N_MAX][RAW_DIM]; bool mask[N_MAX];
        for (int i = 0; i < N_MAX; i++)
            for (int j = 0; j < RAW_DIM; j++) raw[i][j] = i2d((int16_t)(words[i * RAW_DIM + j] & 0xFFFF));
        for (int i = 0; i < N_MAX; i++) mask[i] = (words[60 + i] != 0);
        save_bin("raw.bin", raw, sizeof(raw)); save_bin("mask.bin", mask, sizeof(mask));

        data_t x[N_MAX][E_DIM];
        embed_ffn(raw, mask, embed_w, x);
        x_to_file(x, "data/obj_x_in_L0.txt");

        // pairwise -> wij; per-head slice mirrors obj_attn_send (col N_MAX = 0)
        data_t w_ang[N_MAX][3];
        for (int j = 0; j < N_MAX; j++) { w_ang[j][0] = raw[j][1]; w_ang[j][1] = raw[j][2]; w_ang[j][2] = raw[j][3]; }
        data_t wij[N_MAX][N_MAX];
        pairwise_mlp(w_ang, mlp_w, wij);
        int16_t slice[N_MAX * N_KV];
        for (int i = 0; i < N_MAX; i++)
            for (int j = 0; j < N_KV; j++) {
                // stream carries (score_t)wij bits (Q10.5); the AIE obj
                // kernels expect Q8.7 -> shift left 2 (mirrors the bridge)
                score_t sc = (score_t)wij[i][j];
                ap_int<16> bits = sc.range(15, 0);
                ap_int<16> shifted = (ap_int<16>)(bits << 2);
                slice[i * N_KV + j] = (j < N_MAX) ? (int16_t)(short)shifted : 0;
            }
        for (int h = 0; h < 4; h++) {
            char p[64]; snprintf(p, 64, "data/obj_wij_h%d_L0.txt", h);
            write_plio(p, slice, N_MAX * N_KV);
        }
        printf("p1 done (ev %d)\n", ev);
    } else if (ph == "p2" || ph == "p5") {
        bool L0 = (ph == "p2");
        data_t x[N_MAX][E_DIM]; bool mask[N_MAX];
        file_to_x(L0 ? "x86simulator_output/data/obj_x_out_L0.txt"
                     : "x86simulator_output/data/obj_x_out_L1.txt", x);
        load_bin("mask.bin", mask, sizeof(mask));
        remask(x, mask);
        data_t c[T_DIM][E_DIM]; int tmp[N_MAX];
        build_candidates<N_MAX>(x, c, tmp);
        save_bin(L0 ? "x0_for_cross.bin" : "x1_for_cross.bin", x, sizeof(x));
        c_to_file(c, L0 ? "data/cand_c_in_L0.txt" : "data/cand_c_in_L1.txt");
        printf("%s done\n", ph.c_str());
    } else if (ph == "p3" || ph == "p6") {
        bool L0 = (ph == "p3");
        data_t x[N_MAX][E_DIM]; data_t c[T_DIM][E_DIM];
        load_bin(L0 ? "x0_for_cross.bin" : "x1_for_cross.bin", x, sizeof(x));
        file_to_c(L0 ? "x86simulator_output/data/cand_c_out_L0.txt"
                     : "x86simulator_output/data/cand_c_out_L1.txt", c);
        if (!L0) save_bin("c1_after_cand.bin", c, sizeof(c));
        x_to_file(x, L0 ? "data/cross_x_in_L0.txt" : "data/cross_x_in_L1.txt");
        c_to_file(c, L0 ? "data/cross_c_in_L0.txt" : "data/cross_c_in_L1.txt");
        printf("%s done\n", ph.c_str());
    } else if (ph == "p4") {
        data_t x[N_MAX][E_DIM];
        file_to_x("x86simulator_output/data/cross_x_out_L0.txt", x);
        x_to_file(x, "data/obj_x_in_L1.txt");
        printf("p4 done\n");
    } else if (ph == "p7") {
        data_t x[N_MAX][E_DIM]; data_t c[T_DIM][E_DIM];
        data_t raw[N_MAX][RAW_DIM]; bool mask[N_MAX];
        file_to_x("x86simulator_output/data/cross_x_out_L1.txt", x);
        load_bin("c1_after_cand.bin", c, sizeof(c));
        load_bin("raw.bin", raw, sizeof(raw)); load_bin("mask.bin", mask, sizeof(mask));

        float jp4[N_MAX][P4_DIM]; int ja[N_MAX];
        float cp4[T_DIM][P4_DIM]; float cms[T_DIM];
        data_t ae_in[T_DIM][AE_IN_DIM];
        cand_lorentz(raw, x, c, mask, jp4, ja, cp4, cms, ae_in);

        data_t c0[1][AE_IN_DIM], c1[1][AE_IN_DIM];
        for (int i = 0; i < AE_IN_DIM; i++) { c0[0][i] = ae_in[0][i]; c1[0][i] = ae_in[1][i]; }
        data_t l0[1][AE_DIM], l1[1][AE_DIM], d0[1][AE_IN_DIM], d1[1][AE_IN_DIM];
        float mse, mse_x, ld;
        dual_autoencoder(c0, c1, ae_enc_w, ae_dec_w, l0, l1, d0, d1, mse, mse_x, ld);

        uint32_t h[3]; memcpy(&h[0], &mse, 4); memcpy(&h[1], &mse_x, 4); memcpy(&h[2], &ld, 4);
        printf("PREDICTED: %.5f %.5f %.5f  [hex %08x %08x %08x]\n", mse, mse_x, ld, h[0], h[1], h[2]);
    } else { fprintf(stderr, "unknown phase\n"); return 1; }
    return 0;
}
