#ifndef PL_STREAM_H
#define PL_STREAM_H

// passwd abc stream-based DATAFLOW pipeline
// each pipeline stage is a wrapper function that
// 1) deserializes from input hls::stream(s) into local arrays
// 2) calls the existing array-based kernel (unchanged)
// 3) serializes results to output hls::stream(s)

// #pragma HLS DATAFLOW directive makes stages execute concurrently 
// while AE processes event N, attention processes N+1, embedding
// processes N+2. throughput = 1/slowest stage, not sum of all stages

#include "hls_stream.h"
#include "ap_axi_sdata.h"

#include "../embed_ffn/embed_ffn.h"
#include "../pairwise_mlp/pairwise_mlp.h"
#include "../cand_lorentz/cand_lorentz.h"
#include "../autoencoder/autoencoder.h"
#include "../cand_build/candidate_build.h"

#include "../attn_block_pl/attn_block_obj.h"
#include "../attn_block_pl/attn_block_cand.h"
#include "../attn_block_pl/attn_block_cross.h"

// serialization helpers

// write 2d array to a stream in row-major order

template<int ROWS, int COLS, typename T>
void array2d_to_stream(const T arr[ROWS][COLS], hls::stream<T> &out) {
    ROWS_LOOP: for (int i = 0; i < ROWS; i++) {
        COLS_LOOP: for (int j = 0; j < COLS; j++) {
            #pragma HLS PIPELINE II=1
            out.write(arr[i][j]);
        }
    }
}

// read from stream into a 2d array in row-major order

template<int ROWS, int COLS, typename T>
void stream_to_array2d(hls::stream<T> &in, T arr[ROWS][COLS]) {
    ROWS_LOOP: for (int i = 0; i < ROWS; i++) {
        COLS_LOOP: for (int j = 0; j < COLS; j++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_FLATTEN off
            arr[i][j] = in.read();
        }
    }
}

// write a 1d array to stream 

template<int LEN, typename T>
void array1d_to_stream(const T arr[LEN], hls::stream<T> &out) {
    for (int i = 0; i < LEN; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_FLATTEN off
        out.write(arr[i]);
    }
}

// read from stream into a 1d array

template<int LEN, typename T>
void stream_to_array1d(hls::stream<T> &in, T arr[LEN]) {
    for (int i = 0; i < LEN; i++) {
        #pragma HLS PIPELINE II=1
        arr[i] = in.read();
    }
}

// read_and_fork reads one event from axi-stream input and write copies to all 
// downstream consumers. this solve the multi-consumer problem: DATAFLOW requires
// each stream to have exactly one writer and one reader

// input protoco: 60 words (raw_jets[12][5], row major) + 12 words (mask)
// each word is a 32-bit data_t reinterpreted as ap_uint<32>

inline void read_and_fork(
    hls::stream<ap_uint<32>> &in_s,

    // raw jets go to 3 consumers (embed, pairwise, cand_lorentz)
    hls::stream<data_t> &out_jets_embed,
    hls::stream<data_t> &out_jets_pairwise,
    hls::stream<data_t> &out_jets_cand,

    // mask -> 6 consumers (embed, obj0, cross0, obj1, cross1, cand_lorentz)
    hls::stream<bool> &out_mask_embed,
    hls::stream<bool> &out_mask_obj0,
    hls::stream<bool> &out_mask_cross0,
    hls::stream<bool> &out_mask_obj1,
    hls::stream<bool> &out_mask_cross1,
    hls::stream<bool> &out_mask_cand
    // volatile ap_uint<32>* debug_buf, int dbg_idx
) {
    // debug_buf[dbg_idx] = 1;
    // read raw_jets from axi-stream
    data_t raw_jets[N_MAX][RAW_DIM];
    READ_JETS: for (int i = 0; i < N_MAX; i++) {
        for (int j = 0; j < RAW_DIM; j++) {
            #pragma HLS PIPELINE II=1
            //reinterpret 32-bit unsigned as data_t
            ap_uint<32> bits = in_s.read();
            data_t val;
            val.range(15, 0) =  bits.range(15, 0);
            raw_jets[i][j] = val;
        }
    }

    // read mask from axi-stream
    bool mask[N_MAX];
    READ_MASK: for (int i = 0; i < N_MAX; i++) {
        #pragma HLS PIPELINE II=1
        ap_uint<32> w = in_s.read();
        mask[i] = (w != 0);
    }

    // fork raw jets into 3 output streams

    FORK_JETS: for (int i = 0; i < N_MAX;i++) {
        for (int j = 0 ; j<RAW_DIM; j++) {
            #pragma HLS PIPELINE II=1
            data_t val = raw_jets[i][j];
            out_jets_embed.write(val);
            out_jets_pairwise.write(val);
            out_jets_cand.write(val);
        }
    }

    // fork mask into 6 output streams

    FORK_MASK: for (int i = 0; i < N_MAX;i++) {
        #pragma HLS PIPELINE II=1
        bool val = mask[i];
        out_mask_embed.write(val);
        out_mask_obj0.write(val);
        out_mask_cross0.write(val);
        out_mask_obj1.write(val);
        out_mask_cross1.write(val);
        out_mask_cand.write(val);
    }
    // debug_buf[dbg_idx] = 2;
}

// stage 1 embed
// raw_jets[12*5] + mask[12] -> x[12*16]

inline void embed_stage(
    hls::stream<data_t> &in_jets,
    hls::stream<bool> &in_mask,
    const EmbedWeights &embed_w,
    hls::stream<data_t> &out_embed
    // volatile ap_uint<32>* debug_buf, int dbg_idx
) {
    // debug_buf[dbg_idx] = 1;
    // deserialize
    data_t raw_jets[N_MAX][RAW_DIM];
    stream_to_array2d<N_MAX, RAW_DIM>(in_jets, raw_jets);
    bool mask[N_MAX];
    stream_to_array1d<N_MAX>(in_mask, mask);

    // run kernel
    data_t x[N_MAX][E_DIM];
    embed_ffn(raw_jets, mask, embed_w, x);

    // serialize
    array2d_to_stream<N_MAX, E_DIM>(x, out_embed);
    // debug_buf[dbg_idx] = 2;
}

// pairwise stage
// raw_jets[12*5] -> extract angular -> pairwise_mlp -> expand_wij -> wij_bias[48*13]

inline void pairwise_stage(
    hls::stream<data_t> &in_jets,
    const MLPWeights &mlp_w,
    hls::stream<score_t> &out_wij_bias
    // volatile ap_uint<32>* debug_buf, int dbg_idx
) {
    // debug_buf[dbg_idx] = 1;
    // deserialize
    data_t raw_jets[N_MAX][RAW_DIM];
    stream_to_array2d<N_MAX, RAW_DIM>(in_jets, raw_jets);

    // extract angular features

    data_t w_ang[N_MAX][3];
    EXTRACT_ANG: for (int j = 0; j < N_MAX; j++) {
        #pragma HLS PIPELINE II=1
        w_ang[j][0] = raw_jets[j][1]; //eta
        w_ang[j][1] = raw_jets[j][2]; //cos(phi)
        w_ang[j][2] = raw_jets[j][3]; //sin(phi)
    }

    // run mlp kernel
    data_t wij[N_MAX][N_MAX];
    pairwise_mlp(w_ang, mlp_w, wij);

    // serialize the 144 unique values; the consumer replicates across heads
    // (streaming the expanded [48][13] form cost 4.3x the FIFO + stream traffic)
    WRITE_WIJ: for (int i = 0; i < N_MAX; i++) {
        for (int j = 0; j < N_MAX; j++) {
            #pragma HLS PIPELINE II=1
            out_wij_bias.write((score_t)wij[i][j]);
        }
    }
    // debug_buf[dbg_idx] = 2;
}

// The ABC layers are split into obj / cand / cross DATAFLOW processes.
// Previously each layer ran all three attention blocks sequentially inside
// ONE process, so the pipeline II was ~3 attention blocks; split, the
// steady-state II is the slowest single block. Per-event math is unchanged
// (same op order within an event); only cross-event overlap improves.

// obj stage: obj_attn (+wij for layer 0) -> remask -> build_candidates
// emits the updated x (to cross) and the raw candidates c (to cand)

inline void obj0_stage(
    hls::stream<data_t> &in_embed,
    hls::stream<score_t> &in_wij_bias,
    hls::stream<bool> &in_mask,
    const AttnWeights &obj_w,
    hls::stream<data_t> &out_x,
    hls::stream<data_t> &out_c
) {
    data_t x[N_MAX][E_DIM];
    stream_to_array2d<N_MAX,E_DIM>(in_embed, x);

    data_t wij[N_MAX][N_MAX];
    READ_WIJ: for(int i = 0; i < N_MAX; i++) {
        for (int j = 0; j < N_MAX; j++) {
            #pragma HLS PIPELINE II=1
            wij[i][j] = (data_t)in_wij_bias.read();
        }
    }
    score_t wij_bias[N_MAX*N_HEADS][N_KV];
    expand_wij(wij, wij_bias);

    bool mask[N_MAX];
    stream_to_array1d<N_MAX>(in_mask, mask);

    attn_block_obj(x, mask, wij_bias, /*use_wij =*/ true,
        obj_w.Wq, obj_w.bq, obj_w.Wk, obj_w.bk, obj_w.Wv, obj_w.bv,
        obj_w.bias_k, obj_w.bias_v, obj_w.Wo, obj_w.bo,
        obj_w.attn_ln_g, obj_w.attn_ln_b,
        obj_w.ffn_w, obj_w.ffn_b, obj_w.ffn_ln_g, obj_w.ffn_ln_b,
        obj_w.post_ffn_g, obj_w.post_ffn_b);
    remask(x, mask);

    data_t c[T_DIM][E_DIM];
    int jet_assign_tmp[N_MAX];
    build_candidates<N_MAX>(x, c, jet_assign_tmp);

    array2d_to_stream<N_MAX, E_DIM>(x, out_x);
    array2d_to_stream<T_DIM, E_DIM>(c, out_c);
}

inline void obj1_stage(
    hls::stream<data_t> &in_x,
    hls::stream<bool> &in_mask,
    const AttnWeights &obj_w,
    hls::stream<data_t> &out_x,
    hls::stream<data_t> &out_c
) {
    data_t x[N_MAX][E_DIM];
    stream_to_array2d<N_MAX,E_DIM>(in_x, x);
    bool mask[N_MAX];
    stream_to_array1d<N_MAX>(in_mask, mask);

    // no wij bias in layer 1; the dummy is never read (use_wij=false)
    score_t dummy_wij[N_MAX * N_HEADS][N_KV] = {{0}};
    attn_block_obj(x, mask, dummy_wij, /*use_wij =*/ false,
        obj_w.Wq, obj_w.bq, obj_w.Wk, obj_w.bk, obj_w.Wv, obj_w.bv,
        obj_w.bias_k, obj_w.bias_v, obj_w.Wo, obj_w.bo,
        obj_w.attn_ln_g, obj_w.attn_ln_b,
        obj_w.ffn_w, obj_w.ffn_b, obj_w.ffn_ln_g, obj_w.ffn_ln_b,
        obj_w.post_ffn_g, obj_w.post_ffn_b);
    remask(x, mask);

    data_t c[T_DIM][E_DIM];
    int jet_assign_tmp[N_MAX];
    build_candidates<N_MAX>(x, c, jet_assign_tmp);

    array2d_to_stream<N_MAX, E_DIM>(x, out_x);
    array2d_to_stream<T_DIM, E_DIM>(c, out_c);
}

// cand stage: candidate self-attention. layer 1 also feeds cand_lorentz.

inline void cand_stage(
    hls::stream<data_t> &in_c,
    const AttnWeights &cand_w,
    hls::stream<data_t> &out_c
) {
    data_t c[T_DIM][E_DIM];
    stream_to_array2d<T_DIM,E_DIM>(in_c, c);
    attn_block_cand(c,
        cand_w.Wq, cand_w.bq, cand_w.Wk, cand_w.bk, cand_w.Wv, cand_w.bv,
        cand_w.bias_k, cand_w.bias_v, cand_w.Wo, cand_w.bo,
        cand_w.attn_ln_g, cand_w.attn_ln_b,
        cand_w.ffn_w, cand_w.ffn_b, cand_w.ffn_ln_g, cand_w.ffn_ln_b,
        cand_w.post_ffn_g, cand_w.post_ffn_b);
    array2d_to_stream<T_DIM, E_DIM>(c, out_c);
}

inline void cand_stage2(
    hls::stream<data_t> &in_c,
    const AttnWeights &cand_w,
    hls::stream<data_t> &out_c_cross,
    hls::stream<data_t> &out_c_lorentz
) {
    data_t c[T_DIM][E_DIM];
    stream_to_array2d<T_DIM,E_DIM>(in_c, c);
    attn_block_cand(c,
        cand_w.Wq, cand_w.bq, cand_w.Wk, cand_w.bk, cand_w.Wv, cand_w.bv,
        cand_w.bias_k, cand_w.bias_v, cand_w.Wo, cand_w.bo,
        cand_w.attn_ln_g, cand_w.attn_ln_b,
        cand_w.ffn_w, cand_w.ffn_b, cand_w.ffn_ln_g, cand_w.ffn_ln_b,
        cand_w.post_ffn_g, cand_w.post_ffn_b);
    array2d_to_stream<T_DIM, E_DIM>(c, out_c_cross);
    array2d_to_stream<T_DIM, E_DIM>(c, out_c_lorentz);
}

// cross stage: cross-attention (x attends to c) + remask

inline void cross_stage(
    hls::stream<data_t> &in_x,
    hls::stream<data_t> &in_c,
    hls::stream<bool> &in_mask,
    const AttnWeights &cross_w,
    hls::stream<data_t> &out_x
) {
    data_t x[N_MAX][E_DIM];
    stream_to_array2d<N_MAX,E_DIM>(in_x, x);
    data_t c[T_DIM][E_DIM];
    stream_to_array2d<T_DIM,E_DIM>(in_c, c);
    bool mask[N_MAX];
    stream_to_array1d<N_MAX>(in_mask, mask);

    attn_block_cross(x, c,
        cross_w.Wq, cross_w.bq, cross_w.Wk, cross_w.bk, cross_w.Wv, cross_w.bv,
        cross_w.bias_k, cross_w.bias_v, cross_w.Wo, cross_w.bo,
        cross_w.attn_ln_g, cross_w.attn_ln_b,
        cross_w.ffn_w, cross_w.ffn_b, cross_w.ffn_ln_g, cross_w.ffn_ln_b,
        cross_w.post_ffn_g, cross_w.post_ffn_b);
    remask(x, mask);

    array2d_to_stream<N_MAX, E_DIM>(x, out_x);
}

// cand lorentz stage

inline void cand_lorentz_stage(
    hls::stream<data_t> &in_jets,
    hls::stream<data_t> &in_x,
    hls::stream<data_t> &in_c,
    hls::stream<bool> &in_mask,
    hls::stream<data_t> &out_ae_input
    // volatile ap_uint<32>* debug_buf, int dbg_idx
) {
    //deserialize

    // debug_buf[dbg_idx] = 1;

    data_t raw_jets[N_MAX][RAW_DIM];
    stream_to_array2d<N_MAX, RAW_DIM>(in_jets, raw_jets);
    data_t x[N_MAX][E_DIM];
    stream_to_array2d<N_MAX, E_DIM>(in_x, x);
    data_t c[T_DIM][E_DIM];
    stream_to_array2d<T_DIM, E_DIM>(in_c, c);
    bool mask[N_MAX];
    stream_to_array1d<N_MAX>(in_mask, mask);

    // run kernel

    float jp4[N_MAX][P4_DIM];
    int jet_assign[N_MAX];
    float cand_p4[T_DIM][P4_DIM];
    float cand_mass_scaled[T_DIM];
    data_t ae_input[T_DIM][AE_IN_DIM];
    cand_lorentz(raw_jets, x, c, mask,
        jp4, jet_assign, cand_p4, cand_mass_scaled, ae_input);

    // serialize only candidates 0 and 1 (not ISR at index 2)

    WRITE_AE: for (int t = 0; t < 2; t++) {
        for (int i = 0; i < AE_IN_DIM; i++) {
            #pragma HLS PIPELINE II=1
            out_ae_input.write(ae_input[t][i]);
        } 
    }    

    // debug_buf[dbg_idx] = 2;
}

// ae loss stage
// take ae input [2 * AE_IN_DIM] -> dual autoencoder -> 3 loss scalars

inline void ae_loss_stage(
    hls::stream<data_t> &in_ae,
    const AEEncoderWeights &ae_enc_w,
    const AEDecoderWeights &ae_dec_w,
    hls::stream<float> &out_losses
    // volatile ap_uint<32>* debug_buf, int dbg_idx
) {
    // debug_buf[dbg_idx] = 1;
    // deserialize cand 0 and cand 1
    data_t c0_in[1][AE_IN_DIM], c1_in[1][AE_IN_DIM];
    for (int i = 0; i < AE_IN_DIM; i++) {
        #pragma HLS PIPELINE II=1
        c0_in[0][i] = in_ae.read();
    }
    for (int i = 0; i < AE_IN_DIM; i++) {
        #pragma HLS PIPELINE II=1
        c1_in[0][i] = in_ae.read();
    }

    // run dual autoencoder

    data_t c0_latent[1][AE_DIM], c1_latent[1][AE_DIM];
    data_t c0_decoded[1][AE_IN_DIM], c1_decoded[1][AE_IN_DIM];
    float mse_loss, mse_crossed_loss, latent_dist;
    dual_autoencoder(c0_in, c1_in, ae_enc_w, ae_dec_w,
        c0_latent, c1_latent, c0_decoded, c1_decoded,
        mse_loss, mse_crossed_loss, latent_dist);

    // write 3 loss scalars

    out_losses.write(mse_loss);
    out_losses.write(mse_crossed_loss);
    out_losses.write(latent_dist);
    // debug_buf[dbg_idx] = 2;
}

// serialize 3 loss scalars to axi-stream output

inline void write_output(
    hls::stream<float> &in_losses,
    hls::stream<ap_uint<32>> &out_s
) {
    for (int i = 0; i < 3; i++) {
        #pragma HLS PIPELINE II=1
        float val = in_losses.read();
        ap_uint<32> bits = *(ap_uint<32>*)&val;
        out_s.write(bits);
    }
}


// DDR -> stream (DATAFLOW stage)
static void read_input(const ap_uint<32>* in_buf, int offset, hls::stream<ap_uint<32>>& out) {
    #pragma HLS INLINE off
    for (int i = 0; i < 72; i++) {
        #pragma HLS PIPELINE II=1
        out.write(in_buf[offset + i]);
    }
}

// stream -> DDR
static void write_output_ddr(hls::stream<ap_uint<32>>& in, ap_uint<32>* out_buf, int offset) {
    #pragma HLS INLINE off
    for (int i = 0; i < 3; i++) {
        #pragma HLS PIPELINE II=1
        out_buf[offset + i] = in.read();
    }
}

// top level DATAFLOW 
// stages are run concurrently - data flows through hls::stream FIFOs
// weights are read-only and synthesize to BRAM 

inline void passwd_dataflow(
    // axi stream io
    const ap_uint<32>* in_buf, int in_offset,
    ap_uint<32>* out_buf, int out_offset,
    // weights (stored in bram)

    const EmbedWeights &embed_w,
    const MLPWeights &mlp_w,
    const AttnWeights &obj0_w,
    const AttnWeights &cand0_w,
    const AttnWeights &cross0_w,
    const AttnWeights &obj1_w,
    const AttnWeights &cand1_w,
    const AttnWeights &cross1_w,
    const AEEncoderWeights &ae_enc_w,
    const AEDecoderWeights &ae_dec_w
    // volatile ap_uint<32>* debug_buf
) {
    
    #pragma HLS DATAFLOW

    // internal axi-stream FIFOs
    hls::stream<ap_uint<32>> in_stream("mm2s");
    hls::stream<ap_uint<32>> out_stream("s2mm");
    #pragma HLS STREAM variable=in_stream depth = 72
    #pragma HLS STREAM variable=out_stream depth = 4

    // read from ddr into internal stream

    read_input(in_buf, in_offset, in_stream);

    // internal streams 
    // stream depths = number of elements per event (prevents deadlock)

    // fork outputs for raw_jets (3 consumers)
    hls::stream<data_t> s_jets_embed("jets_embed");
    hls::stream<data_t> s_jets_pairwise("jets_pair");
    hls::stream<data_t> s_jets_cand("jets_cand");
    #pragma HLS STREAM variable = s_jets_embed depth = 60
    #pragma HLS STREAM variable = s_jets_pairwise depth = 60
    #pragma HLS STREAM variable = s_jets_cand depth = 60

    // fork outputs for masks (6 consumers)
    hls::stream<bool> s_mask_embed("mask_embed");
    hls::stream<bool> s_mask_obj0("mask_obj0");
    hls::stream<bool> s_mask_cross0("mask_cross0");
    hls::stream<bool> s_mask_obj1("mask_obj1");
    hls::stream<bool> s_mask_cross1("mask_cross1");
    hls::stream<bool> s_mask_cand("mask_cand");
    #pragma HLS STREAM variable = s_mask_embed depth = 12
    #pragma HLS STREAM variable = s_mask_obj0 depth = 12
    #pragma HLS STREAM variable = s_mask_cross0 depth = 12
    #pragma HLS STREAM variable = s_mask_obj1 depth = 12
    #pragma HLS STREAM variable = s_mask_cross1 depth = 12
    #pragma HLS STREAM variable = s_mask_cand depth = 12

    // inter stage data streams
    // depths cover one full event payload so a producer can always finish
    // its event without consumer progress (deadlock-safe by construction)
    hls::stream<data_t> s_embed("embed");
    hls::stream<score_t> s_wij("wij");
    hls::stream<data_t> s_x0a("x_obj0"), s_c0a("c_obj0"), s_c0b("c_cand0");
    hls::stream<data_t> s_x0("x_layer0");
    hls::stream<data_t> s_x1a("x_obj1"), s_c1a("c_obj1"), s_c1b("c_cand1");
    hls::stream<data_t> s_x1("x_layer1");
    hls::stream<data_t> s_c1("c_layer1");
    hls::stream<data_t> s_ae("ae_input");
    hls::stream<float> s_losses("losses");

    #pragma HLS STREAM variable =  s_embed depth = 192
    #pragma HLS STREAM variable =  s_wij depth = 144
    #pragma HLS STREAM variable =  s_x0a depth = 192
    #pragma HLS STREAM variable =  s_c0a depth = 48
    #pragma HLS STREAM variable =  s_c0b depth = 48
    #pragma HLS STREAM variable =  s_x0 depth = 192
    #pragma HLS STREAM variable =  s_x1a depth = 192
    #pragma HLS STREAM variable =  s_c1a depth = 48
    #pragma HLS STREAM variable =  s_c1b depth = 48
    #pragma HLS STREAM variable =  s_x1 depth = 192
    #pragma HLS STREAM variable =  s_c1 depth = 48
    #pragma HLS STREAM variable =  s_ae depth = 28
    #pragma HLS STREAM variable =  s_losses depth = 4

    // pipeline stages (concurrent under DATAFLOW)

    read_and_fork(in_stream,
        s_jets_embed, s_jets_pairwise, s_jets_cand,
        s_mask_embed, s_mask_obj0, s_mask_cross0,
        s_mask_obj1, s_mask_cross1, s_mask_cand);

    embed_stage(s_jets_embed, s_mask_embed, embed_w, s_embed);

    pairwise_stage(s_jets_pairwise, mlp_w, s_wij);

    // layer 0: obj -> {cand, cross}
    obj0_stage(s_embed, s_wij, s_mask_obj0, obj0_w, s_x0a, s_c0a);
    cand_stage(s_c0a, cand0_w, s_c0b);
    cross_stage(s_x0a, s_c0b, s_mask_cross0, cross0_w, s_x0);

    // layer 1: obj -> {cand, cross}; cand also feeds cand_lorentz
    obj1_stage(s_x0, s_mask_obj1, obj1_w, s_x1a, s_c1a);
    cand_stage2(s_c1a, cand1_w, s_c1b, s_c1);
    cross_stage(s_x1a, s_c1b, s_mask_cross1, cross1_w, s_x1);

    cand_lorentz_stage(s_jets_cand, s_x1, s_c1, s_mask_cand, s_ae);

    ae_loss_stage(s_ae, ae_enc_w, ae_dec_w, s_losses);

    write_output(s_losses, out_stream);
    write_output_ddr(out_stream, out_buf, out_offset);
    
}

#endif