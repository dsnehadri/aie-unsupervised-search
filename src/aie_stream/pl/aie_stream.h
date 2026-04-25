#ifndef AIE_STREAM_H
#define AIE_STREAM_H

// flow per event

// pl : read and fork
// pl : embed stage
// pl : pairwise stage
// pl <-> aie : obj_attn_bridge_L0
// pl : candidate_build_stage_L0
// pl <-> aie : cand_attn_bridge_L0
// pl <-> aie : cross_attn_bridge_L0 (with remask on the way out)
// pl <-> aie : obj_attn_bridge_L1
// pl : candidate_build_stage_L1
// pl <-> aie : cand_attn_bridge_L1
// pl <-> aie : cross_attn_bridge_L1
// pl : cand_lorentz_stage
// pl : ae_loss_stage
// pl : write_output

#include "hls_stream.h"
#include "ap_axi_sdata.h"

#include "../../embed_ffn/embed_ffn.h"
#include "../../pairwise_mlp/pairwise_mlp.h"
#include "../../cand_lorentz/cand_lorentz.h"
#include "../../autoencoder/autoencoder.h"
#include "../../cand_build/candidate_build.h"

#include "../../attn_block_pl/attn_block_obj.h"
#include "../../attn_block_pl/attn_block_cand.h"
#include "../../attn_block_pl/attn_block_cross.h"

#include "../../pl_stream/pl_stream.h"
#include "bridge_stages.h"

// candidate build stage
// takes remasked X, produced C = jet_choice^T @ X
// also re-emits X for downstream cross attn

inline void candidate_build_stage(
    hls::stream<data_t>& in_x, 
    hls::stream<bool>& in_mask,
    hls::stream<data_t>& out_x, 
    hls::stream<data_t>& out_c) {
    data_t x[N_MAX][E_DIM];
    stream_to_array2d<N_MAX, E_DIM>(in_x, x);
    bool mask[N_MAX];
    stream_to_array1d<N_MAX>(in_mask, mask);

    // build candidates
    data_t c[T_DIM][E_DIM];
    int jet_assign_tmp[N_MAX];
    build_candidates<N_MAX>(x, c, jet_assign_tmp);

    // emit c, x and x (for layer transition)
    array2d_to_stream<T_DIM, E_DIM>(c, out_c);
    array2d_to_stream<N_MAX, E_DIM>(x, out_x);
}

// remask stage - zero out padded jets in x

inline void remask_stage(
    hls::stream<data_t>& in_x,
    hls::stream<bool>& in_mask,
    hls::stream<data_t>& out_x
) {
    data_t x[N_MAX][E_DIM];
    stream_to_array2d<N_MAX, E_DIM>(in_x, x);
    bool mask[N_MAX];
    stream_to_array1d<N_MAX>(in_mask, mask);

    remask(x, mask);
    array2d_to_stream<N_MAX, E_DIM>(x, out_x);
}

// top level dataflow

inline void aie_stream(
    //ddr io
    const ap_uint<32>* in_buf, int in_offset,
    ap_uint<32>* out_buf, int out_offset,

    // aie connections - obj_attn layer 0
    hls::stream<pkt64_t>& obj0_x_out, hls::stream<pkt64_t>& obj0_w0_out,
    hls::stream<pkt64_t>& obj0_w1_out, hls::stream<pkt64_t>& obj0_w2_out,
    hls::stream<pkt64_t>& obj0_w3_out, hls::stream<pkt64_t>& obj0_x_in,

    // cand_attn layer 0

    hls::stream<pkt64_t>& cand0_c_out, hls::stream<pkt64_t>& cand0_c_in,

    // cross_attn layer 0
    hls::stream<pkt64_t>& cross0_x_out, hls::stream<pkt64_t>& cross0_c_out,
    hls::stream<pkt64_t>& cross0_x_in,  

    // obj_attn layer 1
    hls::stream<pkt64_t>& obj1_x_out, hls::stream<pkt64_t>& obj1_w0_out,
    hls::stream<pkt64_t>& obj1_w1_out, hls::stream<pkt64_t>& obj1_w2_out,
    hls::stream<pkt64_t>& obj1_w3_out, hls::stream<pkt64_t>& obj1_x_in,

    // cand_attn layer 1

    hls::stream<pkt64_t>& cand1_c_out, hls::stream<pkt64_t>& cand1_c_in,

    // cross_attn layer 1
    hls::stream<pkt64_t>& cross1_x_out, hls::stream<pkt64_t>& cross1_c_out,
    hls::stream<pkt64_t>& cross1_x_in, 

    // pl weights

    const EmbedWeights& embed_w,
    const MLPWeights& mlp_w,
    const AEEncoderWeights& ae_enc_w,
    const AEDecoderWeights& ae_dec_w
) {
    #pragma HLS DATAFLOW

    // ddr read
    hls::stream<ap_uint<32>> in_stream("mm2s");
    #pragma HLS STREAM variable = in_stream depth=72

    hls::stream<ap_uint<32>> out_stream("s2mm");
    #pragma HLS STREAM variable = out_stream depth=72

    read_input(in_buf, in_offset, in_stream);

    // fork raw jets/mask

    hls::stream<data_t> s_jets_embed, s_jets_pairwise, s_jets_cand;
    hls::stream<bool> s_mask_embed, s_mask_cb0, s_mask_cb1, s_mask_cand;
    hls::stream<bool> s_mask_remask0, s_mask_remask1;
    #pragma HLS STREAM variable=s_jets_embed depth = 60
    #pragma HLS STREAM variable=s_jets_pairwise depth = 60
    #pragma HLS STREAM variable=s_jets_cand depth = 60
    #pragma HLS STREAM variable=s_mask_embed depth= 12
    #pragma HLS STREAM variable=s_mask_cb0 depth = 12
    #pragma HLS STREAM variable=s_mask_cb1 depth = 12
    #pragma HLS STREAM variable=s_mask_cand depth = 12
    #pragma HLS STREAM variable=s_mask_remask0 depth = 12
    #pragma HLS STREAM variable=s_mask_remask1 depth = 12

    // adapt read_and_fork to produce additional mask consumers
    read_and_fork_aie(in_stream,
    s_jets_embed, s_jets_pairwise, s_jets_cand,
    s_mask_embed, s_mask_cb0, s_mask_cb1, s_mask_cand,
    s_mask_remask0, s_mask_remask1);

    // embed + pairwise 
    hls::stream<data_t> s_embed;
    hls::stream<score_t> s_wij0; 
    #pragma HLS STREAM variable=s_embed depth = 192
    #pragma HLS STREAM variable=s_wij0 depth = 624

    embed_stage(s_jets_embed, s_mask_embed, embed_w, s_embed);
    pairwise_stage(s_jets_pairwise, mlp_w, s_wij0);

    // abc layer 0

    // obj_attn L0 bridge
    hls::stream<data_t> s_x0_after_obj;
    #pragma HLS STREAM variable=s_x0_after_obj depth = 192
    obj_attn_bridge(s_embed, s_wij0,
        obj0_x_out, obj0_w0_out, obj0_w1_out, obj0_w2_out, obj0_w3_out,
        obj0_x_in, s_x0_after_obj);

    // remask after obj_attn

    hls::stream<data_t> s_x0_remasked;
    #pragma HLS STREAM variable=s_x0_remasked depth=192
    remask_stage(s_x0_after_obj, s_mask_remask0, s_x0_remasked);

    //build candidates
    hls::stream<data_t> s_x0_for_cross, s_c0_for_cand;
    #pragma HLS STREAM variable=s_x0_for_cross depth = 192
    #pragma HLS STREAM variable=s_c0_for_cand depth = 48
    candidate_build_stage(s_x0_remasked, s_mask_cb0,
        s_x0_for_cross, s_c0_for_cand);

    // cand_attn L0 bridge
    hls::stream<data_t> s_c0_after_cand;
    #pragma HLS STREAM variable=s_c0_after_cand depth = 48
    cand_attn_bridge(s_c0_for_cand, cand0_c_out, cand0_c_in, s_c0_after_cand);

    hls::stream<data_t> s_x0_after_cross;
    #pragma HLS STREAM variable = s_x0_after_cross depth = 192
    cross_attn_bridge(s_x0_for_cross, s_c0_after_cand,
        cross0_x_out, cross0_c_out, cross0_x_in, s_x0_after_cross);

    // abc layer 1

    hls::stream<score_t> s_wij1_dummy;
    #pragma HLS STREAM variable=s_wij1_dummy depth=624
    emit_zero_wij(s_wij1_dummy);

    hls::stream<data_t> s_x1_after_obj;
    #pragma HLS STREAM variable=s_x1_after_obj depth = 192
    obj_attn_bridge(s_x0_after_cross, s_wij1_dummy,
        obj1_x_out, obj1_w0_out, obj1_w1_out, obj1_w2_out, obj1_w3_out,
        obj1_x_in, s_x1_after_obj);

    hls::stream<data_t> s_x1_remasked;
    #pragma HLS STREAM variable=s_x1_remasked depth=192
    remask_stage(s_x1_after_obj, s_mask_remask1, s_x1_remasked);

    //build candidates
    hls::stream<data_t> s_x1_for_cross, s_c1_for_cand;
    #pragma HLS STREAM variable=s_x1_for_cross depth = 192
    #pragma HLS STREAM variable=s_c1_for_cand depth = 48
    candidate_build_stage(s_x1_remasked, s_mask_cb1,
        s_x1_for_cross, s_c1_for_cand);

    hls::stream<data_t> s_c1_after_cand;
    #pragma HLS STREAM variable=s_c1_after_cand depth = 48
    cand_attn_bridge(s_c1_for_cand, cand1_c_out, cand1_c_in, s_c1_after_cand);

    hls::stream<data_t> s_x1_after_cross;
    #pragma HLS STREAM variable = s_x1_after_cross depth = 192
    cross_attn_bridge(s_x1_for_cross, s_c1_after_cand,
        cross1_x_out, cross1_c_out, cross1_x_in, s_x1_after_cross);

    // cand lorentz and dual autoencoder

    hls::stream<data_t> s_ae;
    hls::stream<float> s_losses;
    #pragma HLS STREAM variable = s_ae depth = 24
    #pragma HLS STREAM variable = s_losses depth = 4

    // cand_lorentz needs raw_jets, final x, final c, mask

    cand_lorentz_stage(s_jets_cand, s_x1_after_cross, s_c1_after_cand,
        s_mask_cand, s_ae);

    ae_loss_stage(s_ae, ae_enc_w, ae_dec_w, s_losses);
    write_output(s_losses, out_stream);
    write_output_ddr(out_stream, out_buf, out_offset);
}

#endif