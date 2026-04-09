#include "passwd_stream.h"

// hls systhesis top function
// in_stream - axi-stream input (72 words per event, 60 for jets + 12 for mask)
// out_stream - axi_stream output (3 words per event: mse, crossed, latent)
// weights - passed as arguments, synthesized to BRAM

// for deployment, weights would be loaded via AXI-MM or hardcoded
// for csim/cosim: testbench populates them from .npy files

void passwd_stream_top(
    hls::stream<axi_word_t> &in_stream,
    hls::stream<axi_word_t> &out_stream,
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
) {
    // axi-stream ports
    #pragma HLS INTERFACE axis port = in_stream
    #pragma HLS INTERFACE axis port = out_stream

    // control interface - ap_ctrl_none makes kernel free-running
    // (it continuously processes events from the stream without
    // software handshaking). use ap_ctrl_hs if you want start/done signals.
    #pragma HLS INTERFACE ap_ctrl_none port = return

    // weight interfaces - ap_none means they're wired directly (BRAM/ROM)
    // in a real deployment you'd use s_axilite or m_axi to load them
    #pragma HLS INTERFACE ap_none port = embed_w
    #pragma HLS INTERFACE ap_none port = mlp_w
    #pragma HLS INTERFACE ap_none port = obj0_w
    #pragma HLS INTERFACE ap_none port = cand0_w
    #pragma HLS INTERFACE ap_none port = cross0_w
    #pragma HLS INTERFACE ap_none port = obj1_w
    #pragma HLS INTERFACE ap_none port = cand1_w
    #pragma HLS INTERFACE ap_none port = cross1_w
    #pragma HLS INTERFACE ap_none port = ae_enc_w
    #pragma HLS INTERFACE ap_none port = ae_dec_w

    passwd_dataflow(in_stream, out_stream,
        embed_w, mlp_w, 
        obj0_w, cand0_w, cross0_w,
        obj1_w, cand1_w, cross1_w,
        ae_enc_w, ae_dec_w);
}