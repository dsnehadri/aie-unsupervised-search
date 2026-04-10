#include "passwd_stream.h"
#include "weights_rom.h"

// hls systhesis top function
// in_stream - axi-stream input (72 words per event, 60 for jets + 12 for mask)
// out_stream - axi_stream output (3 words per event: mse, crossed, latent)


static bool weights_initialized = false;
static EmbedWeights embed_w;
static MLPWeights mlp_w;
static AttnWeights obj0_w, cand0_w, cross0_w;
static AttnWeights obj1_w, cand1_w, cross1_w;
static AEEncoderWeights ae_enc_w;
static AEDecoderWeights ae_dec_w;


void passwd_stream_top(
    hls::stream<axi_word_t> &in_stream,
    hls::stream<axi_word_t> &out_stream
) {
    // axi-stream ports
    #pragma HLS INTERFACE axis port = in_stream
    #pragma HLS INTERFACE axis port = out_stream

    // control interface - ap_ctrl_none makes kernel free-running
    // (it continuously processes events from the stream without
    // software handshaking). use ap_ctrl_hs if you want start/done signals.
    #pragma HLS INTERFACE ap_ctrl_none port = return

    if (!weights_initialized) {
        init_all_weights(embed_w, mlp_w,
            obj0_w, cand0_w, cross0_w,
            obj1_w, cand1_w, cross1_w,
            ae_enc_w, ae_dec_w);
        weights_initialized = true;
    }

    passwd_dataflow(in_stream, out_stream,
        embed_w, mlp_w, 
        obj0_w, cand0_w, cross0_w,
        obj1_w, cand1_w, cross1_w,
        ae_enc_w, ae_dec_w);
}