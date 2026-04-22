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


extern "C" void passwd_stream_top(
    ap_uint<32>* in_buf,
    ap_uint<32>* out_buf,
    int n_events,
    volatile ap_uint<32>* debug_stage
) {
    // axi-stream ports
    #pragma HLS INTERFACE m_axi port=in_buf  offset=slave bundle=gmem0 depth=720 num_read_outstanding=16 max_read_burst_length=64
    #pragma HLS INTERFACE m_axi port=out_buf offset=slave bundle=gmem1 depth=30  num_write_outstanding=16 max_write_burst_length=16
    #pragma HLS INTERFACE s_axilite port = in_buf
    #pragma HLS INTERFACE s_axilite port = out_buf
    #pragma HLS INTERFACE s_axilite port = n_events
    #pragma HLS INTERFACE s_axilite port = debug_stage
    #pragma HLS INTERFACE s_axilite port = return

    *debug_stage = 1;

    if (!weights_initialized) {
        init_all_weights(embed_w, mlp_w,
            obj0_w, cand0_w, cross0_w,
            obj1_w, cand1_w, cross1_w,
            ae_enc_w, ae_dec_w);
        weights_initialized = true;
    }

    *debug_stage = 2;

    for (int ev = 0; ev < n_events; ev++) {

        *debug_stage = 10 + ev;

        passwd_dataflow(in_buf, ev*72, out_buf, ev*3,
        embed_w, mlp_w, 
        obj0_w, cand0_w, cross0_w,
        obj1_w, cand1_w, cross1_w,
        ae_enc_w, ae_dec_w);

        *debug_stage = 100+ev;
    }

    *debug_stage = 999;
}