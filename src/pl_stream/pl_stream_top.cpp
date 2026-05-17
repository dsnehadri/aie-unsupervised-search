#include "pl_stream.h"
#include "weights_rom.h"

extern "C" void pl_stream_top(
    ap_uint<32>* in_buf,
    ap_uint<32>* out_buf,
    int n_events
) {
#pragma HLS INTERFACE m_axi port=in_buf  offset=slave bundle=gmem0 depth=720
#pragma HLS INTERFACE m_axi port=out_buf offset=slave bundle=gmem1 depth=30
#pragma HLS INTERFACE s_axilite port=in_buf
#pragma HLS INTERFACE s_axilite port=out_buf
#pragma HLS INTERFACE s_axilite port=n_events
#pragma HLS INTERFACE s_axilite port=return

    EmbedWeights embed_w;
    MLPWeights mlp_w;
    AttnWeights obj0_w, cand0_w, cross0_w;
    AttnWeights obj1_w, cand1_w, cross1_w;
    AEEncoderWeights ae_enc_w;
    AEDecoderWeights ae_dec_w;

    init_all_weights(embed_w, mlp_w,
                     obj0_w, cand0_w, cross0_w,
                     obj1_w, cand1_w, cross1_w,
                     ae_enc_w, ae_dec_w);

    for (int ev = 0; ev < n_events; ev++) {
        passwd_dataflow(in_buf, ev*72, out_buf, ev*3,
            embed_w, mlp_w,
            obj0_w, cand0_w, cross0_w,
            obj1_w, cand1_w, cross1_w,
            ae_enc_w, ae_dec_w);
    }
}