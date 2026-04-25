// top level pl kernel. all attn stages route through aie via
// axi-stream ports that v++ connects to aie plio

// port count
// ddr: 2 (m_axi in_buf out_buf)
// obj_attn L0: 5 AXI streams, 1 in (X + 4 wij + X_in from AIE)
// cand_attn L0: 1 out + 1 in
// cross_attn L0: 2 out + in
// obj_attn L1: 5 out + 1 in
// cand_attn L1: 1 out + 1 in
// cross_attn L1: 2 out + 1 in
// total AXI stream: 22 ports

#include "aie_stream.h"
#include "../../pl_stream/weights_rom.h"

// weight state

static bool weights_initialized = false;
static EmbedWeights embed_w;
static MLPWeights mlp_w;
static AEEncoderWeights ae_enc_w;
static AEDecoderWeights ae_dec_w;

extern "C" void aie_stream_top(
    //ddr io
    ap_uint<32>* in_buf,
    ap_uint<32>* out_buf,
    int n_events,

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
    hls::stream<pkt64_t>& cross1_x_in
) 
{
    // ddr interfaces

    #pragma HLS INTERFACE m_axi port=in_buf offset=slave bundle = gmem0 depth = 720
    #pragma HLS INTERFACE m_axi port=out_buf offset=slave bundle = gmem1 depth = 30

    // all 22 axi stream ports
    #pragma HLS INTERFACE axis port = obj0_x_out
    #pragma HLS INTERFACE axis port = obj0_w0_out
    #pragma HLS INTERFACE axis port = obj0_w1_out
    #pragma HLS INTERFACE axis port = obj0_w2_out
    #pragma HLS INTERFACE axis port = obj0_w3_out
    #pragma HLS INTERFACE axis port = obj0_x_in

    #pragma HLS INTERFACE axis port = cand0_c_out
    #pragma HLS INTERFACE axis port = cand0_c_in

    #pragma HLS INTERFACE axis port = cross0_x_out
    #pragma HLS INTERFACE axis port = cross0_c_out
    #pragma HLS INTERFACE axis port = cross0_x_in

    #pragma HLS INTERFACE axis port = obj1_x_out
    #pragma HLS INTERFACE axis port = obj1_w0_out
    #pragma HLS INTERFACE axis port = obj1_w1_out
    #pragma HLS INTERFACE axis port = obj1_w2_out
    #pragma HLS INTERFACE axis port = obj1_w3_out
    #pragma HLS INTERFACE axis port = obj1_x_in

    #pragma HLS INTERFACE axis port = cand1_c_out
    #pragma HLS INTERFACE axis port = cand1_c_in

    #pragma HLS INTERFACE axis port = cross1_x_out
    #pragma HLS INTERFACE axis port = cross1_c_out
    #pragma HLS INTERFACE axis port = cross1_x_in

    #pragma HLS INTERFACE s_axilite port = in_buf
    #pragma HLS INTERFACE s_axilite port = out_buf
    #pragma HLS INTERFACE s_axilite port = n_events
    #pragma HLS INTERFACE s_axilite port = return

    // initialize weights

    if (!weights_initialized) {
        init_pl_only_weights(embed_w, mlp_w, ae_enc_w, ae_dec_w);
        weights_initialized = true;
    }

    // per-event dataflow
    for (int ev = 0; ev < n_events; ev++) {
        aie_stream(
            in_buf, ev * 72,
            out_buf, ev*3,
            obj0_x_out, obj0_w0_out, obj0_w1_out, obj0_w2_out, obj0_w3_out, obj0_x_in,
            cand0_c_out, cand0_c_in,
            cross0_x_out, cross0_c_out, cross0_x_in,
            obj1_x_out, obj1_w0_out, obj1_w1_out, obj1_w2_out, obj1_w3_out, obj1_x_in,
            cand1_c_out, cand1_c_in,
            cross1_x_out, cross1_c_out, cross1_x_in,
            embed_w, mlp_w, ae_enc_w, ae_dec_w);
    }

}