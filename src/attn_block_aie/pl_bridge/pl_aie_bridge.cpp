// hls implementation of pl to aie bridge
// compile with
// v++ -c -k pl_aie_bridge --platform $PLATFORM \
//         --hls.clock 250000000 \
//         -o pl_aie_bridge.xo \
//         src/pl_bridge/pl_aie_bridge.cpp

#include "pl_aie_bridge.h"

static const int N_MAX = 12;
static const int E_DIM = 16;
static const int N_HEADS = 4;
static const int N_KV = 13;

// pack 4 int16 values into one 64-bit AXI-stream beat

static void pack_and_send(hls::stream<pkt64_t>& out, data_t buf[], int count) 
{
    #pragma HLS INLINE off
    for (int i = 0; i < count; i+=4) {
        #pragma HLS PIPELINE II=1
        pkt64_t pkt;
        ap_uint<64> word = 0;
        for (int j = 0; j < 4; j++) {
            if (i + j < count) {
                ap_uint<16> bits = buf[i + j].range(15, 0);
                word.range(j * 16 + 15, j * 16) = bits;
            }
        }
        pkt.data = word;
        pkt.keep = -1;
        pkt.last = (i + 4 >= count) ? 1 : 0;
        out.write(pkt);
    }
}

// unpack 64-bit AXI-stream beats into int16 values
static void recv_and_unpack(hls::stream<pkt64_t>& in, data_t buf[], int count)
{
    #pragma HLS INLINE off
    for (int i = 0; i < count; i += 4) {
        #pragma HLS PIPELINE II=1
        pkt64_t pkt = in.read();
        ap_uint<64> word = pkt.data;
        for (int j = 0; j < 4; j++) {
            if (i + j < count) {
                ap_uint<16> bits = word.range(j * 16 + 15, j * 16);
                data_t val;
                val.range(15, 0) = bits;
                buf[i + j] = val;
            }
        }
    }
}

extern "C" void pl_aie_bridge(
    hls::stream<data_t>& embed_in,
    hls::stream<data_t>& wij_in,
    hls::stream<pkt64_t>& obj_x_out,
    hls::stream<pkt64_t>& wij_h0_out,
    hls::stream<pkt64_t>& wij_h1_out,
    hls::stream<pkt64_t>& wij_h2_out,
    hls::stream<pkt64_t>& wij_h3_out,
    hls::stream<pkt64_t>& obj_x_in,
    hls::stream<data_t>& attn_out
)
{
    #pragma HLS INTERFACE axis port=embed_in
    #pragma HLS INTERFACE axis port=wij_in
    #pragma HLS INTERFACE axis port=obj_x_out
    #pragma HLS INTERFACE axis port=wij_h0_out
    #pragma HLS INTERFACE axis port=wij_h1_out
    #pragma HLS INTERFACE axis port=wij_h2_out
    #pragma HLS INTERFACE axis port=wij_h3_out
    #pragma HLS INTERFACE axis port=obj_x_in
    #pragma HLS INTERFACE ap_ctrl_none port=return

    // read embedded x from PL pipeline
    data_t x_buf[N_MAX * E_DIM];
    #pragma HLS ARRAY_PARTITION variable = x_buf cycle factor = 4
    for (int i = 0; i < N_MAX * E_DIM; i++) {
        #pragma HLS PIPELINE II=1
        x_buf[i] = embed_in.read();
    }

    // read wij from pairwise mlp
    // layout: [N_HEADS][N_MAX][N_KV] - head major
    data_t wij_buf[N_HEADS * N_MAX * N_KV];
    for (int i = 0; i < N_HEADS * N_MAX * N_KV; i++) {
        #pragma HLS PIPELINE II=1
        wij_buf[i] = wij_in.read();
    }

    // send x to aie
    pack_and_send(obj_x_out, x_buf, N_MAX * E_DIM);

    // send per head wij slices
    data_t wij_slice[N_MAX * N_KV];

    // head 0 
    for (int i = 0; i < N_MAX * N_KV; i++) {
        #pragma HLS PIPELINE II=1
        wij_slice[i] = wij_buf[0 * N_MAX * N_KV + i];
    }
    pack_and_send(wij_h0_out, wij_slice, N_MAX * N_KV);

    // head 1
    for (int i = 0; i < N_MAX * N_KV; i++) {
        #pragma HLS PIPELINE II=1
        wij_slice[i] = wij_buf[1 * N_MAX * N_KV + i];
    }
    pack_and_send(wij_h1_out, wij_slice, N_MAX * N_KV);

    // head 2
    for (int i = 0; i < N_MAX * N_KV; i++) {
        #pragma HLS PIPELINE II=1
        wij_slice[i] = wij_buf[2 * N_MAX * N_KV + i];
    }
    pack_and_send(wij_h2_out, wij_slice, N_MAX * N_KV);

    // head 3
    for (int i = 0; i < N_MAX * N_KV; i++) {
        #pragma HLS PIPELINE II=1
        wij_slice[i] = wij_buf[3 * N_MAX * N_KV + i];
    }
    pack_and_send(wij_h3_out, wij_slice, N_MAX * N_KV);

    // receive attention output from aie
    data_t result_buf[N_MAX * E_DIM];
    recv_and_unpack(obj_x_in, result_buf, N_MAX * E_DIM);

    // forward to next PL stage
    for (int i = 0; i < N_MAX * E_DIM; i++) {
        #pragma HLS PIPELINE II = 1
        attn_out.write(result_buf[i]);
    }
}