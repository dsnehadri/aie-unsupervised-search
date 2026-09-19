// p128_top — single-channel 128-bit PLIO bridge @250MHz: per-channel ceiling test.
#include <hls_stream.h>
#include <ap_int.h>
#include <ap_axi_sdata.h>
typedef ap_axiu<128,0,0,0> pkt128_t;

static void feed(ap_uint<128>* in, int nbeats, hls::stream<pkt128_t>& o){
  for(int i=0;i<nbeats;i++){
    #pragma HLS PIPELINE II=1
    #pragma HLS LOOP_TRIPCOUNT min=8 max=1048576
    pkt128_t p; p.data=in[i]; p.keep=-1; p.last=(i==nbeats-1)?1:0; o.write(p);
  }
}
static void drain(hls::stream<pkt128_t>& s, ap_uint<128>* out, int nbeats){
  for(int i=0;i<nbeats;i++){
    #pragma HLS PIPELINE II=1
    #pragma HLS LOOP_TRIPCOUNT min=8 max=1048576
    out[i]=s.read().data;
  }
}
extern "C" void p128_top(ap_uint<128>* in_buf, ap_uint<128>* out_buf, int nbeats,
    hls::stream<pkt128_t>& to_aie, hls::stream<pkt128_t>& from_aie){
  #pragma HLS INTERFACE m_axi port=in_buf  offset=slave bundle=gmem0 depth=1048576
  #pragma HLS INTERFACE m_axi port=out_buf offset=slave bundle=gmem1 depth=1048576
  #pragma HLS INTERFACE axis port=to_aie
  #pragma HLS INTERFACE axis port=from_aie
  #pragma HLS INTERFACE s_axilite port=in_buf
  #pragma HLS INTERFACE s_axilite port=out_buf
  #pragma HLS INTERFACE s_axilite port=nbeats
  #pragma HLS INTERFACE s_axilite port=return
  #pragma HLS DATAFLOW
  feed(in_buf, nbeats, to_aie);
  drain(from_aie, out_buf, nbeats);
}
