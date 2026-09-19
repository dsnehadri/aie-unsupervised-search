// q512_top — 4x128-bit PLIO channels fed from one 512-bit m_axi @250MHz:
// aggregate-bandwidth test (finds the DDR/NoC wall, not the wire).
// Beat layout: 512-bit DDR beat i -> lane j gets bits [128j+127:128j]; the
// drain packs identically, so passthrough preserves the identity pattern.
#include <hls_stream.h>
#include <ap_int.h>
#include <ap_axi_sdata.h>
typedef ap_axiu<128,0,0,0> pkt128_t;

static void feed(ap_uint<512>* in, int nbeats, hls::stream<pkt128_t> o[4]){
  for(int i=0;i<nbeats;i++){
    #pragma HLS PIPELINE II=1
    #pragma HLS LOOP_TRIPCOUNT min=8 max=262144
    ap_uint<512> w=in[i];
    for(int j=0;j<4;j++){
      #pragma HLS UNROLL
      pkt128_t p; p.data=w.range(128*j+127,128*j); p.keep=-1; p.last=(i==nbeats-1)?1:0;
      o[j].write(p);
    }
  }
}
static void drain(hls::stream<pkt128_t> s[4], ap_uint<512>* out, int nbeats){
  for(int i=0;i<nbeats;i++){
    #pragma HLS PIPELINE II=1
    #pragma HLS LOOP_TRIPCOUNT min=8 max=262144
    ap_uint<512> w;
    for(int j=0;j<4;j++){
      #pragma HLS UNROLL
      w.range(128*j+127,128*j)=s[j].read().data;
    }
    out[i]=w;
  }
}
extern "C" void q512_top(ap_uint<512>* in_buf, ap_uint<512>* out_buf, int nbeats,
    hls::stream<pkt128_t> to_aie[4], hls::stream<pkt128_t> from_aie[4]){
  #pragma HLS INTERFACE m_axi port=in_buf  offset=slave bundle=gmem0 depth=262144
  #pragma HLS INTERFACE m_axi port=out_buf offset=slave bundle=gmem1 depth=262144
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
