// lb_top — control kernel: identical structure to pt_top but the stream loops
// back inside the PL (never crosses to the AIE). Its alpha/beta capture XRT
// launch + DDR + PL costs; (pt_top - lb_top) isolates the AIE shim crossing.
// Internal stream is plain ap_uint<64>: HLS forbids hls::axis types off-port.
#include <hls_stream.h>
#include <ap_int.h>

static void feed(ap_uint<64>* in, int nbeats, hls::stream<ap_uint<64> >& o){
  for(int i=0;i<nbeats;i++){
    #pragma HLS PIPELINE II=1
    #pragma HLS LOOP_TRIPCOUNT min=8 max=2097152
    o.write(in[i]);
  }
}
static void drain(hls::stream<ap_uint<64> >& s, ap_uint<64>* out, int nbeats){
  for(int i=0;i<nbeats;i++){
    #pragma HLS PIPELINE II=1
    #pragma HLS LOOP_TRIPCOUNT min=8 max=2097152
    out[i]=s.read();
  }
}
extern "C" void lb_top(ap_uint<64>* in_buf, ap_uint<64>* out_buf, int nbeats){
  #pragma HLS INTERFACE m_axi port=in_buf  offset=slave bundle=gmem0 depth=2097152
  #pragma HLS INTERFACE m_axi port=out_buf offset=slave bundle=gmem1 depth=2097152
  #pragma HLS INTERFACE s_axilite port=in_buf
  #pragma HLS INTERFACE s_axilite port=out_buf
  #pragma HLS INTERFACE s_axilite port=nbeats
  #pragma HLS INTERFACE s_axilite port=return
  #pragma HLS DATAFLOW
  hls::stream<ap_uint<64> > loop_s;
  #pragma HLS STREAM variable=loop_s depth=64
  feed(in_buf, nbeats, loop_s);
  drain(loop_s, out_buf, nbeats);
}
