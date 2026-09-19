// Per-block batch feeders: each kernel streams n_events events back-to-back
// into ONE AIE attention block and drains the results, so a batch-size sweep
// isolates the block's steady-state per-event interval from launch overhead.
// Word layout per event (int16, 4 per 64-bit beat) matches bridge_stages.h:
//   obj  : x 13x16=208 (mask row appended) + 4 heads x wij 12x16=192 (padded) -> out 12x16=192
//   cand : c 3x16=48                                                  -> out 48
//   cross: x 12x16=192 + c 3x16=48                                    -> out 192
#include <hls_stream.h>
#include <ap_int.h>
#include <ap_axi_sdata.h>
typedef ap_axiu<64,0,0,0> pkt64_t;

template<int COUNT>
static void send_words(ap_uint<32>* in, int off, hls::stream<pkt64_t>& o){
  for(int b=0;b<(COUNT+3)/4;b++){
    #pragma HLS PIPELINE II=1
    ap_uint<64> w=0;
    for(int j=0;j<4;j++){
      #pragma HLS UNROLL
      int idx=b*4+j; if(idx<COUNT){ ap_uint<32> v=in[off+idx]; w.range(j*16+15,j*16)=v.range(15,0); }
    }
    pkt64_t p; p.data=w; p.keep=-1; p.last=(b==(COUNT+3)/4-1)?1:0; o.write(p);
  }
}
template<int COUNT>
static void recv_words(hls::stream<pkt64_t>& in, ap_uint<32>* out, int off){
  for(int b=0;b<(COUNT+3)/4;b++){
    #pragma HLS PIPELINE II=1
    pkt64_t p=in.read(); ap_uint<64> w=p.data;
    for(int j=0;j<4;j++){
      #pragma HLS UNROLL
      int idx=b*4+j; if(idx<COUNT){ ap_uint<32> v=0; v.range(15,0)=w.range(j*16+15,j*16); out[off+idx]=v; }
    }
  }
}


// FEED_PRELOAD: the per-word memory reads were the bottleneck, not the block
// (object feeder 1,225 cycles per event: its bursts were reverted). Read the
// first event once into 64-bit beats, then send those beats for every event,
// all of a block's streams in the same clock cycle. The blocks' timing does not
// depend on the data, and every output is still checked for change.
template<int COUNT, int NB>
static void load_beats(ap_uint<32>* in, int off, ap_uint<64> beats[NB]){
  // every COUNT used here is a multiple of 4: shift each word in from the top,
  // so after four words word 0 sits in bits 15:0 (a variable part-select here
  // cost ~72k LUT per loop)
  ap_uint<64> w=0;
  for(int idx=0; idx<COUNT; idx++){
    #pragma HLS PIPELINE II=1
    ap_uint<32> v=in[off+idx];
    w=(w>>16) | (ap_uint<64>(v.range(15,0))<<48);
    if((idx&3)==3) beats[idx>>2]=w;
  }
}
static inline pkt64_t beat(ap_uint<64> d, bool last){ pkt64_t p; p.data=d; p.keep=-1; p.last=last; return p; }

// ---------------- object attention ----------------
static const int OBJ_X=208, OBJ_W=192, OBJ_IN=OBJ_X+4*OBJ_W, OBJ_OUT=192;
static void obj_feed(ap_uint<32>* in,int n,hls::stream<pkt64_t>& x,hls::stream<pkt64_t>& w0,
    hls::stream<pkt64_t>& w1,hls::stream<pkt64_t>& w2,hls::stream<pkt64_t>& w3){
  const int XB=(OBJ_X+3)/4, WB=(OBJ_W+3)/4;
  ap_uint<64> xb[(OBJ_X+3)/4], wb0[(OBJ_W+3)/4], wb1[(OBJ_W+3)/4], wb2[(OBJ_W+3)/4], wb3[(OBJ_W+3)/4];
  load_beats<OBJ_X,(OBJ_X+3)/4>(in, 0, xb);
  load_beats<OBJ_W,(OBJ_W+3)/4>(in, OBJ_X+0*OBJ_W, wb0); load_beats<OBJ_W,(OBJ_W+3)/4>(in, OBJ_X+1*OBJ_W, wb1);
  load_beats<OBJ_W,(OBJ_W+3)/4>(in, OBJ_X+2*OBJ_W, wb2); load_beats<OBJ_W,(OBJ_W+3)/4>(in, OBJ_X+3*OBJ_W, wb3);
  for(int e=0;e<n;e++){
    #pragma HLS LOOP_TRIPCOUNT min=1 max=256
    for(int b=0;b<XB;b++){
      #pragma HLS PIPELINE II=1
      x.write(beat(xb[b], b==XB-1));
      if(b<WB){ w0.write(beat(wb0[b], b==WB-1)); w1.write(beat(wb1[b], b==WB-1));
                w2.write(beat(wb2[b], b==WB-1)); w3.write(beat(wb3[b], b==WB-1)); }
    }
  }
}
static void obj_drain(hls::stream<pkt64_t>& xin,ap_uint<32>* out,int n){
  for(int e=0;e<n;e++){
    #pragma HLS LOOP_TRIPCOUNT min=1 max=256
    recv_words<OBJ_OUT>(xin,out,e*OBJ_OUT);
  }
}
extern "C" void objb_top(ap_uint<32>* in_buf, ap_uint<32>* out_buf, int n_events,
    hls::stream<pkt64_t>& x_s, hls::stream<pkt64_t>& w0_s, hls::stream<pkt64_t>& w1_s,
    hls::stream<pkt64_t>& w2_s, hls::stream<pkt64_t>& w3_s, hls::stream<pkt64_t>& xin_s){
  #pragma HLS INTERFACE m_axi port=in_buf  offset=slave bundle=gmem0 depth=212992 max_widen_bitwidth=32
  #pragma HLS INTERFACE m_axi port=out_buf offset=slave bundle=gmem1 depth=49152
  #pragma HLS INTERFACE axis port=x_s
  #pragma HLS INTERFACE axis port=w0_s
  #pragma HLS INTERFACE axis port=w1_s
  #pragma HLS INTERFACE axis port=w2_s
  #pragma HLS INTERFACE axis port=w3_s
  #pragma HLS INTERFACE axis port=xin_s
  #pragma HLS INTERFACE s_axilite port=in_buf
  #pragma HLS INTERFACE s_axilite port=out_buf
  #pragma HLS INTERFACE s_axilite port=n_events
  #pragma HLS INTERFACE s_axilite port=return
  #pragma HLS DATAFLOW
  obj_feed(in_buf,n_events,x_s,w0_s,w1_s,w2_s,w3_s);
  obj_drain(xin_s,out_buf,n_events);
}

// ---------------- candidate attention ----------------
static const int CAND_IN=48, CAND_OUT=48;
static void cand_feed(ap_uint<32>* in,int n,hls::stream<pkt64_t>& c){
  const int CB=(CAND_IN+3)/4;
  ap_uint<64> cb[(CAND_IN+3)/4];
  load_beats<CAND_IN,(CAND_IN+3)/4>(in, 0, cb);
  for(int e=0;e<n;e++){
    #pragma HLS LOOP_TRIPCOUNT min=1 max=256
    for(int b=0;b<CB;b++){
      #pragma HLS PIPELINE II=1
      c.write(beat(cb[b], b==CB-1));
    }
  }
}
static void cand_drain(hls::stream<pkt64_t>& cin,ap_uint<32>* out,int n){
  for(int e=0;e<n;e++){
    #pragma HLS LOOP_TRIPCOUNT min=1 max=256
    recv_words<CAND_OUT>(cin,out,e*CAND_OUT);
  }
}
extern "C" void candb_top(ap_uint<32>* in_buf, ap_uint<32>* out_buf, int n_events,
    hls::stream<pkt64_t>& c_s, hls::stream<pkt64_t>& cin_s){
  #pragma HLS INTERFACE m_axi port=in_buf  offset=slave bundle=gmem0 depth=12288 max_widen_bitwidth=32
  #pragma HLS INTERFACE m_axi port=out_buf offset=slave bundle=gmem1 depth=12288
  #pragma HLS INTERFACE axis port=c_s
  #pragma HLS INTERFACE axis port=cin_s
  #pragma HLS INTERFACE s_axilite port=in_buf
  #pragma HLS INTERFACE s_axilite port=out_buf
  #pragma HLS INTERFACE s_axilite port=n_events
  #pragma HLS INTERFACE s_axilite port=return
  #pragma HLS DATAFLOW
  cand_feed(in_buf,n_events,c_s);
  cand_drain(cin_s,out_buf,n_events);
}

// ---------------- cross attention ----------------
static const int CR_X=192, CR_C=48, CR_IN=CR_X+CR_C, CR_OUT=192;
static void cross_feed(ap_uint<32>* in,int n,hls::stream<pkt64_t>& x,hls::stream<pkt64_t>& c){
  const int XB=(CR_X+3)/4, CB=(CR_C+3)/4;
  ap_uint<64> xb[(CR_X+3)/4], cb[(CR_C+3)/4];
  load_beats<CR_X,(CR_X+3)/4>(in, 0, xb);
  load_beats<CR_C,(CR_C+3)/4>(in, CR_X, cb);
  for(int e=0;e<n;e++){
    #pragma HLS LOOP_TRIPCOUNT min=1 max=256
    for(int b=0;b<XB;b++){
      #pragma HLS PIPELINE II=1
      x.write(beat(xb[b], b==XB-1));
      if(b<CB) c.write(beat(cb[b], b==CB-1));
    }
  }
}
static void cross_drain(hls::stream<pkt64_t>& xin,ap_uint<32>* out,int n){
  for(int e=0;e<n;e++){
    #pragma HLS LOOP_TRIPCOUNT min=1 max=256
    recv_words<CR_OUT>(xin,out,e*CR_OUT);
  }
}
extern "C" void crossb_top(ap_uint<32>* in_buf, ap_uint<32>* out_buf, int n_events,
    hls::stream<pkt64_t>& x_s, hls::stream<pkt64_t>& c_s, hls::stream<pkt64_t>& xin_s){
  #pragma HLS INTERFACE m_axi port=in_buf  offset=slave bundle=gmem0 depth=61440 max_widen_bitwidth=32
  #pragma HLS INTERFACE m_axi port=out_buf offset=slave bundle=gmem1 depth=49152
  #pragma HLS INTERFACE axis port=x_s
  #pragma HLS INTERFACE axis port=c_s
  #pragma HLS INTERFACE axis port=xin_s
  #pragma HLS INTERFACE s_axilite port=in_buf
  #pragma HLS INTERFACE s_axilite port=out_buf
  #pragma HLS INTERFACE s_axilite port=n_events
  #pragma HLS INTERFACE s_axilite port=return
  #pragma HLS DATAFLOW
  cross_feed(in_buf,n_events,x_s,c_s);
  cross_drain(xin_s,out_buf,n_events);
}
