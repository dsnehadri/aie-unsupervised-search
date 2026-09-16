// obj24_top (independent feeders, v6) -- up to 20 object-attention instances on
// the array. v5 fed every instance in lockstep, so the slowest instance (the
// third) set the pace for all. Here each instance has its own feeder and drain:
//   loader : reads one event (976 words) once and gives each active feeder a copy
//   feed[i]: sends that event n_ev times to instance i
//   drain[i]: reads instance i's outputs; a free-running cycle counter records
//            the clock cycle of the first and last output beat, so instance i's
//            time per event = (last - first) / n_ev cycles, measured while all
//            instances run at the same time
//   writer : writes each active instance's last output (192 words) and its two
//            cycle counts (64 bits each, as two 32-bit words)
#include <hls_stream.h>
#include <ap_int.h>
#include <ap_axi_sdata.h>
typedef ap_axiu<64,0,0,0> pkt64_t;
static const int N_MAX=12, E_DIM=16;
static const int X_WORDS=(N_MAX+1)*E_DIM;   // 208
static const int WIJ_WORDS=N_MAX*16;        // 192 (padded bias)
static const int IN_PER=X_WORDS+4*WIJ_WORDS;// 976
static const int XB=X_WORDS/4, WB=WIJ_WORDS/4, OB=192/4;   // 52, 48, 48 beats
static const int EVB=XB+4*WB;               // 244 beats per event
static const int OUT_PER=192+4;             // output words + first(2) + last(2)
static const int NINST=20;
typedef ap_uint<64> b64;

static inline pkt64_t beat(b64 d, bool last){ pkt64_t p; p.data=d; p.keep=-1; p.last=last; return p; }

static void loader(ap_uint<32>* in, int n_inst, hls::stream<b64> cfg[NINST]){
  b64 ev[EVB];
  b64 w=0;
  for(int idx=0; idx<IN_PER; idx++){
    #pragma HLS PIPELINE II=1
    ap_uint<32> v=in[idx];
    w=(w>>16) | (b64(v.range(15,0))<<48);    // every count is a multiple of 4
    if((idx&3)==3) ev[idx>>2]=w;
  }
  for(int i=0;i<NINST;i++){
    cfg[i].write(b64(i<n_inst ? 1 : 0));      // header: active or not
    if(i<n_inst){
      for(int b=0;b<EVB;b++){
        #pragma HLS PIPELINE II=1
        cfg[i].write(ev[b]);
      }
    }
  }
}

static void feed(hls::stream<b64>& cfg, int n_ev, hls::stream<pkt64_t>& x,
    hls::stream<pkt64_t>& w0, hls::stream<pkt64_t>& w1, hls::stream<pkt64_t>& w2, hls::stream<pkt64_t>& w3){
  if(cfg.read()==0) return;
  b64 xb[XB], wb0[WB], wb1[WB], wb2[WB], wb3[WB];
  for(int b=0;b<XB;b++){
    #pragma HLS PIPELINE II=1
    xb[b]=cfg.read(); }
  for(int b=0;b<WB;b++){
    #pragma HLS PIPELINE II=1
    wb0[b]=cfg.read(); }
  for(int b=0;b<WB;b++){
    #pragma HLS PIPELINE II=1
    wb1[b]=cfg.read(); }
  for(int b=0;b<WB;b++){
    #pragma HLS PIPELINE II=1
    wb2[b]=cfg.read(); }
  for(int b=0;b<WB;b++){
    #pragma HLS PIPELINE II=1
    wb3[b]=cfg.read(); }
  for(int e=0;e<n_ev;e++){
    #pragma HLS LOOP_TRIPCOUNT min=1 max=4096
    for(int b=0;b<XB;b++){
      #pragma HLS PIPELINE II=1
      x.write(beat(xb[b], b==XB-1));
      if(b<WB){ w0.write(beat(wb0[b], b==WB-1)); w1.write(beat(wb1[b], b==WB-1));
                w2.write(beat(wb2[b], b==WB-1)); w3.write(beat(wb3[b], b==WB-1)); }
    }
  }
}

static void drain(int idx, int n_inst, int n_ev, hls::stream<pkt64_t>& xin, hls::stream<b64>& res){
  if(idx>=n_inst) return;
  const ap_uint<32> total=ap_uint<32>(n_ev)*OB;
  b64 lastb[OB];
  b64 cnt=0, first=0, lastc=0;
  ap_uint<32> got=0;
  while(got<total){
    #pragma HLS PIPELINE II=1
    #pragma HLS LOOP_TRIPCOUNT min=48 max=196608
    if(!xin.empty()){
      pkt64_t p=xin.read();
      if(got==0) first=cnt;
      if(got>=total-OB) lastb[got-(total-OB)]=p.data;
      if(got==total-1) lastc=cnt;
      got++;
    }
    cnt++;
  }
  for(int b=0;b<OB;b++){
    #pragma HLS PIPELINE II=1
    res.write(lastb[b]); }
  res.write(first);
  res.write(lastc);
}

static void writer(int n_inst, hls::stream<b64> res[NINST], ap_uint<32>* out){
  for(int i=0;i<NINST;i++){
    if(i<n_inst){
      for(int b=0;b<OB+2;b++){
        b64 d=res[i].read();
        if(b<OB){
          for(int j=0;j<4;j++){
            #pragma HLS PIPELINE II=1
            ap_uint<32> v=0; v.range(15,0)=d.range(j*16+15,j*16); out[i*OUT_PER+b*4+j]=v;
          }
        } else {
          int base=i*OUT_PER+192+(b-OB)*2;
          out[base]=d.range(31,0); out[base+1]=d.range(63,32);
        }
      }
    }
  }
}

extern "C" void obj24_top(ap_uint<32>* in_buf, ap_uint<32>* out_buf, int n_inst, int n_ev,
    hls::stream<pkt64_t> x_s[NINST], hls::stream<pkt64_t> w0_s[NINST],
    hls::stream<pkt64_t> w1_s[NINST], hls::stream<pkt64_t> w2_s[NINST],
    hls::stream<pkt64_t> w3_s[NINST], hls::stream<pkt64_t> xin_s[NINST]){
  #pragma HLS INTERFACE m_axi port=in_buf  offset=slave bundle=gmem0 depth=976 max_widen_bitwidth=32
  #pragma HLS INTERFACE m_axi port=out_buf offset=slave bundle=gmem1 depth=3920 max_widen_bitwidth=32
  #pragma HLS INTERFACE axis port=x_s
  #pragma HLS INTERFACE axis port=w0_s
  #pragma HLS INTERFACE axis port=w1_s
  #pragma HLS INTERFACE axis port=w2_s
  #pragma HLS INTERFACE axis port=w3_s
  #pragma HLS INTERFACE axis port=xin_s
  #pragma HLS INTERFACE s_axilite port=in_buf
  #pragma HLS INTERFACE s_axilite port=out_buf
  #pragma HLS INTERFACE s_axilite port=n_inst
  #pragma HLS INTERFACE s_axilite port=n_ev
  #pragma HLS INTERFACE s_axilite port=return
  #pragma HLS DATAFLOW
  hls::stream<b64> cfg[NINST];
  hls::stream<b64> res[NINST];
  #pragma HLS STREAM variable=cfg depth=256
  #pragma HLS STREAM variable=res depth=64
  loader(in_buf, n_inst, cfg);
#define INST(i) feed(cfg[i], n_ev, x_s[i], w0_s[i], w1_s[i], w2_s[i], w3_s[i]); drain(i, n_inst, n_ev, xin_s[i], res[i]);
  INST(0) INST(1) INST(2) INST(3) INST(4) INST(5) INST(6) INST(7) INST(8) INST(9)
  INST(10) INST(11) INST(12) INST(13) INST(14) INST(15) INST(16) INST(17) INST(18) INST(19)
#undef INST
  writer(n_inst, res, out_buf);
}
