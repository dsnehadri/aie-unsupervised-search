// obj24_top (batched, v5) -- up to 20 object-attention instances on the array
// (15 tiles each). Unlike the first sweep design, which sent ONE event to each
// instance per call through one serial feeder (so the feeder, ~16 us per event,
// capped the curve), this one:
//   - loads one event (976 words, bias rows padded to 16 columns) once per call,
//   - sends n_ev events to every active instance, all streams in the same cycle,
//   - reads every active instance's output in the same cycle and writes only the
//     last event's outputs back to memory.
// The block's timing does not depend on the data. Time per call vs n_ev gives the
// per-event time for n_inst instances; aggregate throughput = n_inst / that time.
#include <hls_stream.h>
#include <ap_int.h>
#include <ap_axi_sdata.h>
typedef ap_axiu<64,0,0,0> pkt64_t;
static const int N_MAX=12, E_DIM=16;
static const int X_WORDS=(N_MAX+1)*E_DIM;   // 208: 12 rows + mask row
static const int WIJ_WORDS=N_MAX*16;        // 192: padded to 16 columns (WIJ_PAD16)
static const int IN_PER=X_WORDS+4*WIJ_WORDS;// 976
static const int OUT_WORDS=N_MAX*E_DIM;     // 192
static const int XB=X_WORDS/4, WB=WIJ_WORDS/4, OB=OUT_WORDS/4;   // 52, 48, 48 beats
static const int NINST=20;

// every count here is a multiple of 4: shift each word in from the top
template<int COUNT, int NB>
static void load_beats(ap_uint<32>* in, int off, ap_uint<64> beats[NB]){
  ap_uint<64> w=0;
  for(int idx=0; idx<COUNT; idx++){
    #pragma HLS PIPELINE II=1
    ap_uint<32> v=in[off+idx];
    w=(w>>16) | (ap_uint<64>(v.range(15,0))<<48);
    if((idx&3)==3) beats[idx>>2]=w;
  }
}
static inline pkt64_t beat(ap_uint<64> d, bool last){ pkt64_t p; p.data=d; p.keep=-1; p.last=last; return p; }

static void feed(ap_uint<32>* in, int n_inst, int n_ev,
    hls::stream<pkt64_t> x[NINST], hls::stream<pkt64_t> w0[NINST],
    hls::stream<pkt64_t> w1[NINST], hls::stream<pkt64_t> w2[NINST], hls::stream<pkt64_t> w3[NINST]){
  ap_uint<64> xb[XB], wb0[WB], wb1[WB], wb2[WB], wb3[WB];
  load_beats<X_WORDS,XB>(in, 0, xb);
  load_beats<WIJ_WORDS,WB>(in, X_WORDS+0*WIJ_WORDS, wb0);
  load_beats<WIJ_WORDS,WB>(in, X_WORDS+1*WIJ_WORDS, wb1);
  load_beats<WIJ_WORDS,WB>(in, X_WORDS+2*WIJ_WORDS, wb2);
  load_beats<WIJ_WORDS,WB>(in, X_WORDS+3*WIJ_WORDS, wb3);
  for(int e=0;e<n_ev;e++){
    #pragma HLS LOOP_TRIPCOUNT min=1 max=256
    for(int b=0;b<XB;b++){
      #pragma HLS PIPELINE II=1
      const ap_uint<64> xd=xb[b];
      const bool wv=(b<WB);
      const ap_uint<64> d0=wv?wb0[b]:ap_uint<64>(0), d1=wv?wb1[b]:ap_uint<64>(0),
                        d2=wv?wb2[b]:ap_uint<64>(0), d3=wv?wb3[b]:ap_uint<64>(0);
      for(int i=0;i<NINST;i++){
        if(i<n_inst){
          x[i].write(beat(xd, b==XB-1));
          if(wv){ w0[i].write(beat(d0, b==WB-1)); w1[i].write(beat(d1, b==WB-1));
                  w2[i].write(beat(d2, b==WB-1)); w3[i].write(beat(d3, b==WB-1)); }
        }
      }
    }
  }
}
static void drain(int n_inst, int n_ev, hls::stream<pkt64_t> xin[NINST], ap_uint<32>* out){
  ap_uint<64> last[NINST][OB];
  #pragma HLS ARRAY_PARTITION variable=last dim=1 complete
  for(int e=0;e<n_ev;e++){
    #pragma HLS LOOP_TRIPCOUNT min=1 max=256
    for(int b=0;b<OB;b++){
      #pragma HLS PIPELINE II=1
      for(int i=0;i<NINST;i++){
        if(i<n_inst){ pkt64_t p=xin[i].read(); last[i][b]=p.data; }
      }
    }
  }
  for(int i=0;i<NINST;i++){
    if(i<n_inst){
      for(int b=0;b<OB;b++){
        for(int j=0;j<4;j++){
          #pragma HLS PIPELINE II=1
          ap_uint<32> v=0; v.range(15,0)=last[i][b].range(j*16+15,j*16); out[i*OUT_WORDS+b*4+j]=v;
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
  #pragma HLS INTERFACE m_axi port=out_buf offset=slave bundle=gmem1 depth=3840 max_widen_bitwidth=32
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
  feed(in_buf, n_inst, n_ev, x_s, w0_s, w1_s, w2_s, w3_s);
  drain(n_inst, n_ev, xin_s, out_buf);
}
