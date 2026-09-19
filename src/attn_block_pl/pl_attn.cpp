// Runtime PL attention vehicle: obj / cand / cross attention blocks in PL fabric
// (no AIE), each as its own kernel, for actual per-block latency/throughput
// measurement (matches the runtime AIE per-block numbers). mask=all-valid and
// wij=0 are set internally (constants) -- the matmul-dominated attention loops
// are data-independent, so timing is representative; output correctness of these
// blocks was validated in the full all-PL model (0.4% vs PyTorch).
#include "../pl_stream/pl_stream.h"
#include "../pl_stream/weights_rom.h"

static const int XW = N_MAX*E_DIM;   // 192
static const int CW = T_DIM*E_DIM;   // 48

static void rd_x(ap_uint<32>* in,int off,data_t x[N_MAX][E_DIM]){
  for(int i=0;i<N_MAX;i++) for(int j=0;j<E_DIM;j++){
    #pragma HLS PIPELINE II=1
    ap_uint<32> w=in[off+i*E_DIM+j]; data_t v; v.range(15,0)=w.range(15,0); x[i][j]=v; } }
static void wr_x(data_t x[N_MAX][E_DIM],ap_uint<32>* out,int off){
  for(int i=0;i<N_MAX;i++) for(int j=0;j<E_DIM;j++){
    #pragma HLS PIPELINE II=1
    data_t v=x[i][j]; ap_uint<32> w=0; w.range(15,0)=v.range(15,0); out[off+i*E_DIM+j]=w; } }
static void rd_c(ap_uint<32>* in,int off,data_t c[T_DIM][E_DIM]){
  for(int i=0;i<T_DIM;i++) for(int j=0;j<E_DIM;j++){
    #pragma HLS PIPELINE II=1
    ap_uint<32> w=in[off+i*E_DIM+j]; data_t v; v.range(15,0)=w.range(15,0); c[i][j]=v; } }
static void wr_c(data_t c[T_DIM][E_DIM],ap_uint<32>* out,int off){
  for(int i=0;i<T_DIM;i++) for(int j=0;j<E_DIM;j++){
    #pragma HLS PIPELINE II=1
    data_t v=c[i][j]; ap_uint<32> w=0; w.range(15,0)=v.range(15,0); out[off+i*E_DIM+j]=w; } }

// Weights are filled locally on every call, as in the fabric-only top, so
// synthesis folds them to constants. Static weights filled once are a writable
// memory whose two ports can make parallel copies take turns (seen in the
// hybrid autoencoder: 389 cycles instead of 264).
#define LOCAL_WEIGHTS EmbedWeights ew; MLPWeights mw; AttnWeights o0,cd0,cr0,o1,cd1,cr1; \
  AEEncoderWeights ae; AEDecoderWeights ad; init_all_weights(ew,mw,o0,cd0,cr0,o1,cd1,cr1,ae,ad);

extern "C" void obj_pl_top(ap_uint<32>* in_buf, ap_uint<32>* out_buf, int n_events){
  #pragma HLS INTERFACE m_axi port=in_buf  offset=slave bundle=gmem0 depth=19200
  #pragma HLS INTERFACE m_axi port=out_buf offset=slave bundle=gmem1 depth=19200
  #pragma HLS INTERFACE s_axilite port=in_buf
  #pragma HLS INTERFACE s_axilite port=out_buf
  #pragma HLS INTERFACE s_axilite port=n_events
  #pragma HLS INTERFACE s_axilite port=return
  LOCAL_WEIGHTS
  bool mask[N_MAX]; for(int i=0;i<N_MAX;i++) mask[i]=true;
  score_t wij[N_MAX*N_HEADS][N_KV];
  for(int i=0;i<N_MAX*N_HEADS;i++) for(int j=0;j<N_KV;j++) wij[i][j]=(score_t)0;
  // BLOCK_ONLY: read one event once and write only the last output, so the
  // per-event time in a batch sweep is the block alone (read and write were
  // ~390 of ~1,590 cycles per event)
  data_t xo[N_MAX][E_DIM];
  // the same arrays are reused every iteration: these blocks take the same
  // number of cycles whatever the values, and a copy loop would add cycles back
  data_t x[N_MAX][E_DIM];
  rd_x(in_buf, 0, x);
  for(int ev=0;ev<n_events;ev++){
    attn_block_obj(x, mask, wij, true,
      o0.Wq,o0.bq,o0.Wk,o0.bk,o0.Wv,o0.bv, o0.bias_k,o0.bias_v,o0.Wo,o0.bo,
      o0.attn_ln_g,o0.attn_ln_b, o0.ffn_w,o0.ffn_b,o0.ffn_ln_g,o0.ffn_ln_b,
      o0.post_ffn_g,o0.post_ffn_b
#ifdef OBJ_DATAFLOW
      , xo);                               // the dataflow block writes a separate output
#else
      );
#endif
  }
#ifdef OBJ_DATAFLOW
  if(n_events>0) wr_x(xo, out_buf, (n_events-1)*XW);
#else
  if(n_events>0) wr_x(x, out_buf, (n_events-1)*XW);
#endif
}
extern "C" void cand_pl_top(ap_uint<32>* in_buf, ap_uint<32>* out_buf, int n_events){
  #pragma HLS INTERFACE m_axi port=in_buf  offset=slave bundle=gmem0 depth=4800
  #pragma HLS INTERFACE m_axi port=out_buf offset=slave bundle=gmem1 depth=4800
  #pragma HLS INTERFACE s_axilite port=in_buf
  #pragma HLS INTERFACE s_axilite port=out_buf
  #pragma HLS INTERFACE s_axilite port=n_events
  #pragma HLS INTERFACE s_axilite port=return
  LOCAL_WEIGHTS
  data_t c[T_DIM][E_DIM];
  rd_c(in_buf, 0, c);
  for(int ev=0;ev<n_events;ev++){
    attn_block_cand(c,
      cd0.Wq,cd0.bq,cd0.Wk,cd0.bk,cd0.Wv,cd0.bv, cd0.bias_k,cd0.bias_v,cd0.Wo,cd0.bo,
      cd0.attn_ln_g,cd0.attn_ln_b, cd0.ffn_w,cd0.ffn_b,cd0.ffn_ln_g,cd0.ffn_ln_b,
      cd0.post_ffn_g,cd0.post_ffn_b);
  }
  if(n_events>0) wr_c(c, out_buf, (n_events-1)*CW);
}
extern "C" void cross_pl_top(ap_uint<32>* in_buf, ap_uint<32>* out_buf, int n_events){
  #pragma HLS INTERFACE m_axi port=in_buf  offset=slave bundle=gmem0 depth=24000
  #pragma HLS INTERFACE m_axi port=out_buf offset=slave bundle=gmem1 depth=19200
  #pragma HLS INTERFACE s_axilite port=in_buf
  #pragma HLS INTERFACE s_axilite port=out_buf
  #pragma HLS INTERFACE s_axilite port=n_events
  #pragma HLS INTERFACE s_axilite port=return
  LOCAL_WEIGHTS
  data_t x[N_MAX][E_DIM], c[T_DIM][E_DIM];
  rd_x(in_buf, 0,  x);
  rd_c(in_buf, XW, c);
  for(int ev=0;ev<n_events;ev++){
    attn_block_cross(x, c,
      cr0.Wq,cr0.bq,cr0.Wk,cr0.bk,cr0.Wv,cr0.bv, cr0.bias_k,cr0.bias_v,cr0.Wo,cr0.bo,
      cr0.attn_ln_g,cr0.attn_ln_b, cr0.ffn_w,cr0.ffn_b,cr0.ffn_ln_g,cr0.ffn_ln_b,
      cr0.post_ffn_g,cr0.post_ffn_b);
  }
  if(n_events>0) wr_x(x, out_buf, (n_events-1)*XW);
}
