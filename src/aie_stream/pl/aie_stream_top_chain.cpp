// Whole two-layer ABC stack on the array: the fabric keeps only what precedes
// the first attention block (read, fork, embedding send, pairwise wij) and what
// follows the last (lorentz, autoencoder, write). One dataflow region, every
// stage a persistent process over n_events (as aie_stream_top_pipe.cpp).
// Ports: embed_j_out, mask_out, obj0_w0..3_out (PL->AIE); x_in, c_in (AIE->PL).
#include "/home/snehadri/repos/aie-unsupervised-search/src/aie_stream/pl/aie_stream.h"
#include "/home/snehadri/repos/aie-unsupervised-search/src/pl_stream/weights_rom.h"

static void read_input_loop(const ap_uint<32>* in_buf, int n, hls::stream<ap_uint<32>>& o) {
    for (int e = 0; e < n; e++) read_input(in_buf, e*72, o);
}
// jets -> embedding, pairwise, lorentz; mask -> the array (one PLIO, read by
// four tile kernels) and lorentz
static void fork_chain(hls::stream<ap_uint<32>>& in_s,
    hls::stream<data_t>& je, hls::stream<data_t>& jp, hls::stream<data_t>& jc,
    hls::stream<bool>& m_aie, hls::stream<bool>& m_cand)
{
    data_t raw_jets[N_MAX][RAW_DIM];
    READ_JETS: for (int i = 0; i < N_MAX; i++)
        for (int j = 0; j < RAW_DIM; j++) {
            #pragma HLS PIPELINE II=1
            ap_uint<32> bits = in_s.read();
            data_t val; val.range(15, 0) = bits.range(15, 0);
            raw_jets[i][j] = val;
        }
    bool mask[N_MAX];
    READ_MASK: for (int i = 0; i < N_MAX; i++) {
        #pragma HLS PIPELINE II=1
        ap_uint<32> w = in_s.read();
        mask[i] = (w != 0);
    }
    FORK_JETS: for (int i = 0; i < N_MAX; i++)
        for (int j = 0; j < RAW_DIM; j++) {
            #pragma HLS PIPELINE II=1
            data_t val = raw_jets[i][j];
            je.write(val); jp.write(val); jc.write(val);
        }
    FORK_MASK: for (int i = 0; i < N_MAX; i++) {
        #pragma HLS PIPELINE II=1
        m_aie.write(mask[i]); m_cand.write(mask[i]);
    }
}
static void fork_loop(hls::stream<ap_uint<32>>& in, int n,
    hls::stream<data_t>& je, hls::stream<data_t>& jp, hls::stream<data_t>& jc,
    hls::stream<bool>& m_aie, hls::stream<bool>& m_cand) {
    for (int e = 0; e < n; e++) fork_chain(in, je, jp, jc, m_aie, m_cand);
}
static void embed_send_loop(hls::stream<data_t>& j, hls::stream<pkt64_t>& jo, int n) {
    for (int e = 0; e < n; e++) embed_send(j, jo);
}
// mask row for the array: 16 words, 1 = padded jet (what obj_attn_send appended)
static void mask_send(hls::stream<bool>& m, hls::stream<pkt64_t>& mo) {
    data_t buf[E_DIM];
    MASKROW: for (int j = 0; j < E_DIM; j++) {
        #pragma HLS PIPELINE II=1
        bool mv = (j < N_MAX) ? m.read() : false;
        buf[j] = mv ? (data_t)1 : (data_t)0;
    }
    pack_buf_to_axi<E_DIM>(buf, mo);
}
static void mask_send_loop(hls::stream<bool>& m, hls::stream<pkt64_t>& mo, int n) {
    for (int e = 0; e < n; e++) mask_send(m, mo);
}
static void pairwise_loop(hls::stream<data_t>& i, const MLPWeights& w, hls::stream<score_t>& o, int n) {
    for (int e = 0; e < n; e++) pairwise_stage(i, w, o);
}
// the wij part of obj_attn_send: 144 unique values -> N_MAX x N_KV slice per
// head (column N_MAX zero), Q10.5 -> Q8.7 by << 2, four identical PLIOs
#if defined(WIJ_ONE_PORT)
static void wij_send(hls::stream<score_t>& wij_in_pl, hls::stream<pkt64_t>& w0)
#else
static void wij_send(hls::stream<score_t>& wij_in_pl,
    hls::stream<pkt64_t>& w0, hls::stream<pkt64_t>& w1, hls::stream<pkt64_t>& w2, hls::stream<pkt64_t>& w3)
#endif
{
    const int WIJ_SZ = N_MAX * N_KV;
    data_t wij_buf[N_MAX * N_MAX];
    for (int i = 0; i < N_MAX * N_MAX; i++) {
        #pragma HLS PIPELINE II=1
        score_t s = wij_in_pl.read();
        data_t v; v.range(15, 0) = s.range(15, 0);
        wij_buf[i] = v;
    }
    data_t slice[WIJ_SZ];
    for (int i = 0; i < N_MAX; i++)
        for (int j = 0; j < N_KV; j++) {
            #pragma HLS PIPELINE II=1
            data_t v = 0;
            if (j < N_MAX) {
                ap_int<16> bits = wij_buf[i * N_MAX + j].range(15, 0);
                ap_int<16> shifted = (ap_int<16>)(bits << 2);
                v.range(15, 0) = shifted.range(15, 0);
            }
            slice[i * N_KV + j] = v;
        }
#if defined(WIJ_ONE_PORT)
    // All four heads were sent the SAME slice: 3 of the 4 copies were pure
    // duplicate traffic over the interface. Send it once and let the array's
    // stream switch multicast it to the four head-post kernels.
    pack_buf_to_axi<WIJ_SZ>(slice, w0);
#else
    pack_buf_to_axi<WIJ_SZ>(slice, w0);
    pack_buf_to_axi<WIJ_SZ>(slice, w1);
    pack_buf_to_axi<WIJ_SZ>(slice, w2);
    pack_buf_to_axi<WIJ_SZ>(slice, w3);
#endif
}
#if defined(WIJ_ONE_PORT)
static void wij_send_loop(hls::stream<score_t>& wij, hls::stream<pkt64_t>& w0, int n) {
    for (int e = 0; e < n; e++) wij_send(wij, w0);
}
#else
static void wij_send_loop(hls::stream<score_t>& wij, hls::stream<pkt64_t>& w0, hls::stream<pkt64_t>& w1,
    hls::stream<pkt64_t>& w2, hls::stream<pkt64_t>& w3, int n) {
    for (int e = 0; e < n; e++) wij_send(wij, w0, w1, w2, w3);
}
#endif
static void x_recv_loop(hls::stream<pkt64_t>& xi, hls::stream<data_t>& o, int n) {
    for (int e = 0; e < n; e++) unpack_axi_to_stream<N_MAX * E_DIM>(xi, o);
}
static void c_recv_loop(hls::stream<pkt64_t>& ci, hls::stream<data_t>& o, int n) {
    for (int e = 0; e < n; e++) unpack_axi_to_stream<T_DIM * E_DIM>(ci, o);
}
static void lorentz_loop(hls::stream<data_t>& jc, hls::stream<data_t>& x,
    hls::stream<data_t>& c, hls::stream<bool>& m, hls::stream<data_t>& o, int n) {
    for (int e = 0; e < n; e++) cand_lorentz_stage(jc, x, c, m, o);
}
static void ae_loop(hls::stream<data_t>& i, const AEEncoderWeights& enc,
    const AEDecoderWeights& dec, hls::stream<float>& o, int n) {
    for (int e = 0; e < n; e++) ae_loss_stage(i, enc, dec, o);
}
static void wout_loop(hls::stream<float>& i, hls::stream<ap_uint<32>>& o, int n) {
    for (int e = 0; e < n; e++) write_output(i, o);
}
static void wddr_loop(hls::stream<ap_uint<32>>& i, ap_uint<32>* out_buf, int n) {
    for (int e = 0; e < n; e++) write_output_ddr(i, out_buf, e*3);
}

static void run_chain(const ap_uint<32>* in_buf, ap_uint<32>* out_buf, int n,
    hls::stream<pkt64_t>& embed_j_out, hls::stream<pkt64_t>& mask_out,
    hls::stream<pkt64_t>& obj0_w0_out,
#if !defined(WIJ_ONE_PORT)
    hls::stream<pkt64_t>& obj0_w1_out,
    hls::stream<pkt64_t>& obj0_w2_out, hls::stream<pkt64_t>& obj0_w3_out,
#endif
    hls::stream<pkt64_t>& x_in, hls::stream<pkt64_t>& c_in,
    const MLPWeights& mlp_w, const AEEncoderWeights& ae_enc_w, const AEDecoderWeights& ae_dec_w)
{
    #pragma HLS DATAFLOW
    hls::stream<ap_uint<32>> in_stream("mm2s");
    #pragma HLS STREAM variable=in_stream depth=432
    hls::stream<ap_uint<32>> out_stream("s2mm");
    #pragma HLS STREAM variable=out_stream depth=192
    hls::stream<data_t> s_jets_embed, s_jets_pairwise, s_jets_cand;
    hls::stream<bool> s_mask_aie, s_mask_cand;
    #pragma HLS STREAM variable=s_jets_embed    depth=384
    #pragma HLS STREAM variable=s_jets_pairwise depth=384
    #pragma HLS STREAM variable=s_jets_cand     depth=1152
    #pragma HLS STREAM variable=s_mask_aie      depth=576
    #pragma HLS STREAM variable=s_mask_cand     depth=768
    hls::stream<score_t> s_wij0;
    #pragma HLS STREAM variable=s_wij0 depth=864
    hls::stream<data_t> s_x1, s_c1;
    #pragma HLS STREAM variable=s_x1 depth=1152
    #pragma HLS STREAM variable=s_c1 depth=288
    hls::stream<data_t> s_ae; hls::stream<float> s_losses;
    #pragma HLS STREAM variable=s_ae depth=144
    #pragma HLS STREAM variable=s_losses depth=48

    read_input_loop(in_buf, n, in_stream);
    fork_loop(in_stream, n, s_jets_embed, s_jets_pairwise, s_jets_cand, s_mask_aie, s_mask_cand);
    embed_send_loop(s_jets_embed, embed_j_out, n);
    mask_send_loop(s_mask_aie, mask_out, n);
    pairwise_loop(s_jets_pairwise, mlp_w, s_wij0, n);
#if defined(WIJ_ONE_PORT)
    wij_send_loop(s_wij0, obj0_w0_out, n);
#else
    wij_send_loop(s_wij0, obj0_w0_out, obj0_w1_out, obj0_w2_out, obj0_w3_out, n);
#endif
    x_recv_loop(x_in, s_x1, n);
    c_recv_loop(c_in, s_c1, n);
    lorentz_loop(s_jets_cand, s_x1, s_c1, s_mask_cand, s_ae, n);
    ae_loop(s_ae, ae_enc_w, ae_dec_w, s_losses, n);
    wout_loop(s_losses, out_stream, n);
    wddr_loop(out_stream, out_buf, n);
}

static bool weights_initialized = false;
static EmbedWeights embed_w; static MLPWeights mlp_w;
static AEEncoderWeights ae_enc_w; static AEDecoderWeights ae_dec_w;

extern "C" void aie_stream_top(
    ap_uint<32>* in_buf, ap_uint<32>* out_buf, int n_events,
    hls::stream<pkt64_t>& embed_j_out, hls::stream<pkt64_t>& mask_out,
    hls::stream<pkt64_t>& obj0_w0_out,
#if !defined(WIJ_ONE_PORT)
    hls::stream<pkt64_t>& obj0_w1_out,
    hls::stream<pkt64_t>& obj0_w2_out, hls::stream<pkt64_t>& obj0_w3_out,
#endif
    hls::stream<pkt64_t>& x_in, hls::stream<pkt64_t>& c_in)
{
    #pragma HLS INTERFACE m_axi port=in_buf offset=slave bundle=gmem0 depth=720
    #pragma HLS INTERFACE m_axi port=out_buf offset=slave bundle=gmem1 depth=30
    #pragma HLS INTERFACE axis port=embed_j_out
    #pragma HLS INTERFACE axis port=mask_out
    #pragma HLS INTERFACE axis port=obj0_w0_out
#if !defined(WIJ_ONE_PORT)
    #pragma HLS INTERFACE axis port=obj0_w1_out
    #pragma HLS INTERFACE axis port=obj0_w2_out
    #pragma HLS INTERFACE axis port=obj0_w3_out
#endif
    #pragma HLS INTERFACE axis port=x_in
    #pragma HLS INTERFACE axis port=c_in
    #pragma HLS INTERFACE s_axilite port=in_buf
    #pragma HLS INTERFACE s_axilite port=out_buf
    #pragma HLS INTERFACE s_axilite port=n_events
    #pragma HLS INTERFACE s_axilite port=return

    if (!weights_initialized) {
        init_pl_only_weights(embed_w, mlp_w, ae_enc_w, ae_dec_w);
        weights_initialized = true;
    }
    run_chain(in_buf, out_buf, n_events, embed_j_out, mask_out,
              obj0_w0_out,
#if !defined(WIJ_ONE_PORT)
              obj0_w1_out, obj0_w2_out, obj0_w3_out,
#endif
              x_in, c_in,
              mlp_w, ae_enc_w, ae_dec_w);
}
