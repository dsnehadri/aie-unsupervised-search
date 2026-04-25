// bridges stages that route data between PL pipeline and AIE attention tiles

// each bridge stage:
// 1. deserializes from incoming pl hls::stream
// 2. packs 4 x int16 per 64-bit AXI beat
// 3. writes to the axi-stream connected to the AIE PLIO
// 4. reads the aie result from the return AXI-stream
// 5. unpacks and writes to the outgoing PL hls::stream

#ifndef BRIDGE_STAGES_H
#define BRIDGE_STAGES_H

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include "../../attn_block_pl/attn_block_types.h"

typedef ap_axiu<64, 0, 0, 0> pkt64_t;

// pack 4 x int16 from a local buffer into a 64 bit AXI beats

template<int COUNT>
static void pack_buf_to_axi(const data_t buf[COUNT], hls::stream<pkt64_t>& out)
{
    #pragma HLS INLINE off
    PACK: for (int i = 0; i < COUNT; i+= 4) {
        #pragma HLS PIPELINE II=1
        pkt64_t pkt;
        ap_uint<64> word = 0;
        for (int j = 0; j < 4; j++) {
            if (i + j < COUNT) {
                ap_uint<16> bits = buf[i + j].range(15, 0);
                word.range(j * 16 + 15, j * 16) = bits;
            }
        }
        pkt.data = word;
        pkt.keep = -1;
        pkt.last = ( i + 4 >= COUNT) ? 1 : 0;
        out.write(pkt);
    }
}

// pack by reading from an hls::stream
template <int COUNT>
static void pack_stream_to_axi(hls::stream<data_t>& in, hls::stream<pkt64_t>& out)
{
    #pragma HLS INLINE off
    data_t buf[COUNT];
    READ: for (int i = 0; i < COUNT; i++) {
        #pragma HLS PIPELINE II=1
        buf[i] = in.read();
    }
    pack_buf_to_axi<COUNT>(buf, out);
}

//unpack 64 bit axi beats into a local buffer
template<int COUNT>
static void unpack_axi_to_buf(hls::stream<pkt64_t>& in, data_t buf[COUNT])
{
    #pragma HLS PIPELINE II=1
    UNPACK: for (int i = 0; i < COUNT; i += 4) {
        pkt64_t pkt = in.read();
        ap_uint<64> word = pkt.data;
        for (int j = 0; j < 4; j++) {
            if (i + j < COUNT) {
                ap_uint<16> bits = word.range(j * 16 + 15, j * 16);
                data_t val;
                val.range(15, 0) = bits;
                buf[i + j] = val;
            }
        }
    }
}

// unpack from aie and write to pl hls::stream
template<int COUNT>
static void unpack_axi_to_stream(hls::stream<pkt64_t>&in, hls::stream<data_t>&out)
{
    #pragma HLS INLINE off
    data_t buf[COUNT];
    unpack_axi_to_buf<COUNT>(in, buf);
    WRITE: for (int i = 0; i < COUNT; i++) {
        #pragma HLS PIPELINE II=1
        out.write(buf[i]);
    }
}


static const int RAW_DIM = 5;

inline void read_and_fork_aie   (
    hls::stream<ap_uint<32>> &in_s,

    // raw jets go to 3 consumers (embed, pairwise, cand_lorentz)
    hls::stream<data_t> &out_jets_embed,
    hls::stream<data_t> &out_jets_pairwise,
    hls::stream<data_t> &out_jets_cand,

    // mask -> 6 consumers 
    hls::stream<bool> &out_mask_embed,
    hls::stream<bool> &out_mask_cb0,
    hls::stream<bool> &out_mask_cb1,
    hls::stream<bool> &out_mask_cand,
    hls::stream<bool> &out_mask_remask0,
    hls::stream<bool> &out_mask_remask1
) {
    // read raw_jets from axi-stream
    data_t raw_jets[N_MAX][RAW_DIM];
    READ_JETS: for (int i = 0; i < N_MAX; i++) {
        for (int j = 0; j < RAW_DIM; j++) {
            #pragma HLS PIPELINE II=1
            //reinterpret 32-bit unsigned as data_t
            ap_uint<32> bits = in_s.read();
            data_t val;
            val.range(15, 0) =  bits.range(15, 0);
            raw_jets[i][j] = val;
        }
    }

    // read mask from axi-stream
    bool mask[N_MAX];
    READ_MASK: for (int i = 0; i < N_MAX; i++) {
        #pragma HLS PIPELINE II=1
        ap_uint<32> w = in_s.read();
        mask[i] = (w != 0);
    }

    // fork raw jets into 3 output streams

    FORK_JETS: for (int i = 0; i < N_MAX;i++) {
        for (int j = 0 ; j<RAW_DIM; j++) {
            #pragma HLS PIPELINE II=1
            data_t val = raw_jets[i][j];
            out_jets_embed.write(val);
            out_jets_pairwise.write(val);
            out_jets_cand.write(val);
        }
    }

    // fork mas into 4 output streams

    FORK_MASK: for (int i = 0; i < N_MAX;i++) {
        #pragma HLS PIPELINE II=1
        bool val = mask[i];
        out_mask_embed.write(val);
        out_mask_cb0.write(val);
        out_mask_cb1.write(val);
        out_mask_cand.write(val);
        out_mask_remask0.write(val);
        out_mask_remask1.write(val);
    }
}

// layer 1 obj_attn doesn't use a wij bias (only layer 0 does),
// but the bridge signature still requires a wij to consume
// this stage feeds N_HEADS * N_MAX * N_KV = 624 zero score_t values
// that the AIE kernel will receive and then be ignored by the
// layer-1 kernel
inline void emit_zero_wij(hls::stream<score_t>& out) {
    EMIT_ZERO: for (int i = 0; i <N_HEADS * N_MAX * N_KV; i++) {
        #pragma HLS PIPELINE II=1
        score_t zero = 0;
        out.write(zero);
    }
}






// object attention bridge

// consumes x[12][16] from PL, Wij[4*12][13] from PL
// produces x[12][16] back to PL after AIE obj_attn
inline void obj_attn_bridge(
    hls::stream<data_t>& x_in_pl,
    hls::stream<score_t>& wij_in_pl,
    // to AIE
    hls::stream<pkt64_t>& x_out_aie,
    hls::stream<pkt64_t>& wij_h0_out_aie,
    hls::stream<pkt64_t>& wij_h1_out_aie,
    hls::stream<pkt64_t>& wij_h2_out_aie,
    hls::stream<pkt64_t>& wij_h3_out_aie,
    // from AIE
    hls::stream<pkt64_t>& x_in_aie,
    // to next PL stage
    hls::stream<data_t>& x_out_pl
) {
    const int X_SZ = N_MAX * E_DIM; //192
    const int WIJ_SZ = N_MAX * N_KV; // 156 per head

    // read x from pl
    data_t x_buf[X_SZ];
    for (int i = 0; i < X_SZ; i++) {
        #pragma HLS PIPELINE II=1
        x_buf[i] = x_in_pl.read();
    }

    // read wij[4*12][13] from PL
    data_t wij_buf[N_HEADS * WIJ_SZ];
    for (int i = 0; i < N_HEADS * WIJ_SZ; i++) {
        #pragma HLS PIPELINE II=1
        score_t s = wij_in_pl.read();
        data_t v;
        v.range(15, 0) = s.range(15, 0);
        wij_buf[i] = v;
    }

    // send x to aie
    pack_buf_to_axi<X_SZ>(x_buf, x_out_aie);

    // send per head wij slices
    data_t slice[WIJ_SZ];

    for (int i = 0; i < WIJ_SZ; i++) slice[i] = wij_buf[0 * WIJ_SZ + i];
    pack_buf_to_axi<WIJ_SZ>(slice, wij_h0_out_aie);

    for (int i = 0; i < WIJ_SZ; i++) slice[i] = wij_buf[1 * WIJ_SZ + i];
    pack_buf_to_axi<WIJ_SZ>(slice, wij_h1_out_aie);

    for (int i = 0; i < WIJ_SZ; i++) slice[i] = wij_buf[2 * WIJ_SZ + i];
    pack_buf_to_axi<WIJ_SZ>(slice, wij_h2_out_aie);

    for (int i = 0; i < WIJ_SZ; i++) slice[i] = wij_buf[3 * WIJ_SZ + i];
    pack_buf_to_axi<WIJ_SZ>(slice, wij_h3_out_aie);

    // receive result from aie, forward to next PL stage
    unpack_axi_to_stream<X_SZ>(x_in_aie, x_out_pl);
}

// candidate attention bridge
// consumes c[3][16] from PL
// produces c[3][16] back to PL after AIE cand_attn

inline void cand_attn_bridge(
    hls::stream<data_t>& c_in_pl,
    hls::stream<pkt64_t>& c_out_aie,
    hls::stream<pkt64_t>& c_in_aie,
    hls::stream<data_t>& c_out_pl
) {
    const int C_SZ = T_DIM * E_DIM; // 48

    // read c from pl
    data_t c_buf[C_SZ];
    for (int i = 0; i < C_SZ; i++) {
        #pragma HLS PIPELINE II=1
        c_buf[i] = c_in_pl.read();
    }
    // send c to aie
    pack_buf_to_axi<C_SZ>(c_buf, c_out_aie);

    // receive result from aie, forward to next PL stage
    unpack_axi_to_stream<C_SZ>(c_in_aie, c_out_pl);
}

// cross attention bridge
// consumes x[12][16] and c[3][16] from PL
// produces x[12][16] back to PL after AIE cross_attn

inline void cross_attn_bridge(
    hls::stream<data_t>& x_in_pl,
    hls::stream<data_t>& c_in_pl,
    hls::stream<pkt64_t>& x_out_aie,
    hls::stream<pkt64_t>& c_out_aie,
    hls::stream<pkt64_t>& x_in_aie,
    hls::stream<data_t>& x_out_pl
) {
    const int X_SZ = N_MAX * E_DIM; // 192
    const int C_SZ = T_DIM * E_DIM; // 48

    // read x from pl
    data_t x_buf[X_SZ];
    for (int i = 0; i < X_SZ; i++) {
        #pragma HLS PIPELINE II=1
        x_buf[i] = x_in_pl.read();
    }

    // read c from pl
    data_t c_buf[C_SZ];
    for (int i = 0; i < C_SZ; i++) {
        #pragma HLS PIPELINE II=1
        c_buf[i] = c_in_pl.read();
    }

    pack_buf_to_axi<X_SZ>(x_buf, x_out_aie);
    pack_buf_to_axi<C_SZ>(c_buf, c_out_aie);

    // receive result from aie, forward to next PL stage
    unpack_axi_to_stream<X_SZ>(x_in_aie, x_out_pl);
}

#endif