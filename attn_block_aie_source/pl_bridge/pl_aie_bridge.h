// hls kernel that sits on the pl side and bridges data
// between existing pl pipeline and aie attention tiles

// 1. receives embedded X[N_MAX][E_DIM] from PL embed stage
// 2. receives Wij[N_HEADS][N_MAX][N_KV] from PL pairwise mlp
// 3. streams X to all 4 AIE head tiles (broadcast)
// 4. streams per-head Wij slices to each head tile
// 5. receives attention output from AIE
// 6. forwards to next PL stage (candidate building)

// AXI-stream input: from PL embed + pairwise MLP
// AXI-stream output: to AIE PLIO
// AXI-stream input: from AIE PLIO (attention result)
// AXI-stream output: to PL candidate building

#ifndef PL_AIE_BRIDGE_H
#define PL_AIE_BRIDGE_H

#include "ap_fixed.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"

// match the PL data types from your existing pipeline
typedef ap_fixed<16, 5> data_t;

// AXI-stream packet type (64-bit for PLIO matching)
typedef ap_axiu<64, 0, 0, 0> pkt64_t;

//top level function
extern "C" void pl_aie_bridge(
    // input from PL embed stage
    hls::stream<data_t>& embed_in,

    // input from PL pairwise MLP (wij bias)
    hls::stream<data_t>& wij_in,

    // output to aie: X data (broadcast to all heads)
    hls::stream<pkt64_t>& obj_x_out,

    //output to aie: per head wij
    hls::stream<pkt64_t>& wij_h0_out,
    hls::stream<pkt64_t>& wij_h1_out,
    hls::stream<pkt64_t>& wij_h2_out,
    hls::stream<pkt64_t>& wij_h3_out,

    // input from aie: attention result
    hls::stream<pkt64_t>& obj_x_in,

    // output to PL next stage  
    hls::stream<data_t>& attn_out
);



#endif