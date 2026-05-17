#include <ap_int.h>

extern "C" void stub_top(
    ap_uint<32>* in_buf,
    ap_uint<32>* out_buf,
    int n_events
) {
#pragma HLS INTERFACE m_axi port=in_buf  offset=slave bundle=gmem0 depth=720
#pragma HLS INTERFACE m_axi port=out_buf offset=slave bundle=gmem1 depth=30
#pragma HLS INTERFACE s_axilite port=in_buf
#pragma HLS INTERFACE s_axilite port=out_buf
#pragma HLS INTERFACE s_axilite port=n_events
#pragma HLS INTERFACE s_axilite port=return

    out_buf[0] = 0xDEADBEEF;
    out_buf[1] = 0xCAFEBABE;
    out_buf[2] = 0x12345678;
}