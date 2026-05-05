#include <ap_int.h>

extern "C" void stub_ctrl(int n_events) {
#pragma HLS INTERFACE s_axilite port=n_events
#pragma HLS INTERFACE s_axilite port=return

    // do nothing; return should immediately assert ap_done
    return;
}