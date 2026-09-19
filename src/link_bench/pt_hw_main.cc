// v3 link-benchmark graph: pushing the AIE<->PL link to its configuration limits.
//   w128     : one 128-bit PLIO pair @250MHz (levers 1+2: width x clock)
//   q0..q3   : four 128-bit PLIO pairs (lever 3: channel aggregation)
// All window-based (tile-DMA transport, the pattern proven on this platform);
// v8int32 vectorized copy so the core is never the bottleneck.
#include <adf.h>
using namespace adf;

void passthru_w128(input_window<int32>* in, output_window<int32>* out);

class PtGraph : public graph {
public:
    kernel kw, kq[4];
    input_plio  pwin,  qin[4];
    output_plio pwout, qout[4];
    PtGraph() {
        kw = kernel::create(passthru_w128);
        source(kw) = "pt_kernel.cc";
        runtime<ratio>(kw) = 0.9;
        pwin  = input_plio::create("p128_in",  plio_128_bits, "data/pt_in.txt", 250);
        pwout = output_plio::create("p128_out", plio_128_bits, "data/pt_out.txt", 250);
        connect<window<1024>>(pwin.out[0], kw.in[0]);
        connect<window<1024>>(kw.out[0], pwout.in[0]);
        for (int i = 0; i < 4; i++) {
            kq[i] = kernel::create(passthru_w128);
            source(kq[i]) = "pt_kernel.cc";
            runtime<ratio>(kq[i]) = 0.9;
            std::string s = std::to_string(i);
            qin[i]  = input_plio::create("q_in_"  + s, plio_128_bits, "data/pt_in.txt", 250);
            qout[i] = output_plio::create("q_out_" + s, plio_128_bits, "data/pt_out.txt", 250);
            connect<window<1024>>(qin[i].out[0], kq[i].in[0]);
            connect<window<1024>>(kq[i].out[0], qout[i].in[0]);
        }
    }
};

PtGraph aie_graph;
int main(void) { aie_graph.init(); aie_graph.run(1); aie_graph.end(); return 0; }
