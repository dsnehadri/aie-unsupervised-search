// One-tile graph around the jet embedding MLP, used to validate the kernel
// against the PyTorch golden (x86sim) and to measure its cycles (aiesim).
#ifndef EMBED_GRAPH_H
#define EMBED_GRAPH_H

#include <adf.h>
#include "../kernels/embed_kernel.h"

using namespace adf;

class EmbedGraph : public graph {
public:
    input_plio  plio_jets_in;
    output_plio plio_embed_out;

private:
    kernel k_embed;

public:
    EmbedGraph() {
        plio_jets_in   = input_plio::create("embed_jets_in", plio_64_bits, "data/embed_jets_in.txt");
        plio_embed_out = output_plio::create("embed_x_out", plio_64_bits, "data/embed_x_out.txt");

        k_embed = kernel::create(embed_mlp);
        source(k_embed) = "kernels/embed_kernel.cc";
        runtime<ratio>(k_embed) = 0.9;

        constexpr int in_bytes  = EMBED_ROWS * EMBED_IN * sizeof(int16);   // 120
        constexpr int out_bytes = EMBED_ROWS * E_DIM * sizeof(int16);      // 384
        connect<window<in_bytes>>(plio_jets_in.out[0], k_embed.in[0]);
        connect<window<out_bytes>>(k_embed.out[0], plio_embed_out.in[0]);
    }
};

#endif // EMBED_GRAPH_H
