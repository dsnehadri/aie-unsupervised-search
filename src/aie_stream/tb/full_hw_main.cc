// Full hybrid AIE graph (PasswdFullGraph: 6 attention subgraphs) for hardware.
// Unguarded main() for the PS partition wrapper. Instance MUST be named
// "aie_graph" so host_aie.cpp's xrt::graph(...,"aie_graph") matches.
#include <adf.h>
#include "../aie/aie_graph.h"
PasswdFullGraph aie_graph;
int main(void){ aie_graph.init(); aie_graph.run(1); aie_graph.end(); return 0; }
