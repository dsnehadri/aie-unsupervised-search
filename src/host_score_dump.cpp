// host_score_dump -- run N events through a deployed design and dump the three
// per-event loss words to a binary file, for an AUC comparison against software.
// Both designs take (in_buf, out_buf, n_events), so one binary serves each.
// usage: ./host_score_dump <xclbin> <kernel> <input.bin> <n_events> <out.bin> [has_graph]
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <fstream>
#include <vector>
#include "xrt/xrt_device.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_graph.h"
static float b2f(uint32_t b){ float f; std::memcpy(&f,&b,4); return f; }
int main(int argc, char** argv) {
  setbuf(stdout, NULL);
  if (argc < 6) { printf("usage: %s <xclbin> <kernel> <in.bin> <n> <out.bin> [has_graph]\n", argv[0]); return 1; }
  const std::string xclbin = argv[1], kname = argv[2], inbin = argv[3], outbin = argv[5];
  const int N = atoi(argv[4]);
  const bool has_graph = (argc > 6) ? atoi(argv[6]) != 0 : (kname.rfind("aie", 0) == 0);
  const int WIN = 72, WOUT = 3;

  std::vector<uint32_t> in((size_t)N * WIN, 0);
  { std::ifstream f(inbin, std::ios::binary);
    f.read((char*)in.data(), (size_t)N * WIN * 4);
    printf("read %zd bytes of input\n", (ssize_t)f.gcount()); }

  auto dev = xrt::device(0);
  auto uuid = dev.load_xclbin(xclbin);
  xrt::graph* g = nullptr;
  if (has_graph) { g = new xrt::graph(dev, uuid, "aie_graph"); g->reset(); g->run(N + 1); }
  auto k = xrt::kernel(dev, uuid, kname, xrt::kernel::cu_access_mode::exclusive);
  auto ib = xrt::bo(dev, (size_t)N * WIN * 4, k.group_id(0));
  auto ob = xrt::bo(dev, (size_t)N * WOUT * 4, k.group_id(1));
  auto im = ib.map<uint32_t*>(); auto om = ob.map<uint32_t*>();
  for (size_t i = 0; i < (size_t)N * WIN; i++) im[i] = in[i];
  for (size_t i = 0; i < (size_t)N * WOUT; i++) om[i] = 0xDEAD0000u | (uint32_t)i;
  ib.sync(XCL_BO_SYNC_BO_TO_DEVICE); ob.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto r = xrt::run(k);
  r.set_arg(0, ib); r.set_arg(1, ob); r.set_arg(2, (uint32_t)N);
  r.start();
  auto st = r.wait(600000);
  printf("wait state=%d (4=COMPLETED)\n", (int)st);
  ob.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  int chg = 0;
  for (int i = 0; i < N * WOUT; i++) if (om[i] != (0xDEAD0000u | (uint32_t)i)) chg++;
  printf("output words changed: %d / %d\n", chg, N * WOUT);
  std::ofstream of(outbin, std::ios::binary);
  of.write((const char*)om, (size_t)N * WOUT * 4);
  printf("first 3 events: ");
  for (int e = 0; e < 3 && e < N; e++) printf("%.5f ", b2f(om[e * WOUT]));
  printf("\nwrote %s\nSCORE_DUMP_DONE\n", outbin.c_str());
  return 0;   // skip graph.end(), it hangs on this board
}
