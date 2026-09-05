// host_block_sweep -- batch-size sweep against one isolated AIE attention
// block. Fits t(N) = L + N/T: slope = the block's steady-state per-event
// interval with launch overhead removed. Generic over the block's word counts.
// usage: ./host_block_sweep <xclbin> <kernel> <words_in> <words_out> [iters]
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include "xrt/xrt_device.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_graph.h"
using clk = std::chrono::high_resolution_clock;
int main(int argc, char** argv) {
  setbuf(stdout, NULL);
  if (argc < 5) { printf("usage: %s <xclbin> <kernel> <words_in> <words_out> [iters]\n", argv[0]); return 1; }
  const std::string xclbin = argv[1], kname = argv[2];
  const int WIN = atoi(argv[3]), WOUT = atoi(argv[4]);
  const int ITERS = (argc > 5) ? atoi(argv[5]) : 30;
  const int NLIST[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
  const int NPTS = sizeof(NLIST) / sizeof(NLIST[0]), NMAX = NLIST[NPTS - 1];
  auto dev = xrt::device(0);
  auto uuid = dev.load_xclbin(xclbin);
  auto g = xrt::graph(dev, uuid, "aie_graph"); g.reset(); g.run(-1);
  auto k = xrt::kernel(dev, uuid, kname, xrt::kernel::cu_access_mode::exclusive);
  auto ib = xrt::bo(dev, (size_t)NMAX * WIN * 4, k.group_id(0));
  auto ob = xrt::bo(dev, (size_t)NMAX * WOUT * 4, k.group_id(1));
  auto im = ib.map<uint32_t*>(); auto om = ob.map<uint32_t*>();
  // small in-range int16 pattern; timing is data-independent (masks all valid)
  for (size_t i = 0; i < (size_t)NMAX * WIN; i++) im[i] = (uint32_t)(uint16_t)(int16_t)((i * 37) % 200 - 100);
  ib.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto r = xrt::run(k); r.set_arg(0, ib); r.set_arg(1, ob);
  printf("kernel,n_events,iters,min_ms,med_ms,us_per_event,out_changed\n");
  for (int p = 0; p < NPTS; p++) {
    int N = NLIST[p];
    for (int i = 0; i < N * WOUT; i++) om[i] = 0xDEAD0000u | (uint32_t)i;
    ob.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    r.set_arg(2, (uint32_t)N);
    r.start();
    if ((int)r.wait(60000) != 4) { printf("%s,%d,WARMUP_FAIL\n", kname.c_str(), N); break; }
    std::vector<double> ms;
    for (int it = 0; it < ITERS; it++) {
      auto t0 = clk::now(); r.start(); auto st = r.wait(60000); auto t1 = clk::now();
      if ((int)st != 4) { printf("%s,%d,ITER_FAIL\n", kname.c_str(), N); break; }
      ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    if (ms.empty()) break;
    ob.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    int chg = 0; for (int i = 0; i < N * WOUT; i++) if (om[i] != (0xDEAD0000u | (uint32_t)i)) chg++;
    std::sort(ms.begin(), ms.end());
    printf("%s,%d,%d,%.5f,%.5f,%.3f,%d/%d\n", kname.c_str(), N, (int)ms.size(),
           ms.front(), ms[ms.size()/2], ms.front()*1000.0/N, chg, N*WOUT);
  }
  printf("SWEEP_DONE\n");
  return 0;   // no graph.end(): hangs on this board
}
