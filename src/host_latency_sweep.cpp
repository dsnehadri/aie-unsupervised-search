// host_latency_sweep -- separate pipeline latency from per-event interval.
//
// Times batches of N events and fits t(N) = L + N/T:
//   slope 1/T  = steady-state per-event interval (the reciprocal of throughput)
//   intercept L = pipeline fill/drain latency PLUS the host launch overhead
// Both deployed designs (pl_stream_top, aie_stream_top) take the same
// (in_buf, out_buf, n_events) argument prefix, so one binary covers both.
//
// usage: ./host_latency_sweep <xclbin> <kernel_name> [input.bin] [iters]
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include "xrt/xrt_device.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_kernel.h"
using clk = std::chrono::high_resolution_clock;

int main(int argc, char** argv) {
  setbuf(stdout, NULL);
  if (argc < 3) { printf("usage: %s <xclbin> <kernel> [input.bin] [iters]\n", argv[0]); return 1; }
  const std::string xclbin = argv[1], kname = argv[2];
  const char* inbin = (argc > 3) ? argv[3] : nullptr;
  const int ITERS = (argc > 4) ? atoi(argv[4]) : 30;
  const int WIN = 72, WOUT = 3;
  const int NLIST[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
  const int NPTS = sizeof(NLIST) / sizeof(NLIST[0]);
  const int NMAX = NLIST[NPTS - 1];

  std::vector<uint32_t> in((size_t)NMAX * WIN, 0);
  if (inbin) {                       // reuse one event's payload for every slot
    std::ifstream f(inbin, std::ios::binary);
    std::vector<uint32_t> one(WIN, 0);
    f.read((char*)one.data(), WIN * 4);
    for (int e = 0; e < NMAX; e++)
      for (int w = 0; w < WIN; w++) in[(size_t)e * WIN + w] = one[w];
  }

  auto dev = xrt::device(0);
  auto uuid = dev.load_xclbin(xclbin);
  auto k = xrt::kernel(dev, uuid, kname, xrt::kernel::cu_access_mode::exclusive);
  auto ib = xrt::bo(dev, (size_t)NMAX * WIN * 4, k.group_id(0));
  auto ob = xrt::bo(dev, (size_t)NMAX * WOUT * 4, k.group_id(1));
  auto im = ib.map<uint32_t*>(); auto om = ob.map<uint32_t*>();
  for (size_t i = 0; i < (size_t)NMAX * WIN; i++) im[i] = in[i];
  ib.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto r = xrt::run(k); r.set_arg(0, ib); r.set_arg(1, ob);
  printf("kernel,n_events,iters,min_ms,med_ms,us_per_event,out_changed\n");
  for (int p = 0; p < NPTS; p++) {
    int N = NLIST[p];
    for (int i = 0; i < N * WOUT; i++) om[i] = 0xDEAD0000u | (uint32_t)i;
    ob.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    r.set_arg(2, (uint32_t)N);
    r.start();
    if ((int)r.wait(60000) != 4) { printf("%s,%d,WARMUP_FAIL\n", kname.c_str(), N); continue; }
    std::vector<double> ms;
    for (int it = 0; it < ITERS; it++) {
      auto t0 = clk::now(); r.start(); auto st = r.wait(60000); auto t1 = clk::now();
      if ((int)st != 4) { printf("%s,%d,ITER_FAIL\n", kname.c_str(), N); break; }
      ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    if (ms.empty()) continue;
    ob.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    int chg = 0;
    for (int i = 0; i < N * WOUT; i++) if (om[i] != (0xDEAD0000u | (uint32_t)i)) chg++;
    std::sort(ms.begin(), ms.end());
    double mn = ms.front(), md = ms[ms.size() / 2];
    printf("%s,%d,%d,%.5f,%.5f,%.3f,%d/%d\n", kname.c_str(), N, (int)ms.size(),
           mn, md, mn * 1000.0 / N, chg, N * WOUT);
  }
  printf("SWEEP_DONE\n");
  return 0;
}
