// The same measurement as plhost_timed, with a second timing loop that starts
// and polls the CU by DIRECT REGISTER ACCESS (xrt::kernel::write_register /
// read_register) instead of xrt::run. That takes the XRT ERT scheduler out of
// the launch path, so the difference between the two loops is what the
// scheduler costs in the one-event latency. Both loops run in one process on
// the same buffers, so nothing else differs.
// Register map from the kernel.xml of the build (HLS layout, ap_ctrl_chain):
//   0x00 control (bit0 ap_start, bit1 ap_done, bit3 ap_ready, bit4 ap_continue)
//   0x10/0x14 in_buf lo/hi   0x1C/0x20 out_buf lo/hi   0x28 n_events
// usage: ./plhost_direct pl_stream.xclbin input.bin <n_events> [iters]
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <stdexcept>
#include "xrt/xrt_device.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_kernel.h"
static float b2f(uint32_t b){ float f; std::memcpy(&f,&b,4); return f; }
using clk = std::chrono::high_resolution_clock;
static void stats(const char* tag, std::vector<double> ms, int N){
  if(ms.empty()){ printf("%s: no samples\n",tag); return; }
  std::sort(ms.begin(),ms.end()); double sum=0; for(double x:ms) sum+=x;
  printf("%-28s min=%.4f ms  median=%.4f  avg=%.4f  max=%.4f   (%zu iters, N=%d)\n",
         tag, ms.front(), ms[ms.size()/2], sum/ms.size(), ms.back(), ms.size(), N);
}
int main(int argc,char**argv){
  setbuf(stdout,NULL);
  const int WIN=72, WOUT=3;
  int N = (argc>3)?atoi(argv[3]):1;
  int ITERS = (argc>4)?atoi(argv[4]):50;
  std::vector<uint32_t> in(N*WIN,0);
  if(argc>2){ std::ifstream f(argv[2],std::ios::binary); f.read((char*)in.data(),N*WIN*4); }
  auto dev=xrt::device(0);
  printf("load_xclbin...\n"); auto uuid=dev.load_xclbin(std::string(argv[1]));
  auto k=xrt::kernel(dev,uuid,"pl_stream_top",xrt::kernel::cu_access_mode::exclusive);
  auto in_bo=xrt::bo(dev,N*WIN*4,k.group_id(0));
  auto out_bo=xrt::bo(dev,N*WOUT*4,k.group_id(1));
  auto im=in_bo.map<uint32_t*>(); auto om=out_bo.map<uint32_t*>();
  for(int i=0;i<N*WIN;i++) im[i]=in[i];
  for(int i=0;i<N*WOUT;i++) om[i]=0xDEAD0000u|i;
  in_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE); out_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // --- A: the ERT path, exactly as plhost_timed ---------------------------
  auto r=xrt::run(k); r.set_arg(0,in_bo); r.set_arg(1,out_bo); r.set_arg(2,(uint32_t)N);
  r.start(); auto wst=r.wait(15000);
  if((int)wst!=4){ printf("ABORT: warmup did not complete (state %d)\n",(int)wst); return 2; }
  std::vector<double> ert; ert.reserve(ITERS);
  for(int it=0;it<ITERS;it++){
    auto t0=clk::now(); r.start(); auto st=r.wait(15000); auto t1=clk::now();
    if((int)st!=4){ printf("ERT iter %d FAILED state=%d\n",it,(int)st); break; }
    ert.push_back(std::chrono::duration<double,std::milli>(t1-t0).count());
  }
  out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  float ref0=b2f(om[0]);
  for(int i=0;i<N*WOUT;i++) om[i]=0xDEAD0000u|i;
  out_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // --- B: direct register control, no scheduler ---------------------------
  const uint32_t CTRL=0x00, IN_LO=0x10, IN_HI=0x14, OUT_LO=0x1C, OUT_HI=0x20, NEV=0x28;
  uint64_t ia=in_bo.address(), oa=out_bo.address();
  std::vector<double> dir; dir.reserve(ITERS);
  bool ok=true;
  try {
    k.write_register(IN_LO,(uint32_t)ia);  k.write_register(IN_HI,(uint32_t)(ia>>32));
    k.write_register(OUT_LO,(uint32_t)oa); k.write_register(OUT_HI,(uint32_t)(oa>>32));
    k.write_register(NEV,(uint32_t)N);
    printf("direct: args written (ctrl=0x%x)\n", k.read_register(CTRL));
  } catch (const std::exception& e) {
    printf("direct register access REFUSED by XRT: %s\n", e.what()); ok=false;
  }
  for(int it=0; ok && it<ITERS; it++){
    auto t0=clk::now();
    k.write_register(CTRL,0x1);                       // ap_start
    uint32_t c=0; long spins=0;
    do { c=k.read_register(CTRL); if(++spins>100000000L){ ok=false; break; } } while(!(c&0x2));
    auto t1=clk::now();
    if(!ok){ printf("direct iter %d TIMEOUT (ctrl=0x%x)\n",it,c); break; }
    k.write_register(CTRL,0x10);                      // ap_continue (ap_ctrl_chain)
    dir.push_back(std::chrono::duration<double,std::milli>(t1-t0).count());
  }
  out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  int chg=0; for(int i=0;i<N*WOUT;i++) if(om[i]!=(0xDEAD0000u|(uint32_t)i)) chg++;
  printf("direct: output words changed from sentinel: %d / %d, ev0 %.5f (ERT path gave %.5f)\n",
         chg, N*WOUT, b2f(om[0]), ref0);

  printf("\n=== one invocation, N=%d events ===\n",N);
  stats("ERT (xrt::run)", ert, N);
  stats("direct registers", dir, N);
  if(!ert.empty() && !dir.empty()){
    std::vector<double> a=ert,b=dir; std::sort(a.begin(),a.end()); std::sort(b.begin(),b.end());
    printf("scheduler cost removed:      %.1f us (median), %.1f us (min)\n",
           1000*(a[a.size()/2]-b[b.size()/2]), 1000*(a.front()-b.front()));
  }
  return 0;
}
