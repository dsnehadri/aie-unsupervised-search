// instrumented PL-stream host: measures kernel latency + throughput.
// usage: ./plhost_timed pl_stream.xclbin input.bin <n_events> [iters]
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include "xrt/xrt_device.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_kernel.h"
static float b2f(uint32_t b){ float f; std::memcpy(&f,&b,4); return f; }
using clk = std::chrono::high_resolution_clock;
int main(int argc,char**argv){
  setbuf(stdout,NULL);
  const int WIN=72, WOUT=3;
  int N = (argc>3)?atoi(argv[3]):10;
  int ITERS = (argc>4)?atoi(argv[4]):20;
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
  printf("running pl_stream_top, %d events, %d timed iters...\n",N,ITERS);
  auto r=xrt::run(k); r.set_arg(0,in_bo); r.set_arg(1,out_bo); r.set_arg(2,(uint32_t)N);
  // warmup
  r.start(); auto wst=r.wait(15000);
  printf("warmup wait state=%d (4=COMPLETED,8=TIMEOUT)\n",(int)wst);
  if((int)wst!=4){ printf("ABORT: warmup did not complete\n"); return 2; }
  std::vector<double> ms; ms.reserve(ITERS);
  for(int it=0;it<ITERS;it++){
    auto t0=clk::now();
    r.start(); auto st=r.wait(15000);
    auto t1=clk::now();
    if((int)st!=4){ printf("iter %d FAILED state=%d\n",it,(int)st); break; }
    ms.push_back(std::chrono::duration<double,std::milli>(t1-t0).count());
  }
  out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  int chg=0; for(int i=0;i<N*WOUT;i++) if(om[i]!=(0xDEAD0000u|(uint32_t)i)) chg++;
  printf("output words changed from sentinel: %d / %d\n", chg, N*WOUT);
  if(!ms.empty()){
    std::sort(ms.begin(),ms.end());
    double sum=0; for(double x:ms) sum+=x; double avg=sum/ms.size();
    double mn=ms.front(), mx=ms.back(), med=ms[ms.size()/2];
    printf("\n=== TIMING (CU start->done, ert_polling) over %zu iters, N=%d events ===\n",ms.size(),N);
    printf("invocation latency ms:  min=%.4f  median=%.4f  avg=%.4f  max=%.4f\n",mn,med,avg,mx);
    printf("per-event latency:      min=%.5f ms  avg=%.5f ms\n",mn/N,avg/N);
    printf("throughput:             best=%.1f ev/s  avg=%.1f ev/s\n",N/(mn/1000.0),N/(avg/1000.0));
  }
  printf("\nfirst events output:\n");
  for(int e=0;e<std::min(N,5);e++) printf("  ev%2d: %.5f %.5f %.5f\n",e,b2f(om[e*3]),b2f(om[e*3+1]),b2f(om[e*3+2]));
  return 0;
}
