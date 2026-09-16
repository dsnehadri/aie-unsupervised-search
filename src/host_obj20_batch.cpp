// host_obj20_batch -- tile-scaling sweep on the batched obj24_top (v5).
// For each instance count: time one call at several batch sizes and fit
// t(n_ev) = L + n_ev * s. s = time per event round across all active instances,
// so aggregate throughput = n_inst / s. The array graph runs continuously.
// usage: ./host_obj20_batch <xclbin> [iters] [n_inst list, e.g. 1,2,4,5,8,16,20]
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include "xrt/xrt_device.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_graph.h"
using clk=std::chrono::high_resolution_clock;
static const int IN_PER=976, OUT=192, NINST=20;
int main(int argc,char**argv){
  setbuf(stdout,NULL);
  std::string xclbin=argv[1];
  int ITERS=(argc>2)?atoi(argv[2]):30;
  int NLIST[16]={1,2,4,5,8,16,20}; int NPTS=7;
  if(argc>3){ NPTS=0; char* tok=strtok(argv[3],","); while(tok&&NPTS<16){ NLIST[NPTS++]=atoi(tok); tok=strtok(NULL,","); } }
  const int ELIST[]={8,16,32,64,128,256}; const int NE=sizeof(ELIST)/sizeof(ELIST[0]);
  auto dev=xrt::device(0); auto uuid=dev.load_xclbin(xclbin);
  auto g=xrt::graph(dev,uuid,"aie_graph"); g.reset(); g.run(-1);
  auto k=xrt::kernel(dev,uuid,"obj24_top",xrt::kernel::cu_access_mode::exclusive);
  auto ib=xrt::bo(dev,(size_t)IN_PER*4,k.group_id(0));
  auto ob=xrt::bo(dev,(size_t)NINST*OUT*4,k.group_id(1));
  auto im=ib.map<uint32_t*>(); auto om=ob.map<uint32_t*>();
  // small in-range int16 pattern; the block's timing does not depend on the values
  for(int i=0;i<IN_PER;i++) im[i]=(uint32_t)(uint16_t)(int16_t)((i*37)%200-100);
  ib.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto r=xrt::run(k); r.set_arg(0,ib); r.set_arg(1,ob);
  printf("n_inst,tiles,n_ev,iters,min_ms,med_ms,out_changed\n");
  for(int p=0;p<NPTS;p++){
    int n=NLIST[p]; r.set_arg(2,(uint32_t)n);
    std::vector<double> xs, ys;
    for(int q=0;q<NE;q++){
      int e=ELIST[q]; r.set_arg(3,(uint32_t)e);
      for(int i=0;i<NINST*OUT;i++) om[i]=0xDEAD0000u|(uint32_t)i;
      ob.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      r.start(); if((int)r.wait(60000)!=4){ printf("%d,%d,%d,WARMUP_FAIL\n",n,15*n,e); return 1; }
      std::vector<double> ms;
      for(int it=0;it<ITERS;it++){ auto t0=clk::now(); r.start(); auto st=r.wait(60000); auto t1=clk::now();
        if((int)st!=4){ printf("%d,%d,%d,ITER_FAIL\n",n,15*n,e); return 1; }
        ms.push_back(std::chrono::duration<double,std::milli>(t1-t0).count()); }
      ob.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      int chg=0; for(int i=0;i<n*OUT;i++) if(om[i]!=(0xDEAD0000u|(uint32_t)i)) chg++;
      std::sort(ms.begin(),ms.end());
      printf("%d,%d,%d,%d,%.5f,%.5f,%d/%d\n",n,15*n,e,(int)ms.size(),ms.front(),ms[ms.size()/2],chg,n*OUT);
      xs.push_back(e); ys.push_back(ms.front());
    }
    double mx=0,my=0; for(size_t i=0;i<xs.size();i++){mx+=xs[i];my+=ys[i];} mx/=xs.size(); my/=ys.size();
    double sxy=0,sxx=0; for(size_t i=0;i<xs.size();i++){sxy+=(xs[i]-mx)*(ys[i]-my); sxx+=(xs[i]-mx)*(xs[i]-mx);}
    double s=sxy/sxx;   // ms per event round
    printf("FIT,n_inst=%d,tiles=%d,us_per_round=%.3f,agg_ev_s=%.0f,per_inst_ev_s=%.0f\n",n,15*n,s*1000,n*1000.0/s,1000.0/s);
  }
  printf("SWEEP_DONE\n");
  return 0;   // no graph.end(): hangs on this board
}
