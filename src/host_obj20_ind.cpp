// host_obj20_ind -- tile-scaling sweep on obj24_top with independent feeders (v6).
// Each instance's drain counts clock cycles between its first and last output
// beat, so every instance's own time per event is measured while all active
// instances run at once. Aggregate throughput = sum over instances of 1/time.
// usage: ./host_obj20_ind <xclbin> <n_ev> <clock period ns> [runs] [n_inst list]
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
static const int IN_PER=976, OUT_PER=196, NINST=20;
int main(int argc,char**argv){
  setbuf(stdout,NULL);
  if(argc<4){ printf("usage: %s <xclbin> <n_ev> <period_ns> [runs] [n_inst list]\n",argv[0]); return 1; }
  std::string xclbin=argv[1]; int NEV=atoi(argv[2]); double PER_NS=atof(argv[3]);
  int RUNS=(argc>4)?atoi(argv[4]):5;
  int NLIST[32]={1,2,3,4,5,8,12,16,20}; int NPTS=9;
  if(argc>5){ NPTS=0; char* tok=strtok(argv[5],","); while(tok&&NPTS<32){ NLIST[NPTS++]=atoi(tok); tok=strtok(NULL,","); } }
  auto dev=xrt::device(0); auto uuid=dev.load_xclbin(xclbin);
  auto g=xrt::graph(dev,uuid,"aie_graph"); g.reset(); g.run(-1);
  auto k=xrt::kernel(dev,uuid,"obj24_top",xrt::kernel::cu_access_mode::exclusive);
  auto ib=xrt::bo(dev,(size_t)IN_PER*4,k.group_id(0));
  auto ob=xrt::bo(dev,(size_t)NINST*OUT_PER*4,k.group_id(1));
  auto im=ib.map<uint32_t*>(); auto om=ob.map<uint32_t*>();
  for(int i=0;i<IN_PER;i++) im[i]=(uint32_t)(uint16_t)(int16_t)((i*37)%200-100);   // timing does not depend on values
  ib.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto r=xrt::run(k); r.set_arg(0,ib); r.set_arg(1,ob); r.set_arg(3,(uint32_t)NEV);
  printf("n_inst,tiles,inst,us_per_event,ev_s\n");
  for(int p=0;p<NPTS;p++){
    int n=NLIST[p]; r.set_arg(2,(uint32_t)n);
    std::vector<std::vector<double>> per(n);
    double call_ms_min=1e9;
    for(int run=0; run<RUNS+1; run++){           // run 0 is a warm-up
      for(int i=0;i<NINST*OUT_PER;i++) om[i]=0xDEAD0000u|(uint32_t)i;
      ob.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      auto t0=clk::now(); r.start(); auto st=r.wait(120000); auto t1=clk::now();
      if((int)st!=4){ printf("%d,%d,FAIL state=%d\n",n,15*n,(int)st); return 1; }
      if(run==0) continue;
      call_ms_min=std::min(call_ms_min, std::chrono::duration<double,std::milli>(t1-t0).count());
      ob.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      for(int i=0;i<n;i++){
        const uint32_t* o=om+i*OUT_PER;
        int chg=0; for(int w=0;w<192;w++) if(o[w]!=(0xDEAD0000u|(uint32_t)(i*OUT_PER+w))) chg++;
        uint64_t first=(uint64_t)o[192] | ((uint64_t)o[193]<<32), last=(uint64_t)o[194] | ((uint64_t)o[195]<<32);
        if(chg==0 || last<=first){ printf("%d,%d,%d,BAD_OUTPUT chg=%d first=%llu last=%llu\n",n,15*n,i,chg,(unsigned long long)first,(unsigned long long)last); continue; }
        if (NEV < 2) { fprintf(stderr, "NEV must be >= 2 for interval computation\n"); continue; }
        per[i].push_back((double)(last-first)/(NEV-1)*PER_NS*1e-3);   // us per event
      }
    }
    double agg=0;
    for(int i=0;i<n;i++){
      if(per[i].empty()) continue;
      std::sort(per[i].begin(),per[i].end()); double us=per[i][per[i].size()/2];
      printf("%d,%d,%d,%.3f,%.0f\n",n,15*n,i,us,1e6/us); agg+=1e6/us;
    }
    printf("AGG,n_inst=%d,tiles=%d,agg_ev_s=%.0f,call_min_ms=%.2f,call_ev_s=%.0f\n",n,15*n,agg,call_ms_min,n*NEV*1000.0/call_ms_min);
  }
  printf("SWEEP_DONE\n");
  return 0;   // no graph.end(): hangs on this board
}
