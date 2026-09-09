// scalar mirror of the vector integer layer norm in attn_post_kernel.cc
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstdlib>
typedef int16_t int16; typedef int32_t int32; typedef int64_t int64; typedef uint32_t uint32; typedef uint64_t uint64;
constexpr float PIPE_SCALE = 512.0f;
#include "../../../../../../home/snehadri/repos/aie-unsupervised-search/src/attn_block_aie/kernels/ln_rsqrt_lut.h"
static constexpr int bitlen32_ce(uint32 v){return v?1+bitlen32_ce(v>>1):0;}
static inline int bitlen32(uint32 v){int n=0;if(v>>16){n+=16;v>>=16;}if(v>>8){n+=8;v>>=8;}if(v>>4){n+=4;v>>=4;}if(v>>2){n+=2;v>>=2;}if(v>>1){n+=1;v>>=1;}return n+(int)v;}
// srs: round half to even, then saturate to bits
static inline int64 srs(int64 v, int sh){ if(sh==0) return v; int64 q=v>>sh; int64 rem=v-(q<<sh); int64 half=(int64)1<<(sh-1); if(rem>half||(rem==half&&(q&1))) q++; return q; }
static inline int64 sat(int64 v,int bits){int64 hi=((int64)1<<(bits-1))-1, lo=-((int64)1<<(bits-1)); return v>hi?hi:(v<lo?lo:v);}
int sy_min=99, sy_max=-1, sd_max=-1, e_max=-1;
static void layernorm_row(int16* x,int n_rows,int n_cols,const int16* gamma,const int16* beta){
  constexpr int64 EPS_V=(int64)(1e-5f*16.0f*PIPE_SCALE*PIPE_SCALE+0.5f); constexpr int32 EPS_W4=(int32)(256.0f*1e-5f*16.0f*PIPE_SCALE*PIPE_SCALE+0.5f); constexpr int KD_MIN=-((32-bitlen32_ce((uint32)EPS_W4))/2);
  for(int r=0;r<n_rows;r++){ int16* row=x+r*n_cols; int32 sum=0; for(int c=0;c<16;c++) sum+=row[c];
    int32 d[16], m=0; for(int c=0;c<16;c++){ d[c]=((int32)row[c]<<4)-sum; if(abs(d[c])>m) m=abs(d[c]); }
    int kd=bitlen32(m)-14; if(kd<KD_MIN)kd=KD_MIN; int up=kd<0?-kd:0, down=kd>0?kd:0;
    int16 dn[16]; int32 S=0; for(int c=0;c<16;c++){ dn[c]=(int16)sat(srs((int64)d[c]<<up,down),16); S+=(int32)srs((int64)dn[c]*dn[c],2); }
    int32 W=S+(kd>=0?(EPS_W4>>(2*kd+2)):(EPS_W4<<(2*up-2)));
    int e=(32-bitlen32((uint32)W))&~1; uint32 Wn=(uint32)W<<e; int idx=(int)(Wn>>23)-128; int32 frac=(int32)((Wn>>7)&0xFFFF);
    int32 l0=LN_RSQRT_LUT[idx], l1=LN_RSQRT_LUT[idx+1]; int32 R16=l0+(((l1-l0)*frac)>>16); int32 Rq=(R16+2)>>2; if(Rq>32767)Rq=32767;
    int32 mq=kd>=0?(m>>kd):(m<<up); int sd=bitlen32((uint32)(mq*Rq))-15; if(sd<0)sd=0;
    int16 dn2[16]; for(int c=0;c<16;c++) dn2[c]=(int16)sat(srs((int64)dn[c]*Rq,sd),16);
    int sy=29-sd-(e>>1); if(sy<0)sy=0;
    if(sy<sy_min)sy_min=sy; static int pr=0; if(!pr){pr=1;printf("KD_MIN %d\n",KD_MIN);} if(sy>sy_max)sy_max=sy; if(sd>sd_max)sd_max=sd; if(e>e_max)e_max=e;
    for(int c=0;c<16;c++){ int64 y=srs((int64)gamma[c]*dn2[c],sy)+beta[c]; row[c]=(int16)sat(y,16); }
  }
}
int main(){
  srand(7); double maxerr=0,sumerr=0; int n=0; double worst_scale=0;
  for(int t=0;t<40000;t++){
    int16 x[16],g[16],bt[16],xi[16];
    float scale=(t%5==0)?60.0f:(t%5==1?1.0f:(t%5==2?0.02f:(t%5==3?0.004f:8.0f)));
    for(int c=0;c<16;c++){ float v=((rand()%20001)-10000)/10000.0f*scale; int q=(int)lroundf(v*512); if(q>32767)q=32767; if(q<-32768)q=-32768; x[c]=xi[c]=(int16)q;
      g[c]=(int16)lroundf((0.5f+(rand()%1000)/1000.0f)*512*((t%7==0)?4.0f:1.0f)); bt[c]=(int16)lroundf((((rand()%2001)-1000)/1000.0f)*512); }
    if(t%11==0){ for(int c=1;c<16;c++) x[c]=xi[c]=xi[0]; x[3]=xi[3]=xi[0]+1; }   // near-constant row
    double rf[16],mean=0; for(int c=0;c<16;c++){ rf[c]=xi[c]/512.0; mean+=rf[c]; } mean/=16;
    double var=0; for(int c=0;c<16;c++){ double dd=rf[c]-mean; var+=dd*dd; } var/=16; double inv=1.0/sqrt(var+1e-5);
    layernorm_row(x,1,16,g,bt);
    for(int c=0;c<16;c++){ double y=(g[c]/512.0)*(rf[c]-mean)*inv+bt[c]/512.0; if(y>32767/512.0)y=32767/512.0; if(y<-32768/512.0)y=-32768/512.0; double e=fabs(y-x[c]/512.0); if(e>maxerr){maxerr=e;worst_scale=scale;} sumerr+=e; n++; }
  }
  printf("vector-mirror LN vs float reference: max |err| = %.5f (%.2f LSB, LSB=%.5f) at scale %g, mean |err| = %.6f LSB over %d values\n",maxerr,maxerr*512,1/512.0,worst_scale,sumerr/n*512,n);
  printf("ranges: sy [%d,%d] sd_max %d e_max %d\n",sy_min,sy_max,sd_max,e_max);
  return maxerr*512<1.0?0:1;
}
