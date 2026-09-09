// scalar mirror of the fabric integer layer norm (LN_MODE=2): data 2^9,
// gamma/beta 2^12, output 2^9 -- checked against the float path it replaces.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstdlib>
typedef int int32;
#include "/home/snehadri/repos/aie-unsupervised-search/src/attn_block_aie/kernels/ln_rsqrt_lut.h"
#define LN_EPSV 10737
static int clz46(unsigned long long v){ int n=0; for(int b=45;b>=0;b--){ if(v>>b) break; n++; } return n; }
int main(){
  srand(11); double maxe=0, sume=0; int n=0;
  for(int t=0;t<40000;t++){
    int16_t xq[16], gq[16], bq[16];
    float scale = (t%5==0)?40.0f:(t%5==1?1.0f:(t%5==2?0.02f:(t%5==3?0.003f:6.0f)));
    for(int j=0;j<16;j++){
      float v=((rand()%20001)-10000)/10000.0f*scale; int q=(int)lroundf(v*512);
      if(q>32767)q=32767; if(q<-32768)q=-32768; xq[j]=(int16_t)q;
      gq[j]=(int16_t)lroundf((0.5f+(rand()%1000)/1000.0f)*4096);
      bq[j]=(int16_t)lroundf((((rand()%2001)-1000)/1000.0f)*4096);
    }
    if(t%9==0){ for(int j=1;j<16;j++) xq[j]=xq[0]; xq[5]=xq[0]+1; }   // near-constant row
    // float reference, exactly what LN_MODE=0 computes
    float mean=0; for(int j=0;j<16;j++) mean+=xq[j]/512.0f; mean/=16;
    float var=0; for(int j=0;j<16;j++){ float d=xq[j]/512.0f-mean; var+=d*d; } var/=16;
    float inv=1.0f/sqrtf(var+1e-5f);
    // integer path
    int32_t isum=0; for(int j=0;j<16;j++) isum+=xq[j];
    long long d16[16]; unsigned long long V=0;
    for(int j=0;j<16;j++){ d16[j]=((long long)xq[j]<<4)-isum; V+=(unsigned long long)(d16[j]*d16[j]); }
    unsigned long long Vp=V+LN_EPSV;
    int L=46-clz46(Vp);                  // bit length of Vp
    int nsh=L-32; if(nsh&1) nsh++;       // even, so sqrt splits cleanly
    unsigned int Vn = (nsh>=0) ? (unsigned int)(Vp>>nsh) : (unsigned int)(Vp<<(-nsh));
    int idx=(int)(Vn>>23)-128; int32_t frac=(int32_t)((Vn>>7)&0xFFFF);
    int32_t l0=LN_RSQRT_LUT[idx], l1=LN_RSQRT_LUT[idx+1];
    int32_t R=l0+(((l1-l0)*frac)>>16);
    int sh=33+(nsh>>1);
    for(int j=0;j<16;j++){
      long long num=(long long)gq[j]*d16[j]*R;
      long long yq=((num+((long long)1<<(sh-1)))>>sh)+((bq[j]+4)>>3);
      if(yq>32767)yq=32767; if(yq<-32768)yq=-32768;
      float ref=(gq[j]/4096.0f)*((xq[j]/512.0f)-mean)*inv+(bq[j]/4096.0f);
      if(ref>32767/512.0f)ref=32767/512.0f; if(ref<-32768/512.0f)ref=-32768/512.0f;
      double err=fabs(ref-yq/512.0)*512; if(err>maxe)maxe=err; sume+=err; n++;
    }
  }
  printf("fabric integer LN vs its float path: max %.2f LSB, mean %.3f LSB over %d values\n", maxe, sume/n, n);
  return maxe < 2.0 ? 0 : 1;
}
