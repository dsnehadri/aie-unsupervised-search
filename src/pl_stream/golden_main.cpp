#include "pl_stream.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <fstream>
// Vitis csim normally links these math intrinsics; provide native equivalents.
namespace hls {
    float  rsqrt(float x)  { return 1.0f / std::sqrt(x); }
    double rsqrt(double x) { return 1.0  / std::sqrt(x); }
}
extern "C" void pl_stream_top(ap_uint<32>* in_buf, ap_uint<32>* out_buf, int n_events);
int main(int argc, char** argv){
    const int N = (argc>1)? atoi(argv[1]) : 1;
    const int WIN=72, WOUT=3;
    ap_uint<32>* in_buf  = new ap_uint<32>[N*WIN];
    ap_uint<32>* out_buf = new ap_uint<32>[N*WOUT];
    std::ifstream f("input.bin", std::ios::binary);
    if(!f){ printf("no input.bin\n"); return 1; }
    uint32_t* tmp = new uint32_t[N*WIN];
    f.read((char*)tmp, (long)N*WIN*4);
    for(int i=0;i<N*WIN;i++) in_buf[i]=tmp[i];
    pl_stream_top(in_buf, out_buf, N);
    for(int e=0;e<N;e++){
        float l[3];
        for(int j=0;j<3;j++){ uint32_t w=(uint32_t)out_buf[e*3+j]; std::memcpy(&l[j],&w,4); }
        printf("GOLDEN(all-PL) ev%d: MSE=%.5f  crossed=%.5f  latent=%.5f\n", e, l[0], l[1], l[2]);
    }
    return 0;
}
