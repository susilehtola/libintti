// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
// Benchmark: GEMM (gemm_nn/BLAS) vs hand-loop for the phase-G contraction
// G_d[combo,i] = sum_n fh_d[n,combo] B_d[n,i], per direction, over nt t-nodes.
// Reports time(hand)/time(gemm) per angular-momentum tuple -> the crossover.
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>
extern "C" void dgemm_(const char*,const char*,const int*,const int*,const int*,
  const double*,const double*,const int*,const double*,const int*,const double*,double*,const int*);
using Real=double;
static void gemm(int m,int n,int k,const Real*A,int lda,const Real*B,int ldb,Real*C,int ldc){
  const Real one=1,zero=0; dgemm_("N","N",&m,&n,&k,&one,A,&lda,B,&ldb,&zero,C,&ldc); }
int main(){
  std::mt19937 rng(1); std::uniform_real_distribution<Real> u(-1,1);
  const int nt=64; // typical Mobius grid
  struct C{int la,lb,lc,ld;};
  C cases[]={{0,0,0,0},{1,0,0,0},{1,1,0,0},{1,1,1,0},{1,1,1,1},{2,1,1,1},{2,2,1,1},{2,2,2,2},{3,2,2,2},{3,3,3,3}};
  std::printf("%-12s %5s %5s %6s  %10s %10s  %s\n","l-tuple","ncomb","nf","nt","hand(us)","gemm(us)","speedup");
  for(auto&c:cases){
    const int nf=c.la+c.lb+c.lc+c.ld+1;
    const int ncomb=(c.la+1)*(c.lb+1)*(c.lc+1)*(c.ld+1);
    std::vector<Real> fhT(3*ncomb*nf), B(3*nf*nt), G(3*ncomb*nt);
    for(auto&x:fhT)x=u(rng); for(auto&x:B)x=u(rng);
    const int reps=std::max(2000, 200000/(ncomb*nt/8+1));
    // hand loop
    auto t0=std::chrono::high_resolution_clock::now();
    for(int r=0;r<reps;++r) for(int d=0;d<3;++d)
      for(int i=0;i<nt;++i) for(int cb=0;cb<ncomb;++cb){ Real s=0;
        for(int n=0;n<nf;++n) s+=fhT[cb+ncomb*(n+nf*d)]*B[n+nf*(i+nt*d)];
        G[cb+ncomb*(i+nt*d)]=s; }
    auto t1=std::chrono::high_resolution_clock::now();
    for(int r=0;r<reps;++r) for(int d=0;d<3;++d)
      gemm(ncomb,nt,nf,&fhT[ncomb*nf*d],ncomb,&B[nf*nt*d],nf,&G[ncomb*nt*d],ncomb);
    auto t2=std::chrono::high_resolution_clock::now();
    double th=std::chrono::duration<double,std::micro>(t1-t0).count()/reps;
    double tg=std::chrono::duration<double,std::micro>(t2-t1).count()/reps;
    std::printf("(%d%d%d%d)       %5d %5d %6d  %10.3f %10.3f  %.2fx\n",
      c.la,c.lb,c.lc,c.ld,ncomb,nf,nt,th,tg,th/tg);
  }
  return 0;}
