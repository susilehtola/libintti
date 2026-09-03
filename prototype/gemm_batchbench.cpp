// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
// Realistic-batch test: N quartets at fixed L. Compare
//  (a) OpenMP-parallel hand-loop contraction over quartets (~ current phase G)
//  (b) serial loop of per-quartet BLAS GEMMs (the naive #1 integration)
// If (a) wins, the naive drop-in regresses; #1 needs a batched GEMM.
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>
#include <omp.h>
extern "C" void dgemm_(const char*,const char*,const int*,const int*,const int*,
  const double*,const double*,const int*,const double*,const int*,const double*,double*,const int*);
using Real=double;
int main(){
  std::mt19937 rng(1); std::uniform_real_distribution<Real> u(-1,1);
  const int nt=64, N=1024; // 1024 quartets per batch
  struct C{int la,lb,lc,ld;};
  C cases[]={{1,1,0,0},{1,1,1,1},{2,2,1,1},{2,2,2,2},{3,3,2,2}};
  std::printf("N=%d quartets, nt=%d, threads=%d\n", N, nt, omp_get_max_threads());
  std::printf("%-10s %5s %5s  %12s %12s  %s\n","l-tuple","ncomb","nf","omp-hand(ms)","serial-gemm(ms)","hand/gemm");
  for(auto&c:cases){
    const int nf=c.la+c.lb+c.lc+c.ld+1, ncomb=(c.la+1)*(c.lb+1)*(c.lc+1)*(c.ld+1);
    std::vector<Real> f(static_cast<size_t>(N)*3*nf*ncomb), B(static_cast<size_t>(N)*3*nf*nt),
                      G(static_cast<size_t>(N)*3*ncomb*nt);
    for(auto&x:f)x=u(rng); for(auto&x:B)x=u(rng);
    auto foff=[&](int q,int d){return (static_cast<size_t>(q)*3+d)*nf*ncomb;};
    auto boff=[&](int q,int d){return (static_cast<size_t>(q)*3+d)*nf*nt;};
    auto goff=[&](int q,int d){return (static_cast<size_t>(q)*3+d)*ncomb*nt;};
    const int reps=20;
    // (a) OpenMP hand-loop over quartets
    auto t0=std::chrono::high_resolution_clock::now();
    for(int r=0;r<reps;++r)
      #pragma omp parallel for
      for(int q=0;q<N;++q) for(int d=0;d<3;++d){
        const Real*ff=&f[foff(q,d)], *bb=&B[boff(q,d)]; Real*gg=&G[goff(q,d)];
        for(int i=0;i<nt;++i) for(int cb=0;cb<ncomb;++cb){ Real s=0;
          for(int n=0;n<nf;++n) s+=ff[n+nf*cb]*bb[n+nf*i]; gg[cb+ncomb*i]=s; }
      }
    auto t1=std::chrono::high_resolution_clock::now();
    // (b) serial loop of per-quartet GEMMs
    const Real one=1,zero=0;
    for(int r=0;r<reps;++r)
      for(int q=0;q<N;++q) for(int d=0;d<3;++d)
        dgemm_("T","N",&nt,&ncomb,&nf,&one,&B[boff(q,d)],&nf,&f[foff(q,d)],&nf,&zero,&G[goff(q,d)],&nt);
    auto t2=std::chrono::high_resolution_clock::now();
    double ta=std::chrono::duration<double,std::milli>(t1-t0).count()/reps;
    double tb=std::chrono::duration<double,std::milli>(t2-t1).count()/reps;
    std::printf("(%d%d%d%d)     %5d %5d  %12.3f %12.3f  %.2fx\n",c.la,c.lb,c.lc,c.ld,ncomb,nf,ta,tb,ta/tb);
  }
  return 0;}
