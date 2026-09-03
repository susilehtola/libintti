// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
// Measure, in the J-engine per (p,q) node, the shareable work (hermite_b B
// arrays) vs the density-specific contraction (t1/t2/jp). #3 (bra-ket symmetry)
// can only remove the former; if it's a small fraction, #3 doesn't help jbuild.
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>
#include "intti/hermite1d.hpp"
using Real=double;
int main(){
  std::mt19937 rng(1); std::uniform_real_distribution<Real> u(-1,1);
  const int nt=64;
  // per-pair angular sizes: np1=Lp+1, nq1=Lq+1 (Lp=la+lb)
  struct C{int Lp,Lq;const char*name;};
  C cs[]={{0,0,"ss"},{2,2,"pp*"},{4,4,"dd*"},{2,4,"pd*"}}; // Lp=la+lb; dd pair -> Lp up to 4
  std::printf("%-6s %4s %4s  %10s %10s  %s\n","pair","np1","nq1","B(us)","contract(us)","B/total");
  for(auto&c:cs){
    const int np1=c.Lp+1, nq1=c.Lq+1, nB=c.Lp+c.Lq+1;
    std::vector<Real> dq(nq1*nq1*nq1), t1(np1*nq1*nq1), t2(np1*np1*nq1), jp(np1*np1*np1);
    for(auto&x:dq)x=u(rng);
    std::vector<Real> Bx(2*8+9),By(2*8+9),Bz(2*8+9);
    const int reps=200000;
    Real theta=0.7, X0=0.3,X1=-0.2,X2=0.4;
    // time B only
    auto t0=std::chrono::high_resolution_clock::now();
    for(int r=0;r<reps;++r) for(int it=0;it<nt;++it){
      intti::hermite_b(nB-1,theta,X0,Bx.data());
      intti::hermite_b(nB-1,theta,X1,By.data());
      intti::hermite_b(nB-1,theta,X2,Bz.data());
    }
    auto t1c=std::chrono::high_resolution_clock::now();
    // time contraction (t1/t2/jp), reusing fixed B
    intti::hermite_b(nB-1,theta,X0,Bx.data()); intti::hermite_b(nB-1,theta,X1,By.data()); intti::hermite_b(nB-1,theta,X2,Bz.data());
    for(int r=0;r<reps;++r) for(int it=0;it<nt;++it){
      for(int t=0;t<np1;++t)for(int nu=0;nu<nq1;++nu)for(int ph=0;ph<nq1;++ph){Real s=0;
        for(int tau=0;tau<nq1;++tau){Real tm=dq[(tau*nq1+nu)*nq1+ph]*Bx[t+tau]; s+=tau%2?-tm:tm;} t1[(t*nq1+nu)*nq1+ph]=s;}
      for(int t=0;t<np1;++t)for(int uu=0;uu<np1;++uu)for(int ph=0;ph<nq1;++ph){Real s=0;
        for(int nu=0;nu<nq1;++nu){Real tm=t1[(t*nq1+nu)*nq1+ph]*By[uu+nu]; s+=nu%2?-tm:tm;} t2[(t*np1+uu)*nq1+ph]=s;}
      for(int t=0;t<np1;++t)for(int uu=0;uu<np1;++uu)for(int v=0;v<np1;++v){Real s=0;
        for(int ph=0;ph<nq1;++ph){Real tm=t2[(t*np1+uu)*nq1+ph]*Bz[v+ph]; s+=ph%2?-tm:tm;} jp[(t*np1+uu)*np1+v]+=s;}
    }
    auto t2c=std::chrono::high_resolution_clock::now();
    double tB=std::chrono::duration<double,std::micro>(t1c-t0).count()/reps;
    double tC=std::chrono::duration<double,std::micro>(t2c-t1c).count()/reps;
    std::printf("%-6s %4d %4d  %10.4f %10.4f  %.1f%%\n",c.name,np1,nq1,tB,tC,100*tB/(tB+tC));
  }
  return 0;}
