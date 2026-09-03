// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
// #4 ceiling: the 2e gradient recomputes promoted (l+1) quartets per position.
// Time base quartet vs the l+1 promote (per position) to see what D,P saves.
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include "intti/quartet.hpp"
using Real=double;
static double timeq(const intti::ShellPair<Real>&a,const intti::ShellPair<Real>&b,
                    const intti::TGrid<Real>&g,int reps){
  const int nout=intti::ncart(a.la)*intti::ncart(a.lb)*intti::ncart(b.la)*intti::ncart(b.lb);
  std::vector<Real> o(nout);
  auto t0=std::chrono::high_resolution_clock::now();
  for(int r=0;r<reps;++r) intti::eri_quartet(a,b,g,o.data());
  auto t1=std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double,std::micro>(t1-t0).count()/reps;
}
int main(int argc,char**argv){ Kokkos::ScopeGuard G(argc,argv);
  auto sh=[](Real al,Real x,Real y,Real z,int l){return intti::PrimitiveShell<Real>{al,{x,y,z},l};};
  auto grid=intti::make_tgrid(intti::coulomb());
  int L[][4]={{0,0,0,0},{1,1,1,1},{2,2,1,1},{2,2,2,2}};
  std::printf("%-10s %10s %10s  %s\n","base","base(us)","+1pos0(us)","promote/base");
  for(auto&l:L){ const int reps=std::max(500,50000/(intti::ncart(l[0])*intti::ncart(l[1])*intti::ncart(l[2])*intti::ncart(l[3])+1));
    auto A=sh(0.8,0,0,0,l[0]),B=sh(1.3,0.5,-0.2,0.4,l[1]),C=sh(2.1,1,0.8,0,l[2]),Dd=sh(0.35,-0.4,0.3,1.1,l[3]);
    double tb=timeq(intti::make_pair(A,B),intti::make_pair(C,Dd),grid,reps);
    auto Ap=sh(0.8,0,0,0,l[0]+1); // promote shell a
    double tp=timeq(intti::make_pair(Ap,B),intti::make_pair(C,Dd),grid,reps);
    std::printf("(%d%d%d%d)     %10.3f %10.3f  %.2fx\n",l[0],l[1],l[2],l[3],tb,tp,tp/tb);
  }
  std::printf("gradient does ~4 positions x (1 promote + 1 demote); D,P avoids the promotes\n");
  return 0;}
