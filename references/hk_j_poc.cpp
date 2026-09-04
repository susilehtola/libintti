// Isolate the G_kappa J build: for a FIXED density D, the eigenpair of
// F = T + V_nuc + J[D] satisfies M_eff(kappa0) c0 = -1/2 S c0 with
// M_eff = M_nuc + M_J[D], kappa0 = sqrt(-2 eps0). No SCF needed.
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <vector>
#include <cmath>
#include "intti/oneel.hpp"
#include "intti/nuclear.hpp"
#include "intti/helmholtz.hpp"
using Real=double;
extern "C" void dsygv_(const int*,const char*,const char*,const int*,double*,const int*,double*,const int*,double*,double*,const int*,int*);
int main(int c,char**v){Kokkos::ScopeGuard S(c,v);
  int nb=c>1?atoi(v[1]):8;
  std::vector<intti::PrimitiveShell<Real>> sh;
  for(int i=0;i<nb;++i) sh.push_back({0.08*std::pow(2.5,i),{0,0,0},0});
  auto basis=intti::make_basis<Real>(sh); const int n=basis.nao;
  auto Sm=intti::overlap_matrix(basis), Tm=intti::kinetic_matrix(basis);
  std::vector<intti::PointCharge<Real>> ch={{-2.0,{0,0,0}}};
  auto cgrid=intti::make_tgrid(intti::coulomb<Real>());
  auto Vn=intti::nuclear_matrix(basis,ch,cgrid);
  std::vector<double> ERI((size_t)n*n*n*n);
  for(int a=0;a<n;++a)for(int b=0;b<n;++b)for(int d=0;d<n;++d)for(int e=0;e<n;++e){
    Real out; intti::eri_quartet(intti::make_pair(sh[a],sh[b]),intti::make_pair(sh[d],sh[e]),cgrid,&out);
    ERI[((size_t)a*n+b)*n*n+((size_t)d*n+e)]=out;}
  auto geig=[&](std::vector<double> F,std::vector<double>&eps,std::vector<double>&C){
    std::vector<double> Fc(F),Sc(Sm.begin(),Sm.end()),w(n);int it=1,lw=8*n,info;std::vector<double> wk(lw);
    dsygv_(&it,"V","U",&n,Fc.data(),&n,Sc.data(),&n,w.data(),wk.data(),&lw,&info);eps=w;C=Fc;};
  // fixed density from H_core ground state
  std::vector<double> eps,C,ch0(n); std::vector<double> H0(n*n);for(int i=0;i<n*n;++i)H0[i]=Tm[i]+Vn[i];
  geig(H0,eps,C); for(int i=0;i<n;++i)ch0[i]=C[i];
  std::vector<double> D(n*n); for(int i=0;i<n;++i)for(int j=0;j<n;++j)D[i*n+j]=ch0[i]*ch0[j];
  // J[D] and F = T+Vn+J
  std::vector<double> J(n*n,0);
  for(int mu=0;mu<n;++mu)for(int nu=0;nu<n;++nu){double s=0;for(int r=0;r<n;++r)for(int sg=0;sg<n;++sg)s+=D[r*n+sg]*ERI[((size_t)mu*n+nu)*n*n+((size_t)r*n+sg)];J[mu*n+nu]=s;}
  std::vector<double> F(n*n);for(int i=0;i<n*n;++i)F[i]=Tm[i]+Vn[i]+J[i];
  geig(F,eps,C); std::vector<double> c0(n);for(int i=0;i<n;++i)c0[i]=C[i]; double e0=eps[0];
  printf("n=%d  eps0(F) = %.8f\n",n,e0);
  double kappa=std::sqrt(-2*e0);
  auto Mn=intti::helmholtz_nuclear_matrix(basis,ch,kappa);
  auto yg=intti::yukawa_grid<Real>(kappa,1e-2,1e3,0.15); const Real pi=M_PI;
  std::vector<double> MJ(n*n,0); intti::PrimitiveShell<Real> ghost{0.0,{0,0,0},0};
  for(int mu=0;mu<n;++mu)for(int nu=0;nu<n;++nu){Real acc=0;
    for(int r=0;r<n;++r)for(int sg=0;sg<n;++sg){Real Drs=D[r*n+sg]; if(Drs==0)continue; Real p=sh[r].alpha+sh[sg].alpha;
      for(int j=0;j<cgrid.n();++j){Real t2=cgrid.t[j]*cgrid.t[j],mu_rs=p*t2/(p+t2),smear=std::pow(pi/(p+t2),1.5);
        intti::PrimitiveShell<Real> gS{mu_rs,{0,0,0},0}; Real out;
        intti::eri_quartet(intti::make_pair(sh[mu],ghost),intti::make_pair(sh[nu],gS),yg,&out);
        acc+=Drs*(1.0/(4*pi))*cgrid.w[j]*smear*out;}}
    MJ[mu*n+nu]=acc;}
  double num=0,den=0,numN=0;
  for(int i=0;i<n;++i){double Mc=0,Mnc=0,Sc0=0;for(int k=0;k<n;++k){Mc+=(Mn[i*n+k]+MJ[i*n+k])*c0[k];Mnc+=Mn[i*n+k]*c0[k];Sc0+=Sm[i*n+k]*c0[k];}
    num+=(Mc+0.5*Sc0)*(Mc+0.5*Sc0);numN+=(Mnc+0.5*Sc0)*(Mnc+0.5*Sc0);den+=(0.5*Sc0)*(0.5*Sc0);}
  printf("  M_nuc only residual         = %.3e (should be large, J missing)\n",std::sqrt(numN/den));
  printf("  M_nuc + M_J residual        = %.3e (should be basis-limited)\n",std::sqrt(num/den));
  return 0;}
