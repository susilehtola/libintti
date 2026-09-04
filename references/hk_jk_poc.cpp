// M_K (exchange) de-risk: three-kernel like M_J. For a FIXED density D=2cc^T, the
// eigenpair of the RHF-like F = T+V_nuc+J[D]-1/2 K[D] satisfies
// M_eff(kappa0)c0 = -1/2 S c0 with M_eff = M_nuc + M_J[D] - 1/2 M_K[D].
//   M_K[D]_munu = sum_{rs} D_rs (1/4pi) sum_t w_t (pi/(p_sn+t^2))^{3/2}
//     <mu.ghost | e^{-kappa r12}/r12 | rho . G(O, p_sn t^2/(p_sn+t^2)) >   [comp mu,rho]
//   where p_sn = alpha_sigma + alpha_nu (the (sigma,nu) pair smear).
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
    Real out;intti::eri_quartet(intti::make_pair(sh[a],sh[b]),intti::make_pair(sh[d],sh[e]),cgrid,&out);
    ERI[((size_t)a*n+b)*n*n+((size_t)d*n+e)]=out;}
  auto geig=[&](std::vector<double> F,std::vector<double>&eps,std::vector<double>&C){
    std::vector<double> Fc(F),Sc(Sm.begin(),Sm.end()),w(n);int it=1,lw=8*n,info;std::vector<double> wk(lw);
    dsygv_(&it,"V","U",&n,Fc.data(),&n,Sc.data(),&n,w.data(),wk.data(),&lw,&info);eps=w;C=Fc;};
  std::vector<double> eps,C,ch0(n),H0(n*n);for(int i=0;i<n*n;++i)H0[i]=Tm[i]+Vn[i];
  geig(H0,eps,C);for(int i=0;i<n;++i)ch0[i]=C[i];
  std::vector<double> D(n*n);for(int i=0;i<n;++i)for(int j=0;j<n;++j)D[i*n+j]=2*ch0[i]*ch0[j]; // RHF density
  // J,K matrices and RHF-like F = T+Vn+J-1/2 K
  std::vector<double> Jm(n*n,0),Km(n*n,0);
  for(int mu=0;mu<n;++mu)for(int nu=0;nu<n;++nu){double sj=0,sk=0;
    for(int r=0;r<n;++r)for(int sg=0;sg<n;++sg){double d=D[r*n+sg];
      sj+=d*ERI[((size_t)mu*n+nu)*n*n+((size_t)r*n+sg)]; sk+=d*ERI[((size_t)mu*n+r)*n*n+((size_t)nu*n+sg)];}
    Jm[mu*n+nu]=sj;Km[mu*n+nu]=sk;}
  std::vector<double> F(n*n);for(int i=0;i<n*n;++i)F[i]=Tm[i]+Vn[i]+Jm[i]-0.5*Km[i];
  geig(F,eps,C);std::vector<double> c0(n);for(int i=0;i<n;++i)c0[i]=C[i];double e0=eps[0];
  printf("n=%d  eps0(RHF-like F) = %.8f\n",n,e0);
  double kappa=std::sqrt(-2*e0);
  auto Mn=intti::helmholtz_nuclear_matrix(basis,ch,kappa);
  auto yg=intti::yukawa_grid<Real>(kappa,1e-2,1e3,0.15);const Real pi=M_PI;
  intti::PrimitiveShell<Real> ghost{0.0,{0,0,0},0};
  auto smearbuild=[&](bool exch)->std::vector<double>{ // false=J, true=K
    std::vector<double> Mx(n*n,0);
    for(int mu=0;mu<n;++mu)for(int nu=0;nu<n;++nu){Real acc=0;
      for(int r=0;r<n;++r)for(int sg=0;sg<n;++sg){Real d=D[r*n+sg];if(d==0)continue;
        // J: smear (r,sg), ket first shell = nu ; K: smear (sg,nu), ket first shell = r
        int ketfirst = exch? r : nu; Real p = exch? (sh[sg].alpha+sh[nu].alpha) : (sh[r].alpha+sh[sg].alpha);
        for(int j=0;j<cgrid.n();++j){Real t2=cgrid.t[j]*cgrid.t[j],mrs=p*t2/(p+t2),sm=std::pow(pi/(p+t2),1.5);
          intti::PrimitiveShell<Real> gS{mrs,{0,0,0},0};Real out;
          intti::eri_quartet(intti::make_pair(sh[mu],ghost),intti::make_pair(sh[ketfirst],gS),yg,&out);
          acc+=d*(1.0/(4*pi))*cgrid.w[j]*sm*out;}}
      Mx[mu*n+nu]=acc;}
    return Mx;};
  auto MJ=smearbuild(false), MK=smearbuild(true);
  std::vector<double> Meff(n*n);for(int i=0;i<n*n;++i)Meff[i]=Mn[i]+MJ[i]-0.5*MK[i];
  double num=0,den=0;for(int i=0;i<n;++i){double Mc=0,Sc0=0;for(int k=0;k<n;++k){Mc+=Meff[i*n+k]*c0[k];Sc0+=Sm[i*n+k]*c0[k];}
    num+=(Mc+0.5*Sc0)*(Mc+0.5*Sc0);den+=(0.5*Sc0)*(0.5*Sc0);}
  printf("  M_nuc+M_J-1/2 M_K residual = %.3e (should be basis-limited)\n",std::sqrt(num/den));
  return 0;}
