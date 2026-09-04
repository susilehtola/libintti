// M-HK de-risk: the two-kernel M(kappa)_munu = <mu|G_kappa V|nu> for the H atom,
// validated by the fixed-point consistency  M(kappa0) c0 = -1/2 S c0  at the
// H_core ground state (eps0,c0), kappa0=sqrt(-2 eps0). M(kappa) is built as
//   -Z (1/4pi) sum_j w_j <mu.ghost | e^{-kappa r12}/r12 | nu.G_C^{t_j}>
// (Yukawa kernel = 4pi G_kappa; nuclear 1/r' unfolds into the Coulomb t-grid).
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <vector>
#include <cmath>
#include "intti/oneel.hpp"
#include "intti/nuclear.hpp"
#include "intti/quartet.hpp"
#include "intti/tgrid.hpp"
using Real=double;
extern "C" void dsygv_(const int*,const char*,const char*,const int*,double*,const int*,double*,const int*,double*,double*,const int*,int*);
int main(int c,char**v){Kokkos::ScopeGuard S(c,v);
  // even-tempered s-basis for H at origin
  std::vector<Real> exps; for(int i=0;i<12;++i) exps.push_back(0.04*std::pow(2.6,i));
  std::vector<intti::PrimitiveShell<Real>> sh;
  for(Real e:exps) sh.push_back({e,{0,0,0},0});
  auto basis=intti::make_basis<Real>(sh);
  const int n=basis.nao;
  auto Sm=intti::overlap_matrix(basis);
  auto Tm=intti::kinetic_matrix(basis);
  std::vector<intti::PointCharge<Real>> ch={{-1.0,{0,0,0}}}; // -Z, Z=1
  intti::TGridSpec<Real> cs; cs.n=96; auto cgrid=intti::make_tgrid(intti::coulomb<Real>(),cs);
  auto Vm=intti::nuclear_matrix(basis,ch,cgrid);
  // H_core = T+V ; solve (Hc)=eps S c via dsygv (copies, column-major symmetric)
  std::vector<double> H(n*n),Sc(n*n),w(n);
  for(int i=0;i<n*n;++i){H[i]=Tm[i]+Vm[i];Sc[i]=Sm[i];}
  int itype=1,lwork=8*n,info; std::vector<double> work(lwork);
  dsygv_(&itype,"V","U",&n,H.data(),&n,Sc.data(),&n,w.data(),work.data(),&lwork,&info);
  // eigenvectors in columns of H (column-major). lowest = column 0.
  double eps0=w[0]; std::vector<double> c0(n); for(int i=0;i<n;++i)c0[i]=H[i]; // col 0
  printf("H_core ground state eps0 = %.10f (exact H = -0.5)  info=%d\n",eps0,info);
  if(eps0>=0){printf("no bound state\n");return 1;}
  double kappa=std::sqrt(-2*eps0);
  auto ygrid=intti::make_tgrid(intti::yukawa<Real>(kappa),intti::exp_sum_spec_for_range<Real>(1e-3,1e4,0.08));
  const Real pi=M_PI;
  // M(kappa)_munu
  std::vector<double> M(n*n,0);
  for(int mu=0;mu<n;++mu)for(int nu=0;nu<n;++nu){
    intti::PrimitiveShell<Real> ghost{0.0,{0,0,0},0};
    Real acc=0;
    for(int j=0;j<cgrid.n();++j){
      intti::PrimitiveShell<Real> gC{cgrid.t[j]*cgrid.t[j],{0,0,0},0};
      auto bra=intti::make_pair(sh[mu],ghost), ket=intti::make_pair(sh[nu],gC);
      Real out; intti::eri_quartet(bra,ket,ygrid,&out);
      acc += cgrid.w[j]*out;
    }
    M[mu*n+nu] = -1.0*(1.0/(4*pi))*acc; // -Z (1/4pi) sum_j w_j (...)
  }
  // consistency: M c0 vs -1/2 S c0
  double num=0,den=0;
  for(int i=0;i<n;++i){double Mc=0,Sc0=0;for(int k=0;k<n;++k){Mc+=M[i*n+k]*c0[k];Sc0+=Sm[i*n+k]*c0[k];}
    num+=(Mc+0.5*Sc0)*(Mc+0.5*Sc0); den+=(0.5*Sc0)*(0.5*Sc0);}
  printf("||M c0 + 1/2 S c0|| / ||1/2 S c0|| = %.3e\n",std::sqrt(num/den));
  return 0;}
