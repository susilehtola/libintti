// M-HK step 3: Helmholtz-SCF PoC for the H atom. The pure integral-form
// eigenvalue eps* is the self-consistent point where A(eps)=-2 S^{-1} M(kappa)
// (kappa=sqrt(-2 eps)) has dominant eigenvalue 1 -- extracted WITHOUT the
// projected kinetic. Compare eps* (integral) vs eps0 (Galerkin H_core) vs the
// exact -0.5 to test the kinetic-exactness claim.
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <vector>
#include <cmath>
#include "intti/oneel.hpp"
#include "intti/nuclear.hpp"
#include "intti/helmholtz.hpp"
using Real=double;
extern "C" void dsygv_(const int*,const char*,const char*,const int*,double*,const int*,double*,const int*,double*,double*,const int*,int*);
extern "C" void dsyev_(const char*,const char*,const int*,double*,const int*,double*,double*,const int*,int*);
int main(int c,char**v){Kokkos::ScopeGuard S(c,v);
  int nb = c>1? atoi(v[1]) : 8;
  std::vector<intti::PrimitiveShell<Real>> sh;
  for(int i=0;i<nb;++i) sh.push_back({0.05*std::pow(2.5,i),{0,0,0},0});
  auto basis=intti::make_basis<Real>(sh); const int n=basis.nao;
  auto Sm=intti::overlap_matrix(basis), Tm=intti::kinetic_matrix(basis);
  std::vector<intti::PointCharge<Real>> ch={{-1.0,{0,0,0}}};
  auto Vm=intti::nuclear_matrix(basis,ch,intti::make_tgrid(intti::coulomb<Real>()));
  // Galerkin H_core ground state
  std::vector<double> H(n*n),Sc(n*n),w(n); for(int i=0;i<n*n;++i){H[i]=Tm[i]+Vm[i];Sc[i]=Sm[i];}
  int it=1,lw=8*n,info; std::vector<double> wk(lw);
  dsygv_(&it,"V","U",&n,H.data(),&n,Sc.data(),&n,w.data(),wk.data(),&lw,&info);
  double eps0=w[0];
  // Sinv = U diag(1/s) U^T  (dsyev on a copy of S)
  std::vector<double> Se(Sm.begin(),Sm.end()), sw(n); dsyev_("V","U",&n,Se.data(),&n,sw.data(),wk.data(),&lw,&info);
  auto Uev=[&](int i,int k){return Se[(size_t)k*n+i];};
  std::vector<double> Sinv(n*n,0); for(int i=0;i<n;++i)for(int j=0;j<n;++j){double s=0;for(int k=0;k<n;++k)s+=Uev(i,k)*(1.0/sw[k])*Uev(j,k);Sinv[i*n+j]=s;}
  auto Snorm=[&](const std::vector<double>&x){double s=0;for(int i=0;i<n;++i)for(int j=0;j<n;++j)s+=x[i]*Sm[i*n+j]*x[j];return std::sqrt(s);};
  // dominant eigenvalue of A(eps) = -2 Sinv M(kappa) via power iteration
  auto lam=[&](double eps)->double{
    double kap=std::sqrt(-2*eps);
    auto M=intti::helmholtz_nuclear_matrix(basis,ch,kap);
    std::vector<double> x(n,0); x[0]=1; double d=Snorm(x); for(auto&e:x)e/=d;
    double l=0;
    for(int iter=0;iter<60;++iter){
      std::vector<double> Mx(n,0); for(int i=0;i<n;++i){double s=0;for(int k=0;k<n;++k)s+=M[i*n+k]*x[k];Mx[i]=s;}
      std::vector<double> y(n,0); for(int i=0;i<n;++i){double s=0;for(int k=0;k<n;++k)s+=Sinv[i*n+k]*(-2*Mx[k]);y[i]=s;}
      // Rayleigh in S-metric: l = x^T S y / x^T S x  (x is S-normalised)
      double num=0; for(int i=0;i<n;++i)for(int j=0;j<n;++j)num+=x[i]*Sm[i*n+j]*y[j];
      l=num; double d2=Snorm(y); for(int i=0;i<n;++i)x[i]=y[i]/d2;
    }
    return l;
  };
  // secant root find lam(eps)-1=0
  double e1=-0.6,e2=-0.4,f1=lam(e1)-1,f2=lam(e2)-1,es=e2;
  for(int k=0;k<40 && std::fabs(f2)>1e-11;++k){ es=e2-f2*(e2-e1)/(f2-f1); e1=e2;f1=f2;e2=es;f2=lam(es)-1; }
  printf("basis n=%d\n",n);
  printf("  Galerkin  eps0 = %.8f   err vs -0.5 = %+.2e\n",eps0,eps0+0.5);
  printf("  Helmholtz eps* = %.8f   err vs -0.5 = %+.2e   (lambda-1=%.1e)\n",es,es+0.5,f2);
  printf("  eps* - eps0 = %+.2e   kinetic-exact %s\n",es-eps0, std::fabs(es+0.5)<std::fabs(eps0+0.5)?"BETTER":"not better");
  return 0;}
