// M-FE exchange route (one-center prototype): validate the co-density /
// occupied-orbital ASSEMBLY of exchange against the analytic exchange_build.
//
//   K_pq = sum_rs (pr|qs) D_rs = occ_scale sum_i (g_pi | g_qi),
//   g_pi = chi_p phi_i,  phi_i = sum_r C_ri chi_r,  D = occ_scale sum_i C_i C_i^T.
//
// Exchange is thus a sum over occupied orbitals of Coulomb interactions between
// co-densities; per orbital i it is an Nbf x Nbf matrix (fits memory), and one
// loops i accumulating K. Here the couplings (g_pi|g_qi) = sum_rs C_ri C_si
// (pr|qs) use the machine-precise SEPARABLE grid (pr|qs) from the step-1 M-FE
// prototype (near/far-split t-adapted quadrature). s-type one-center basis so
// each (pr|qs) is a single scalar A(t)^3.
//
// This validates the co-density FORMULATION + occupied-orbital assembly to
// machine precision. The genuine seminumerical ADVANTAGE -- form g_qi as one 3D
// grid function and take a single 3D DAGE Coulomb solve per (q,i), avoiding the
// O(Nbf^2) r,s expansion -- is the step-3 follow-on (needs the 3D t-adapted DAGE
// with per-axis FE interpolation).

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <Kokkos_Core.hpp>

#include "intti/fock.hpp"

using intti::PrimitiveShell;

static void gauss_legendre(int n, double a, double b, std::vector<double> &x,
                           std::vector<double> &w) {
  x.assign(n, 0.0); w.assign(n, 0.0);
  for (int i = 0; i < n; ++i) {
    double xi = std::cos(M_PI * (i + 0.75) / (n + 0.5)), dp = 0;
    for (int it = 0; it < 100; ++it) {
      double p0 = 1, p1 = xi;
      for (int k = 2; k <= n; ++k) { double p2 = ((2*k-1)*xi*p1-(k-1)*p0)/k; p0=p1; p1=p2; }
      dp = n * (xi * p1 - p0) / (xi * xi - 1);
      double dx = -p1 / dp; xi += dx;
      if (std::abs(dx) < 1e-15) break;
    }
    x[i] = 0.5*(a+b) + 0.5*(b-a)*xi;
    w[i] = (b-a) / ((1 - xi*xi) * dp*dp);
  }
}

// separable one-axis integral A(t) = int int e^{-pbra x^2} e^{-t^2 (x-x')^2}
// e^{-pket x'^2} dx dx', by the near/far-split t-adapted quadrature (step-1).
struct SpaceGrid {
  std::vector<double> u, w, rnv, rwv; // outer x-grid, inner reference v-grid
  double t_switch, vmax; int nq, nv;
};
static double A_axis(const SpaceGrid &g, double t, double pbra, double pket) {
  double A = 0;
  for (int p = 0; p < g.nq; ++p) {
    const double x = g.u[p];
    const double rb = std::exp(-pbra * x * x);
    double inner = 0;
    if (t < g.t_switch) {
      for (int k = 0; k < g.nq; ++k) { double du = x - g.u[k];
        inner += g.w[k] * std::exp(-pket * g.u[k] * g.u[k] - t*t*du*du); }
    } else {
      const double lo = -g.vmax, hi = g.vmax, vc = 0.5*(lo+hi), vh = 0.5*(hi-lo);
      for (int gI = 0; gI < g.nv; ++gI) { const double v = vc + vh*g.rnv[gI];
        const double xp = x - v / t;
        inner += vh * g.rwv[gI] * std::exp(-pket*xp*xp - v*v); }
      inner /= t;
    }
    A += g.w[p] * rb * inner;
  }
  return A;
}

int run();
int main() { Kokkos::initialize(); int rc = run(); Kokkos::finalize(); return rc; }

int run() {
  // s-type one-center basis (single component each) so (pr|qs) is scalar A(t)^3
  std::vector<PrimitiveShell<double>> shells = {
      {1.6, {0, 0, 0}, 0}, {0.9, {0, 0, 0}, 0}, {0.5, {0, 0, 0}, 0}, {0.3, {0, 0, 0}, 0}};
  auto basis = intti::make_basis(shells);
  const int n = basis.nao; // = nshell (all s)
  const int nocc = 2;
  const double occ_scale = 2.0;

  // model occupied orbitals C (n x nocc) and density D = occ_scale C C^T
  std::mt19937 rng(7);
  std::normal_distribution<double> nd;
  std::vector<double> C((size_t)n * nocc);
  for (auto &c : C) c = nd(rng);
  std::vector<double> D((size_t)n * n, 0.0);
  for (int p = 0; p < n; ++p)
    for (int qq = 0; qq < n; ++qq) {
      double s = 0;
      for (int i = 0; i < nocc; ++i) s += C[p * nocc + i] * C[qq * nocc + i];
      D[p * n + qq] = occ_scale * s;
    }

  // analytic oracle
  auto tgrid = intti::make_tgrid(intti::coulomb());
  std::vector<double> Kref((size_t)n * n);
  intti::exchange_build(basis, D.data(), tgrid, Kref.data(), 0.0);

  // grid space quadrature (near/far-split t-adapted)
  SpaceGrid g;
  g.nq = 160; g.nv = 96; g.t_switch = 2.0; g.vmax = 8.0;
  gauss_legendre(g.nq, -8.0, 8.0, g.u, g.w);
  gauss_legendre(g.nv, -1.0, 1.0, g.rnv, g.rwv);
  const int nt = tgrid.n();

  // grid ERIs (pr|qs) = sum_t w_t A(t)^3, A per axis with pbra=a_p+a_r, pket=a_q+a_s
  auto alpha = [&](int i) { return shells[i].alpha; };
  auto grid_eri = [&](int p, int r, int qq, int s) {
    const double pbra = alpha(p) + alpha(r), pket = alpha(qq) + alpha(s);
    double v = 0;
    for (int it = 0; it < nt; ++it) { const double a = A_axis(g, tgrid.t[it], pbra, pket);
      v += tgrid.w[it] * a * a * a; }
    return v;
  };

  // co-density / occupied-orbital assembly: K_pq = occ_scale sum_i (pi|qi),
  //   (pi|qi) = sum_rs C_ri C_si (pr|qs)
  std::vector<double> Kco((size_t)n * n, 0.0);
  for (int i = 0; i < nocc; ++i)
    for (int p = 0; p < n; ++p)
      for (int qq = 0; qq < n; ++qq) {
        double kpq = 0;
        for (int r = 0; r < n; ++r)
          for (int s = 0; s < n; ++s)
            kpq += C[r * nocc + i] * C[s * nocc + i] * grid_eri(p, r, qq, s);
        Kco[p * n + qq] += occ_scale * kpq;
      }

  double worst = 0, scale = 0;
  for (size_t k = 0; k < Kco.size(); ++k) {
    worst = std::max(worst, std::abs(Kco[k] - Kref[k]));
    scale = std::max(scale, std::abs(Kref[k]));
  }
  std::printf("== one-center exchange via co-densities vs exchange_build ==\n");
  std::printf("   n(s-basis)=%d nocc=%d\n", n, nocc);
  std::printf("   max|Kco - Kref| = %.3e   rel = %.3e\n", worst, worst / (scale + 1e-300));
  return worst / (scale + 1e-300) < 1e-10 ? 0 : 1;
}
