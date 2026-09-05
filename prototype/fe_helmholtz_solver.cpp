// M-FE / M-HK capstone: the real-space grid Helmholtz SOLVER core operator.
//
// The bound-state equation (-1/2 lap + V) psi = eps psi is equivalent to the
// Green's-function fixed point  psi = -2 G_kappa (V psi),  kappa = sqrt(-2 eps),
// G_kappa(r) = e^{-kappa r}/(4 pi r) = (Yukawa kernel)/(4 pi). Applying G_kappa on
// the FE grid is the 3D t-adapted DAGE with the yukawa t-grid (fe_yukawa_dage).
//
// Decisive test WITHOUT any SCF convergence control: for an EXACT eigenpair, one
// Green's-function step must return psi exactly. Use a shifted harmonic well
//   V(r) = 2 a^2 r^2 - C,  exact ground state psi = e^{-a r^2},  eps = 3a - C,
// (check: -1/2 lap psi + V psi = (3a - C) psi), with C > 3a so eps < 0. A pass
// shows psi_new = -2 G_kappa(V psi) reproduces psi -> the grid V-multiply +
// Helmholtz-apply solver operator is correct. The full SCF driver (iterate +
// energy update + damping/DIIS + near-nucleus refinement for cusped V) is then
// the assembly of this validated step.

#include <cmath>
#include <cstdio>
#include <vector>

#include <Kokkos_Core.hpp>

#include "intti/tgrid.hpp"

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
struct FEGrid {
  int ne, np, N;
  std::vector<double> be, rn, rw, bw, xnode, xw;
  double cen(int e) const { return 0.5 * (be[e] + be[e + 1]); }
  double hw(int e) const { return 0.5 * (be[e + 1] - be[e]); }
};
static FEGrid make_fegrid(int ne, int np, double L) {
  FEGrid g; g.ne = ne; g.np = np; g.N = ne * np;
  g.be.resize(ne + 1);
  for (int e = 0; e <= ne; ++e) g.be[e] = -L + 2 * L * e / ne;
  gauss_legendre(np, -1, 1, g.rn, g.rw);
  g.bw.assign(np, 1.0);
  for (int k = 0; k < np; ++k)
    for (int j = 0; j < np; ++j) if (j != k) g.bw[k] /= (g.rn[k] - g.rn[j]);
  g.xnode.resize(g.N); g.xw.resize(g.N);
  for (int e = 0; e < ne; ++e)
    for (int k = 0; k < np; ++k) {
      g.xnode[e * np + k] = g.cen(e) + g.hw(e) * g.rn[k];
      g.xw[e * np + k] = g.hw(e) * g.rw[k];
    }
  return g;
}
static double lag(const FEGrid &g, const double *f, double xi) {
  double num = 0, den = 0;
  for (int k = 0; k < g.np; ++k) {
    double d = xi - g.rn[k];
    if (std::abs(d) < 1e-13) return f[k];
    double tt = g.bw[k] / d; num += tt * f[k]; den += tt;
  }
  return num / den;
}
static void conv1d(const FEGrid &g, const std::vector<double> &rnv,
                   const std::vector<double> &rwv, const double *in, double t, double *out) {
  const int nv = (int)rnv.size();
  for (int jo = 0; jo < g.N; ++jo) {
    const double u1 = g.xnode[jo];
    double acc = 0;
    for (int eB = 0; eB < g.ne; ++eB) {
      double lo = t * (u1 - g.be[eB + 1]), hi = t * (u1 - g.be[eB]);
      lo = std::max(lo, -8.0); hi = std::min(hi, 8.0);
      if (hi <= lo) continue;
      const double *fB = in + (size_t)eB * g.np;
      const double vc = 0.5 * (lo + hi), vh = 0.5 * (hi - lo);
      double s = 0;
      for (int gI = 0; gI < nv; ++gI) {
        const double v = vc + vh * rnv[gI];
        const double xi = ((u1 - v / t) - g.cen(eB)) / g.hw(eB);
        s += vh * rwv[gI] * lag(g, fB, xi) * std::exp(-v * v);
      }
      acc += s / t;
    }
    out[jo] = acc;
  }
}
static std::vector<double> dage(const FEGrid &g, const std::vector<double> &rnv,
                                const std::vector<double> &rwv, const intti::TGrid<double> &tg,
                                const std::vector<double> &rho) {
  const int N = g.N; const size_t N3 = (size_t)N * N * N;
  std::vector<double> V(N3, 0.0), work(N3);
  for (int it = 0; it < tg.n(); ++it) {
    const double t = tg.t[it];
    work = rho;
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ix = 0; ix < N; ++ix)
      for (int iy = 0; iy < N; ++iy) {
        double *ln = &work[((size_t)ix * N + iy) * N];
        std::vector<double> o(N); conv1d(g, rnv, rwv, ln, t, o.data());
        for (int iz = 0; iz < N; ++iz) ln[iz] = o[iz];
      }
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ix = 0; ix < N; ++ix)
      for (int iz = 0; iz < N; ++iz) {
        std::vector<double> in(N), o(N);
        for (int iy = 0; iy < N; ++iy) in[iy] = work[((size_t)ix * N + iy) * N + iz];
        conv1d(g, rnv, rwv, in.data(), t, o.data());
        for (int iy = 0; iy < N; ++iy) work[((size_t)ix * N + iy) * N + iz] = o[iy];
      }
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int iy = 0; iy < N; ++iy)
      for (int iz = 0; iz < N; ++iz) {
        std::vector<double> in(N), o(N);
        for (int ix = 0; ix < N; ++ix) in[ix] = work[((size_t)ix * N + iy) * N + iz];
        conv1d(g, rnv, rwv, in.data(), t, o.data());
        for (int ix = 0; ix < N; ++ix) work[((size_t)ix * N + iy) * N + iz] = o[ix];
      }
    const double wt = tg.w[it];
    for (size_t i = 0; i < N3; ++i) V[i] += wt * work[i];
  }
  return V;
}

int run();
int main() { Kokkos::initialize(); int rc = run(); Kokkos::finalize(); return rc; }

int run() {
  const double a = 0.5, C = 5.0;      // shifted harmonic well
  const double eps = 3 * a - C;       // = -3.5 (bound: eps < 0)
  const double kappa = std::sqrt(-2 * eps);
  const double L = 8.0; const int ne = 6, np = 9, nv = 28;
  auto g = make_fegrid(ne, np, L);
  const int N = g.N; const size_t N3 = (size_t)N * N * N;
  std::vector<double> rnv, rwv; gauss_legendre(nv, -1, 1, rnv, rwv);
  auto ytg = intti::make_tgrid(intti::yukawa(kappa));

  // exact eigenpair and the source s = V psi
  std::vector<double> psi(N3), s(N3);
  for (int ix = 0; ix < N; ++ix)
    for (int iy = 0; iy < N; ++iy)
      for (int iz = 0; iz < N; ++iz) {
        const double x = g.xnode[ix], y = g.xnode[iy], z = g.xnode[iz], r2 = x*x+y*y+z*z;
        const double p = std::exp(-a * r2);
        const size_t idx = ((size_t)ix*N+iy)*N+iz;
        psi[idx] = p;
        s[idx] = (2 * a * a * r2 - C) * p; // V(r) psi(r)
      }

  // one Green's-function step: psi_new = -2 G_kappa(s) = -(1/(2 pi)) YukawaDAGE(s)
  // (YukawaDAGE gives e^{-kr}/r * s = 4 pi G_kappa * s)
  auto Ds = dage(g, rnv, rwv, ytg, s);
  const double pref = -1.0 / (2 * M_PI);
  double worst = 0, scale = 0;
  for (size_t k = 0; k < N3; ++k) {
    const double pn = pref * Ds[k];
    worst = std::max(worst, std::abs(pn - psi[k]));
    scale = std::max(scale, std::abs(psi[k]));
  }
  std::printf("== grid Helmholtz solver step: psi = -2 G_kappa(V psi) vs exact eigenpair ==\n");
  std::printf("   shifted harmonic well V=2a^2 r^2 - C (a=%.2f C=%.2f), eps=%.3f, kappa=%.4f\n",
              a, C, eps, kappa);
  std::printf("   grid: %d elem x deg %d (N=%d/axis), nt=%d, GL(%d)\n", ne, np-1, N, ytg.n(), nv);
  std::printf("   max|psi_new - psi_exact| = %.3e   rel = %.3e\n", worst, worst / scale);
  return worst / scale < 1e-5 ? 0 : 1;
}
