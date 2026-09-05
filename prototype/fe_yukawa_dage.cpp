// M-FE toward the grid solver: the Helmholtz Green's-function apply on the FE
// grid. G_kappa = e^{-kappa r}/(4 pi r) = Yukawa/(4 pi); applying it to a grid
// function is the SAME 3D t-adapted DAGE as the Coulomb case, only with the
// YUKAWA t-grid (the exp(-kappa^2/4t^2) weight is already in make_tgrid(yukawa)).
// This is the core operator of the real-space SCF orbital update psi <- -2 G_k V
// psi (the M-HK capstone), now on the FE grid.
//
// Validated: (rho | e^{-kappa r}/r | rho) for a non-separable density (sum of
// s-Gaussians) via the FE-grid Yukawa-DAGE vs the analytic Yukawa integral from
// eri_quartet (libintti's validated analytic Yukawa ERI).

#include <cmath>
#include <cstdio>
#include <vector>

#include <Kokkos_Core.hpp>

#include "intti/quartet.hpp"
#include "intti/tgrid.hpp"

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
  const double kappa = 1.0;
  // non-separable density rho = sum_k c_k e^{-a_k r^2}
  const int nk = 2; const double ck[nk] = {1.0, 0.7}, ak[nk] = {1.2, 0.5};

  // FE-grid Yukawa-DAGE
  const double L = 8.0; const int ne = 6, np = 9, nv = 28;
  auto g = make_fegrid(ne, np, L);
  const int N = g.N; const size_t N3 = (size_t)N * N * N;
  std::vector<double> rnv, rwv; gauss_legendre(nv, -1, 1, rnv, rwv);
  auto ytg = intti::make_tgrid(intti::yukawa(kappa)); // exp(-kappa r)/r t-grid

  std::vector<double> rho(N3);
  for (int ix = 0; ix < N; ++ix)
    for (int iy = 0; iy < N; ++iy)
      for (int iz = 0; iz < N; ++iz) {
        const double x = g.xnode[ix], y = g.xnode[iy], z = g.xnode[iz], r2 = x*x+y*y+z*z;
        double s = 0; for (int k = 0; k < nk; ++k) s += ck[k] * std::exp(-ak[k] * r2);
        rho[((size_t)ix*N+iy)*N+iz] = s;
      }
  auto V = dage(g, rnv, rwv, ytg, rho); // V = (e^{-kappa r}/r) * rho
  double J = 0;
  for (int ix = 0; ix < N; ++ix)
    for (int iy = 0; iy < N; ++iy)
      for (int iz = 0; iz < N; ++iz) {
        const double W = g.xw[ix]*g.xw[iy]*g.xw[iz];
        const size_t idx = ((size_t)ix*N+iy)*N+iz;
        J += W * rho[idx] * V[idx];
      }

  // analytic (rho|Yukawa|rho) = sum_kl c_k c_l (g_k|Yukawa|g_l), g_k=e^{-a_k r^2};
  // (g_k|Yukawa|g_l) = eri_quartet(pair(s_{a_k/2},s_{a_k/2}),pair(s_{a_l/2},s_{a_l/2}))
  double Jref = 0;
  for (int k = 0; k < nk; ++k)
    for (int l = 0; l < nk; ++l) {
      PrimitiveShell<double> A{ak[k]/2,{0,0,0},0}, B{ak[l]/2,{0,0,0},0};
      double blk;
      intti::eri_quartet(intti::make_pair(A, A), intti::make_pair(B, B), ytg, &blk);
      Jref += ck[k] * ck[l] * blk;
    }

  std::printf("== FE-grid Helmholtz/Yukawa apply (G_kappa, kappa=%.1f) vs analytic ==\n", kappa);
  std::printf("   grid: %d elem x deg %d (N=%d/axis), nt=%d, GL(%d)\n", ne, np-1, N, ytg.n(), nv);
  std::printf("   (rho|e^{-kr}/r|rho) grid = %.12e\n   analytic = %.12e\n   rel error = %.3e\n",
              J, Jref, std::abs(J - Jref) / Jref);
  return std::abs(J - Jref) / Jref < 1e-6 ? 0 : 1;
}
