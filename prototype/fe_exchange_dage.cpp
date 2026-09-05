// M-FE: GENUINE seminumerical exchange -- form each co-density g_qi = chi_q phi_i
// as ONE 3D grid tensor and take a SINGLE 3D DAGE Coulomb solve per (q,i), then
//   K_pq = occ_scale sum_i (g_pi | g_qi) = occ_scale sum_i int g_pi(r) V_qi(r) dr,
//   V_qi = DAGE(g_qi).
// Unlike fe_exchange_onecenter.cpp (which validated the assembly via the O(Nbf^2)
// r,s expansion of separable pair densities), this is the production form: the
// co-density is a non-separable 3D function and the 3D t-adapted DAGE (step-3)
// handles it directly. Cost = nocc*Nbf DAGE solves. Validated vs exchange_build.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <Kokkos_Core.hpp>

#include "intti/fock.hpp"
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
// V = DAGE(rho) on the 3D grid: sum_t w_t (conv_x conv_y conv_z rho)
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
  // small s-basis on one center; model 1 occupied orbital
  std::vector<PrimitiveShell<double>> shells = {
      {1.4, {0, 0, 0}, 0}, {0.7, {0, 0, 0}, 0}, {0.4, {0, 0, 0}, 0}};
  auto basis = intti::make_basis(shells);
  const int n = basis.nao, nocc = 1;
  const double occ_scale = 2.0;
  std::mt19937 rng(11); std::normal_distribution<double> nd;
  std::vector<double> C((size_t)n * nocc);
  for (auto &c : C) c = nd(rng);
  std::vector<double> D((size_t)n * n, 0.0);
  for (int p = 0; p < n; ++p)
    for (int q = 0; q < n; ++q) { double s = 0;
      for (int i = 0; i < nocc; ++i) s += C[p*nocc+i]*C[q*nocc+i];
      D[p*n+q] = occ_scale * s; }

  auto tg = intti::make_tgrid(intti::coulomb());
  std::vector<double> Kref((size_t)n * n);
  intti::exchange_build(basis, D.data(), tg, Kref.data(), 0.0);

  // FE grid
  const double L = 8.0; const int ne = 5, np = 8, nv = 24;
  auto g = make_fegrid(ne, np, L);
  const int N = g.N; const size_t N3 = (size_t)N * N * N;
  std::vector<double> rnv, rwv; gauss_legendre(nv, -1, 1, rnv, rwv);

  // basis functions on the 3D grid (s: separable, but stored as full tensors)
  auto alpha = [&](int r) { return shells[r].alpha; };
  std::vector<std::vector<double>> chi(n, std::vector<double>(N3));
  for (int r = 0; r < n; ++r)
    for (int ix = 0; ix < N; ++ix)
      for (int iy = 0; iy < N; ++iy)
        for (int iz = 0; iz < N; ++iz) {
          const double x = g.xnode[ix], y = g.xnode[iy], z = g.xnode[iz];
          chi[r][((size_t)ix*N+iy)*N+iz] = std::exp(-alpha(r) * (x*x+y*y+z*z));
        }

  std::vector<double> Kco((size_t)n * n, 0.0);
  std::vector<double> W3(N3);
  for (int ix = 0; ix < N; ++ix)
    for (int iy = 0; iy < N; ++iy)
      for (int iz = 0; iz < N; ++iz)
        W3[((size_t)ix*N+iy)*N+iz] = g.xw[ix]*g.xw[iy]*g.xw[iz];

  for (int i = 0; i < nocc; ++i) {
    // phi_i on the grid = sum_r C_ri chi_r  (non-separable 3D tensor)
    std::vector<double> phi(N3, 0.0);
    for (int r = 0; r < n; ++r) { const double c = C[r*nocc+i];
      for (size_t k = 0; k < N3; ++k) phi[k] += c * chi[r][k]; }
    for (int q = 0; q < n; ++q) {
      // co-density g_qi = chi_q * phi_i ; one DAGE solve -> V_qi
      std::vector<double> gq(N3);
      for (size_t k = 0; k < N3; ++k) gq[k] = chi[q][k] * phi[k];
      auto Vq = dage(g, rnv, rwv, tg, gq);
      for (int p = 0; p < n; ++p) {
        double kpq = 0;
        for (size_t k = 0; k < N3; ++k) kpq += W3[k] * chi[p][k] * phi[k] * Vq[k];
        Kco[p*n+q] += occ_scale * kpq;
      }
    }
  }

  double worst = 0, scale = 0;
  for (size_t k = 0; k < Kco.size(); ++k) {
    worst = std::max(worst, std::abs(Kco[k] - Kref[k]));
    scale = std::max(scale, std::abs(Kref[k]));
  }
  std::printf("== seminumerical exchange (co-density + 3D DAGE) vs exchange_build ==\n");
  std::printf("   s-basis n=%d nocc=%d; grid %d elem x deg %d (N=%d/axis)\n", n, nocc, ne, np-1, N);
  std::printf("   max|Kco - Kref| = %.3e   rel = %.3e   (%d DAGE solves)\n",
              worst, worst / (scale + 1e-300), n * nocc);
  return worst / (scale + 1e-300) < 1e-3 ? 0 : 1;
}
