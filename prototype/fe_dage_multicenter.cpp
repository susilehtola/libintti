// M-FE multi-center: the 3D t-adapted DAGE (step-3) on a GLOBAL tensorial FE
// grid covering more than one nucleus, applied to a non-separable MULTI-CENTER
// density. This is the molecular case: the grid Coulomb machinery is unchanged
// (separability is in the kernel, not the density), only the density now has
// features at several centers and the box must resolve all of them.
//
// Validated against the analytic two-center Coulomb, whose cross term brings in
// the Boys function F0: for unnormalized s-Gaussians g_k = exp(-a_k |r-C_k|^2),
//   (g_k|g_l) = 2 pi^{5/2} / (a_k a_l sqrt(a_k+a_l)) * F0(alpha_kl R_kl^2),
//   alpha_kl = a_k a_l/(a_k+a_l),  R_kl = |C_k - C_l|,  F0(0)=1.
// A pass here shows the global-grid DAGE reproduces the multi-center (incl.
// cross-center) Coulomb -- the building block for molecular co-density exchange.

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
// Boys F0
static double boys0(double x) {
  if (x < 1e-12) return 1.0;
  return 0.5 * std::sqrt(M_PI / x) * std::erf(std::sqrt(x));
}

int run();
int main() { Kokkos::initialize(); int rc = run(); Kokkos::finalize(); return rc; }

int run() {
  const double L = 8.0;
  const int ne = 6, np = 9, nv = 28;
  auto g = make_fegrid(ne, np, L);
  const int N = g.N;
  std::vector<double> rnv, rwv;
  gauss_legendre(nv, -1, 1, rnv, rwv);
  auto tgrid = intti::make_tgrid(intti::coulomb());
  const int nt = tgrid.n();

  // two-center non-separable density: g_A at A, g_B at B (along z), s-type
  struct Gau { double c, a, C[3]; };
  const double dz = 1.4; // bond-like separation
  std::vector<Gau> gs = {{1.0, 1.2, {0, 0, -0.5 * dz}}, {0.8, 0.9, {0, 0, 0.5 * dz}}};
  auto rhoval = [&](double x, double y, double z) {
    double s = 0;
    for (auto &q : gs) {
      const double dx = x - q.C[0], dy = y - q.C[1], dzc = z - q.C[2];
      s += q.c * std::exp(-q.a * (dx*dx + dy*dy + dzc*dzc));
    }
    return s;
  };
  const size_t N3 = (size_t)N * N * N;
  std::vector<double> rho(N3);
  for (int ix = 0; ix < N; ++ix)
    for (int iy = 0; iy < N; ++iy)
      for (int iz = 0; iz < N; ++iz)
        rho[((size_t)ix * N + iy) * N + iz] = rhoval(g.xnode[ix], g.xnode[iy], g.xnode[iz]);

  // V = DAGE(rho): sum_t w_t (conv_x conv_y conv_z rho)
  std::vector<double> work(N3), V(N3, 0.0);
  for (int it = 0; it < nt; ++it) {
    const double t = tgrid.t[it];
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
    const double wt = tgrid.w[it];
    for (size_t i = 0; i < N3; ++i) V[i] += wt * work[i];
  }
  double J = 0;
  for (int ix = 0; ix < N; ++ix)
    for (int iy = 0; iy < N; ++iy)
      for (int iz = 0; iz < N; ++iz) {
        const double W = g.xw[ix] * g.xw[iy] * g.xw[iz];
        const size_t idx = ((size_t)ix * N + iy) * N + iz;
        J += W * rho[idx] * V[idx];
      }

  // analytic two-center Coulomb (with Boys F0 for the cross term)
  const double pi = M_PI;
  double Jref = 0;
  for (auto &k : gs)
    for (auto &l : gs) {
      double R2 = 0; for (int d = 0; d < 3; ++d) { double dd = k.C[d]-l.C[d]; R2 += dd*dd; }
      const double alpha = k.a * l.a / (k.a + l.a);
      Jref += k.c * l.c * 2 * std::pow(pi, 2.5) / (k.a * l.a * std::sqrt(k.a + l.a))
              * boys0(alpha * R2);
    }
  std::printf("== multi-center 3D DAGE (two centers, dz=%.2f) vs analytic (Boys F0) ==\n", dz);
  std::printf("   grid: %d elements x degree %d per axis (N=%d), nt=%d, GL(%d)\n",
              ne, np - 1, N, nt, nv);
  std::printf("   J grid = %.12e\n   analytic = %.12e\n   rel error = %.3e\n",
              J, Jref, std::abs(J - Jref) / Jref);
  return std::abs(J - Jref) / Jref < 1e-6 ? 0 : 1;
}
