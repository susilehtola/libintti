// M-FE step-3: 3D t-adapted DAGE with per-axis FE interpolation.
//
// Coulomb potential of a GENERAL 3D density on a tensorial piecewise-polynomial
// FE grid: V(r1) = sum_t w_t ∫ rho(r2) e^{-t^2|r1-r2|^2} dr2. The kernel
// factorizes per axis, so at each t-node the 3D convolution is three successive
// 1D convolutions on the tensor (along z, y, x). Each 1D convolution uses the
// t-adapted substitution u2 = u1 - v/t (kernel exp(-v^2) flat for all t) and
// evaluates the FE function at the off-grid points u1 - v/t by barycentric
// Lagrange interpolation within the source element (far elements screen via the
// clamped v-range). No delta tail; machine-exact for all t.
//
// This works for ANY density (separability is in the kernel, not the density),
// so it is the engine for the co-density exchange (form g_qi = chi_q phi_i as one
// 3D tensor, DAGE it once). Validated here on a NON-separable density (a sum of
// s-Gaussians) against the analytic Coulomb sum.

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
  std::vector<double> be, rn, rw, bw, xnode, xw; // boundaries, ref nodes/wts/bary, global nodes/wts
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

// t-adapted 1D convolution of a line of FE nodal values with exp(-t^2(u1-u2)^2)
static void conv1d(const FEGrid &g, const std::vector<double> &rnv,
                   const std::vector<double> &rwv, const double *in, double t, double *out) {
  const int nv = (int)rnv.size();
  for (int jo = 0; jo < g.N; ++jo) {
    const double u1 = g.xnode[jo];
    double acc = 0;
    for (int eB = 0; eB < g.ne; ++eB) {
      double lo = t * (u1 - g.be[eB + 1]), hi = t * (u1 - g.be[eB]);
      lo = std::max(lo, -8.0); hi = std::min(hi, 8.0);
      if (hi <= lo) continue; // far element screened
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

int run();
int main() { Kokkos::initialize(); int rc = run(); Kokkos::finalize(); return rc; }

int run() {
  const double L = 8.0;
  const int ne = 5, np = 8, nv = 24;
  auto g = make_fegrid(ne, np, L);
  const int N = g.N;
  std::vector<double> rnv, rwv;
  gauss_legendre(nv, -1, 1, rnv, rwv);
  auto tgrid = intti::make_tgrid(intti::coulomb());
  const int nt = tgrid.n();

  // NON-separable density: rho(r) = sum_k c_k e^{-a_k r^2}
  const int nk = 2;
  const double ck[nk] = {1.0, 0.7}, ak[nk] = {1.2, 0.5};
  auto rhoval = [&](double x, double y, double z) {
    double s = 0;
    for (int k = 0; k < nk; ++k) s += ck[k] * std::exp(-ak[k] * (x*x + y*y + z*z));
    return s;
  };
  const size_t N3 = (size_t)N * N * N;
  std::vector<double> rho(N3);
  for (int ix = 0; ix < N; ++ix)
    for (int iy = 0; iy < N; ++iy)
      for (int iz = 0; iz < N; ++iz)
        rho[((size_t)ix * N + iy) * N + iz] = rhoval(g.xnode[ix], g.xnode[iy], g.xnode[iz]);

  // V = DAGE(rho): sum_t w_t * (conv_x conv_y conv_z rho); accumulate J=(rho|V) directly
  std::vector<double> work(N3), tmp(N3), V(N3, 0.0);
  std::vector<double> lineIn(N), lineOut(N);
  for (int it = 0; it < nt; ++it) {
    const double t = tgrid.t[it];
    work = rho;
    // conv along z (contiguous)
#pragma omp parallel for collapse(2) firstprivate(lineOut) schedule(dynamic)
    for (int ix = 0; ix < N; ++ix)
      for (int iy = 0; iy < N; ++iy) {
        double *ln = &work[((size_t)ix * N + iy) * N];
        std::vector<double> o(N);
        conv1d(g, rnv, rwv, ln, t, o.data());
        for (int iz = 0; iz < N; ++iz) ln[iz] = o[iz];
      }
    // conv along y (stride N)
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ix = 0; ix < N; ++ix)
      for (int iz = 0; iz < N; ++iz) {
        std::vector<double> in(N), o(N);
        for (int iy = 0; iy < N; ++iy) in[iy] = work[((size_t)ix * N + iy) * N + iz];
        conv1d(g, rnv, rwv, in.data(), t, o.data());
        for (int iy = 0; iy < N; ++iy) work[((size_t)ix * N + iy) * N + iz] = o[iy];
      }
    // conv along x (stride N^2)
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

  // J = (rho|rho) = sum_ijk W_i W_j W_k rho V
  double J = 0;
  for (int ix = 0; ix < N; ++ix)
    for (int iy = 0; iy < N; ++iy)
      for (int iz = 0; iz < N; ++iz) {
        const double W = g.xw[ix] * g.xw[iy] * g.xw[iz];
        const size_t idx = ((size_t)ix * N + iy) * N + iz;
        J += W * rho[idx] * V[idx];
      }

  // analytic (rho|rho) = sum_kl c_k c_l * 2 pi^{5/2} / (a_k a_l sqrt(a_k+a_l))
  const double pi = M_PI;
  double Jref = 0;
  for (int k = 0; k < nk; ++k)
    for (int l = 0; l < nk; ++l)
      Jref += ck[k] * ck[l] * 2 * std::pow(pi, 2.5) / (ak[k] * ak[l] * std::sqrt(ak[k] + ak[l]));

  std::printf("== 3D t-adapted DAGE (general non-separable density) ==\n");
  std::printf("   grid: %d elements x degree %d per axis (N=%d/axis), nt=%d, inner GL(%d)\n",
              ne, np - 1, N, nt, nv);
  std::printf("   J(rho|rho) grid = %.12e\n   analytic       = %.12e\n", J, Jref);
  std::printf("   rel error = %.3e\n", std::abs(J - Jref) / Jref);
  return std::abs(J - Jref) / Jref < 1e-9 ? 0 : 1;
}
