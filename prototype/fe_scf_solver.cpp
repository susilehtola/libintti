// M-FE / M-HK capstone: a real-space grid SCF driver via the Helmholtz
// Green's-function power iteration, converging from a WRONG guess to the ground
// state.  psi <- -2 G_kappa(V psi),  kappa = sqrt(-2 eps).  G_kappa apply = the
// 3D t-adapted DAGE with the yukawa t-grid.
//
// Energy update with NO kinetic operator: psi~ = -2 G_kappa(V psi) satisfies
// (-1/2 lap - eps_n) psi~ = -V psi, so <psi~|T|psi~> = eps_n <psi~|psi~> -
// <psi~|V psi>, and the Rayleigh quotient of psi~ is
//   eps_{n+1} = eps_n + <psi~ | V (psi~ - psi_n)> / <psi~|psi~>,
// using only V-multiplies and grid inner products.
//
// Test: shifted harmonic well V = 1/2 omega^2 r^2 - C (omega=1, C=4): exact
// ground state e^{-omega r^2/2}=e^{-r^2/2}, eps = 3/2 omega - C = -2.5. Start from
// a wrong-exponent Gaussian and iterate; eps -> -2.5 (grid-limited) and the
// wavefunction overlap with the exact ground state -> 1.

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
// reference differentiation matrix D[i][j] = dL_j/dxi at node rn[i] (barycentric)
static std::vector<double> diff_matrix(const FEGrid &g) {
  const int np = g.np;
  std::vector<double> D((size_t)np * np, 0.0);
  for (int i = 0; i < np; ++i) {
    double diag = 0;
    for (int j = 0; j < np; ++j)
      if (j != i) { D[i * np + j] = (g.bw[j] / g.bw[i]) / (g.rn[i] - g.rn[j]); diag -= D[i * np + j]; }
    D[i * np + i] = diag;
  }
  return D;
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
  const double omega = 1.0, C = 4.0;   // V = 1/2 omega^2 r^2 - C
  const double eps_exact = 1.5 * omega - C; // = -2.5
  const double aex = 0.5 * omega;       // exact ground state e^{-aex r^2}

  const double L = 8.0; const int ne = 4, np = 8, nv = 20;
  auto g = make_fegrid(ne, np, L);
  const int N = g.N; const size_t N3 = (size_t)N * N * N;
  std::vector<double> rnv, rwv; gauss_legendre(nv, -1, 1, rnv, rwv);

  std::vector<double> Vr(N3), W3(N3), psiex(N3);
  for (int ix = 0; ix < N; ++ix)
    for (int iy = 0; iy < N; ++iy)
      for (int iz = 0; iz < N; ++iz) {
        const double x = g.xnode[ix], y = g.xnode[iy], z = g.xnode[iz], r2 = x*x+y*y+z*z;
        const size_t k = ((size_t)ix*N+iy)*N+iz;
        Vr[k] = 0.5 * omega * omega * r2 - C;
        W3[k] = g.xw[ix]*g.xw[iy]*g.xw[iz];
        psiex[k] = std::exp(-aex * r2);
      }
  auto dot = [&](const std::vector<double> &f, const std::vector<double> &h) {
    double s = 0; for (size_t k = 0; k < N3; ++k) s += W3[k] * f[k] * h[k]; return s;
  };

  // wrong-exponent starting guess, normalized
  std::vector<double> psi(N3);
  for (int ix = 0; ix < N; ++ix)
    for (int iy = 0; iy < N; ++iy)
      for (int iz = 0; iz < N; ++iz) {
        const double x = g.xnode[ix], y = g.xnode[iy], z = g.xnode[iz];
        psi[((size_t)ix*N+iy)*N+iz] = std::exp(-0.8 * (x*x+y*y+z*z)); // b=0.8 != 0.5
      }
  { double nrm = std::sqrt(dot(psi, psi)); for (auto &v : psi) v /= nrm; }

  // robust Rayleigh quotient eps = <psi|T|psi> + <psi|V|psi>, T = 1/2 int |grad|^2
  // with grad from the FE differentiation matrix (no cancellation).
  auto D = diff_matrix(g);
  auto grad_line = [&](const double *in, double *out, int e0) {
    for (int e = 0; e < g.ne; ++e) {
      const double inv = 1.0 / g.hw(e);
      for (int i = 0; i < np; ++i) {
        double s = 0;
        for (int l = 0; l < np; ++l) s += D[i * np + l] * in[e * np + l];
        out[e * np + i] = inv * s;
      }
    }
    (void)e0;
  };
  auto rayleigh = [&](const std::vector<double> &p) {
    double T = 0, Vv = 0;
    std::vector<double> lin(N), lo(N);
    // d/dx
    for (int iy = 0; iy < N; ++iy)
      for (int iz = 0; iz < N; ++iz) {
        for (int ix = 0; ix < N; ++ix) lin[ix] = p[((size_t)ix*N+iy)*N+iz];
        grad_line(lin.data(), lo.data(), 0);
        for (int ix = 0; ix < N; ++ix) { const size_t k=((size_t)ix*N+iy)*N+iz; T += 0.5*W3[k]*lo[ix]*lo[ix]; }
      }
    // d/dy
    for (int ix = 0; ix < N; ++ix)
      for (int iz = 0; iz < N; ++iz) {
        for (int iy = 0; iy < N; ++iy) lin[iy] = p[((size_t)ix*N+iy)*N+iz];
        grad_line(lin.data(), lo.data(), 0);
        for (int iy = 0; iy < N; ++iy) { const size_t k=((size_t)ix*N+iy)*N+iz; T += 0.5*W3[k]*lo[iy]*lo[iy]; }
      }
    // d/dz
    for (int ix = 0; ix < N; ++ix)
      for (int iy = 0; iy < N; ++iy) {
        const double *ln = &p[((size_t)ix*N+iy)*N];
        grad_line(ln, lo.data(), 0);
        for (int iz = 0; iz < N; ++iz) { const size_t k=((size_t)ix*N+iy)*N+iz; T += 0.5*W3[k]*lo[iz]*lo[iz]; }
      }
    for (size_t k = 0; k < N3; ++k) Vv += W3[k] * Vr[k] * p[k] * p[k];
    return (T + Vv) / dot(p, p);
  };

  double eps = rayleigh(psi); // initial energy from the guess
  std::printf("== grid Helmholtz SCF: V=1/2 w^2 r^2 - C (w=%.1f C=%.1f), exact eps=%.4f ==\n",
              omega, C, eps_exact);
  std::printf("   grid: %d elem x deg %d (N=%d/axis); start guess e^{-0.8 r^2}, eps0=%.5f\n",
              ne, np-1, N, eps);
  std::printf("   iter        eps          |eps-exact|     overlap^2 with exact\n");
  double eps_prev = eps;
  for (int iter = 0; iter < 30; ++iter) {
    const double kappa = std::sqrt(-2 * eps);
    auto ytg = intti::make_tgrid(intti::yukawa(kappa));
    std::vector<double> s(N3);
    for (size_t k = 0; k < N3; ++k) s[k] = Vr[k] * psi[k];             // s = V psi_n
    auto Ds = dage(g, rnv, rwv, ytg, s);
    std::vector<double> pt(N3);
    for (size_t k = 0; k < N3; ++k) pt[k] = -1.0 / (2 * M_PI) * Ds[k]; // psi~ = -2 G_k(V psi)
    // Kalos/BSH energy update: eps += <V psi_n | psi~ - psi_n> / <psi~|psi~>
    // (for eps > eps0, ||psi~|| > 1 -> this correction is negative, pushing eps
    // down toward the ground state, and vanishes at the fixed point psi~=psi_n).
    double num = 0, den = dot(pt, pt);
    for (size_t k = 0; k < N3; ++k) num += W3[k] * s[k] * (pt[k] - psi[k]);
    const double dstep = num / den;
    eps += dstep;
    const double nrm = std::sqrt(den);
    for (size_t k = 0; k < N3; ++k) psi[k] = pt[k] / nrm;             // psi_{n+1} = psi~/||psi~||
    const double ov = dot(psi, psiex);
    const double ov2 = ov * ov / (dot(psi, psi) * dot(psiex, psiex));
    std::printf("   %3d   %14.8f   %.4e      %.10f\n", iter, eps,
                std::abs(eps - eps_exact), ov2);
    if (std::abs(dstep) < 1e-2) break; // grid-limited convergence (coarse N=32)
    eps_prev = eps;
  }
  std::printf("   converged eps = %.8f (exact %.8f, diff %.2e) -- grid-limited\n", eps, eps_exact,
              std::abs(eps - eps_exact));
  return std::abs(eps - eps_exact) < 1e-2 ? 0 : 1;
}
