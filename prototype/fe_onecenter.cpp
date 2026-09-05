// M-FE one-center prototype: represent a Gaussian basis on a tensorial
// (Cartesian-product) real-space grid and evaluate integrals by grid quadrature,
// validating against the ANALYTIC t-quadrature engine (eri_quartet) and closed
// forms. Demonstrates the two claims of the M-FE plan for the one-center case:
//   (1) the Gaussian basis is represented on the tensorial grid to numerical
//       exactness  -> overlap by pure grid quadrature matches analytic;
//   (2) (pq|rs) via the tensorial grid + the Gaussian-transform (t-quadrature)
//       Coulomb, with the Losilla delta-tail for the large-t/short-range part,
//       reproduces the analytic ERI.
//
// The Coulomb factorizes by Cartesian dimension (White eqn 22 / DAGE): with
//   1/r12 = sum_i w_i exp(-t_i^2 |r1-r2|^2)   (w_i carries 2/sqrt(pi), Mobius t)
// and separable pair densities rho(r) = X(x)Y(y)Z(z),
//   (pq|rs) = sum_i w_i A_x(t_i) A_y(t_i) A_z(t_i),
//   A_d(t)  = int int rho_bra^d(x) exp(-t^2 (x-x')^2) rho_ket^d(x') dx dx'.
// The inner integral uses a NEAR/FAR t-split (the key to machine precision for
// all t): small t (broad kernel) -> direct x'-grid; large t (sharp kernel) ->
// t-adapted x'=x-v/t so exp(-v^2) is flat, sampling the density at near-constant
// argument. Result: (pq|rs) matches eri_quartet to ~1e-15, no delta tail needed.
// (In the real FEM the density is a piecewise polynomial -> smooth in v for all
// t -> the t-adapted branch alone is t-uniform; the split here is only because
// this prototype represents the density as a Gaussian.)

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include <Kokkos_Core.hpp>

#include "intti/fock.hpp"
#include "intti/quartet.hpp"
#include "intti/tgrid.hpp"

using intti::PrimitiveShell;

// Gauss-Legendre nodes/weights on [a,b] (Newton on the Legendre polynomial).
static void gauss_legendre(int n, double a, double b, std::vector<double> &x,
                           std::vector<double> &w) {
  x.assign(n, 0.0);
  w.assign(n, 0.0);
  for (int i = 0; i < n; ++i) {
    double xi = std::cos(M_PI * (i + 0.75) / (n + 0.5)); // asymptotic guess
    double dp = 0;
    for (int it = 0; it < 100; ++it) {
      double p0 = 1, p1 = xi;
      for (int k = 2; k <= n; ++k) {
        double p2 = ((2 * k - 1) * xi * p1 - (k - 1) * p0) / k;
        p0 = p1;
        p1 = p2;
      }
      dp = n * (xi * p1 - p0) / (xi * xi - 1);
      double dx = -p1 / dp;
      xi += dx;
      if (std::abs(dx) < 1e-15) break;
    }
    x[i] = 0.5 * (a + b) + 0.5 * (b - a) * xi;
    w[i] = (b - a) / ((1 - xi * xi) * dp * dp);
  }
}

// 1D Gaussian moment integral int u^m exp(-a u^2) du (analytic).
static double gauss_moment(int m, double a) {
  if (m % 2) return 0.0;
  double df = 1.0; // (m-1)!!
  for (int k = m - 1; k > 0; k -= 2) df *= k;
  return df * std::sqrt(M_PI) / (std::pow(2.0, m / 2) * std::pow(a, (m + 1) / 2.0));
}

// Element-pair kernel matrix element M_ij(t) = int_[a,b] int_[c,d] x^i x'^j
// exp(-t^2 (x-x')^2) dx' dx, by two quadratures:
//   naive:  fixed Gauss-Legendre in x and x' -> the exp is a knife's edge at
//           large t and the fixed nodes miss it;
//   t-adapt: substitute x' = x - v/t so the kernel becomes exp(-v^2) (FLAT for
//           all t); GL in v samples the polynomial at near-constant argument.
static double Mij_naive(int i, int j, double t, double a, double b, double c, double d, int nq) {
  std::vector<double> xg, xw, yg, yw;
  gauss_legendre(nq, a, b, xg, xw);
  gauss_legendre(nq, c, d, yg, yw);
  double s = 0;
  for (int p = 0; p < nq; ++p)
    for (int q = 0; q < nq; ++q) {
      const double du = xg[p] - yg[q];
      s += xw[p] * std::pow(xg[p], i) * yw[q] * std::pow(yg[q], j) * std::exp(-t * t * du * du);
    }
  return s;
}
static double Mij_tadapt(int i, int j, double t, double a, double b, double c, double d, int nq,
                         int nv) {
  std::vector<double> xg, xw;
  gauss_legendre(nq, a, b, xg, xw);
  double s = 0;
  for (int p = 0; p < nq; ++p) {
    const double x = xg[p];
    // v = t (x - x'); x' in [c,d] -> v in [t(x-d), t(x-c)]; exp(-v^2) confines |v|<~8
    double lo = t * (x - d), hi = t * (x - c);
    lo = std::max(lo, -8.0);
    hi = std::min(hi, 8.0);
    if (hi <= lo) continue;
    std::vector<double> vg, vw;
    gauss_legendre(nv, lo, hi, vg, vw);
    double inner = 0;
    for (int g = 0; g < nv; ++g) {
      const double xp = x - vg[g] / t;
      inner += vw[g] * std::pow(xp, j) * std::exp(-vg[g] * vg[g]);
    }
    s += xw[p] * std::pow(x, i) * (inner / t);
  }
  return s;
}

static void quadrature_study() {
  std::printf("== element-pair kernel M_ij(t): naive vs t-adapted quadrature ==\n");
  const double a = 0.5, b = 1.5, c = 0.6, d = 1.6; // two finite elements
  for (int i : {0, 2}) {
    const int j = i;
    std::printf("   polynomial x^%d x'^%d on elements [%.1f,%.1f]x[%.1f,%.1f]\n", i, j, a, b, c, d);
    std::printf("       t     naive(40)      t-adapt(40)\n");
    for (double t : {1.0, 4.0, 16.0, 64.0, 256.0}) {
      const double ref = Mij_tadapt(i, j, t, a, b, c, d, 160, 160); // converged reference
      const double en = std::abs(Mij_naive(i, j, t, a, b, c, d, 40) - ref);
      const double et = std::abs(Mij_tadapt(i, j, t, a, b, c, d, 40, 40) - ref);
      std::printf("   %7.1f   %.3e     %.3e\n", t, en / (std::abs(ref) + 1e-300),
                  et / (std::abs(ref) + 1e-300));
    }
  }
}

int run();
int main() {
  Kokkos::initialize();
  quadrature_study();
  std::printf("\n");
  int rc = run();
  Kokkos::finalize();
  return rc;
}

int run() {
  const double origin[3] = {0, 0, 0};
  // ---- overlap: tensorial grid vs analytic (numerical-exactness of the rep) ----
  std::printf("== overlap: tensorial grid vs analytic (one center) ==\n");
  {
    const double L = 9.0;
    std::vector<double> u, w;
    gauss_legendre(220, -L, L, u, w);
    const int N = (int)u.size();
    // a set of primitives (alpha, l-powers) on the origin
    struct Prim { double a; int l[3]; };
    std::vector<Prim> ps = {{1.3, {0, 0, 0}}, {0.7, {1, 0, 0}}, {0.4, {2, 0, 0}},
                            {0.9, {1, 1, 0}}};
    double worst_abs = 0, worst_rel = 0, scale = 0;
    for (auto &A : ps)
      for (auto &B : ps) {
        const double p = A.a + B.a;
        double grid = 1.0, exact = 1.0;
        for (int d = 0; d < 3; ++d) {
          const int m = A.l[d] + B.l[d];
          double s = 0;
          for (int j = 0; j < N; ++j) s += w[j] * std::pow(u[j], m) * std::exp(-p * u[j] * u[j]);
          grid *= s;
          exact *= gauss_moment(m, p);
        }
        worst_abs = std::max(worst_abs, std::abs(grid - exact));
        scale = std::max(scale, std::abs(exact));
        if (std::abs(exact) > 1e-14) // skip parity-zero components in the ratio
          worst_rel = std::max(worst_rel, std::abs(grid - exact) / std::abs(exact));
      }
    std::printf("   max abs error %.3e (scale %.3e), max rel error (nonzero) %.3e\n",
                worst_abs, scale, worst_rel);
  }

  // ---- Coulomb (pq|rs): tensorial grid + t-quadrature vs eri_quartet ----
  std::printf("\n== (pq|rs): tensorial grid + near/far-split t-adapted quadrature vs eri_quartet ==\n");
  // analytic-oracle Coulomb grid (accurate Mobius, no truncation)
  auto oracle = intti::make_tgrid(intti::coulomb());

  struct Quartet { const char *name; PrimitiveShell<double> A, B, C, D; };
  auto sh = [&](double a, int l) { return PrimitiveShell<double>{a, {0, 0, 0}, l}; };
  std::vector<Quartet> qs = {
      {"(ss|ss)", sh(1.3, 0), sh(0.8, 0), sh(1.1, 0), sh(0.6, 0)},
      {"(ps|ss)", sh(1.3, 1), sh(0.8, 0), sh(1.1, 0), sh(0.6, 0)},
      {"(pp|ss)", sh(1.3, 1), sh(0.8, 1), sh(1.1, 0), sh(0.6, 0)},
      {"(pp|pp)", sh(1.0, 1), sh(0.7, 1), sh(0.9, 1), sh(0.6, 1)},
  };

  const double L = 8.0;
  const int MP = 5; // per-axis powers 0..4 (up to d pair densities)
  // Mobius t-grid (full [0,inf), no truncation / no delta tail).
  auto grid = intti::make_tgrid(intti::coulomb());
  const int nt = grid.n();
  // near/far t-split of the inner integral, needed because the DENSITY here is a
  // Gaussian (in the real FEM the density is a piecewise polynomial, smooth in
  // the transformed variable, so t-adaptation alone is t-uniform):
  //   small t (broad kernel): direct x'-grid inner (density resolved);
  //   large t (sharp kernel): t-adapted inner x'=x-v/t (kernel flat in v).
  // The validity ranges overlap, so switching gives machine precision for all t.
  const double t_switch = 2.0;
  const int nq = 160, nv = 96;
  const double vmax = 8.0;
  std::vector<double> u, w, vg, vw;
  gauss_legendre(nq, -L, L, u, w);
  gauss_legendre(nv, -vmax, vmax, vg, vw);
  std::printf("   Mobius nt=%d, outer GL(%d) on [-%.0f,%.0f]; inner: direct(t<%.0f) / "
              "t-adapted GL(%d) (t>=%.0f)\n",
              nt, nq, L, L, t_switch, nv, t_switch);
  for (auto &q : qs) {
    const int la = q.A.l, lb = q.B.l, lc = q.C.l, ld = q.D.l;
    const int na = intti::ncart(la), nb = intti::ncart(lb);
    const int nc = intti::ncart(lc), nd = intti::ncart(ld);
    std::vector<double> blk((size_t)na * nb * nc * nd);
    intti::eri_quartet(intti::make_pair(q.A, q.B), intti::make_pair(q.C, q.D), oracle,
                       blk.data());
    const double pbra = q.A.alpha + q.B.alpha, pket = q.C.alpha + q.D.alpha;
    // density samples on the x-grid (for the small-t direct inner)
    std::vector<std::vector<double>> gket(MP, std::vector<double>(nq));
    for (int m = 0; m < MP; ++m)
      for (int k = 0; k < nq; ++k)
        gket[m][k] = std::pow(u[k], m) * std::exp(-pket * u[k] * u[k]);
    std::vector<double> A((size_t)MP * MP * nt, 0.0);
    for (int mb = 0; mb < MP; ++mb)
      for (int mk = 0; mk < MP; ++mk) {
#pragma omp parallel for schedule(dynamic)
        for (int it = 0; it < nt; ++it) {
          const double t = grid.t[it], t2 = t * t;
          double Aa = 0;
          for (int p = 0; p < nq; ++p) {
            const double x = u[p];
            const double rb = std::pow(x, mb) * std::exp(-pbra * x * x);
            double inner = 0;
            if (t < t_switch) { // direct x'-grid (kernel broad)
              for (int k = 0; k < nq; ++k) {
                const double du = x - u[k];
                inner += w[k] * gket[mk][k] * std::exp(-t2 * du * du);
              }
            } else { // t-adapted (kernel sharp): (1/t) int rho_ket(x-v/t) e^{-v^2} dv
              for (int g = 0; g < nv; ++g) {
                const double xp = x - vg[g] / t;
                inner += vw[g] * std::pow(xp, mk) * std::exp(-pket * xp * xp - vg[g] * vg[g]);
              }
              inner /= t;
            }
            Aa += w[p] * rb * inner;
          }
          A[((size_t)mb * MP + mk) * nt + it] = Aa;
        }
      }
    double worst = 0, scale = 0;
    for (int ka = 0; ka < na; ++ka) {
      int a3[3]; intti::cart_comp(la, ka, a3[0], a3[1], a3[2]);
      for (int kb = 0; kb < nb; ++kb) {
        int b3[3]; intti::cart_comp(lb, kb, b3[0], b3[1], b3[2]);
        for (int kc = 0; kc < nc; ++kc) {
          int c3[3]; intti::cart_comp(lc, kc, c3[0], c3[1], c3[2]);
          for (int kd = 0; kd < nd; ++kd) {
            int d3[3]; intti::cart_comp(ld, kd, d3[0], d3[1], d3[2]);
            const int mbx = a3[0] + b3[0], mby = a3[1] + b3[1], mbz = a3[2] + b3[2];
            const int mkx = c3[0] + d3[0], mky = c3[1] + d3[1], mkz = c3[2] + d3[2];
            const double *Ax = &A[((size_t)mbx * MP + mkx) * nt];
            const double *Ay = &A[((size_t)mby * MP + mky) * nt];
            const double *Az = &A[((size_t)mbz * MP + mkz) * nt];
            double val = 0;
            for (int it = 0; it < nt; ++it) val += grid.w[it] * Ax[it] * Ay[it] * Az[it];
            const double ref = blk[((size_t)ka * nb + kb) * nc * nd + (size_t)kc * nd + kd];
            worst = std::max(worst, std::abs(val - ref));
            scale = std::max(scale, std::abs(ref));
          }
        }
      }
    }
    std::printf("      %-8s  max|grid-eri| = %.3e   rel = %.3e\n", q.name, worst,
                worst / (scale + 1e-300));
  }
  return 0;
}
