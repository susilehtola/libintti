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
//   1/r12 = sum_i w_i exp(-t_i^2 |r1-r2|^2)   (+ delta tail; w_i carries 2/sqrt(pi))
// and separable pair densities rho(r) = X(x)Y(y)Z(z),
//   (pq|rs) = sum_i w_i A_x(t_i) A_y(t_i) A_z(t_i) + (pi/t_c^2) O,
//   A_d(t)  = sum_{jk} w_j rho_bra^d(u_j) exp(-t^2 (u_j-u_k)^2) w_k rho_ket^d(u_k),
//   O       = prod_d sum_j w_j rho_bra^d(u_j) rho_ket^d(u_j)   (four-orbital overlap).
// The large-t nodes need spatial resolution h <~ 1/t_c; beyond t_c the tail term
// supplies the short-range remainder (this is the "large-t -> moment" handling).

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
  std::printf("\n== (pq|rs): tensorial grid + t-quadrature + delta tail vs analytic eri_quartet ==\n");
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
  for (double tc : {15.0, 30.0, 60.0}) {
    // spatial resolution matched to t_c: h ~ 1/tc (rule t_c*h <~ 1)
    const int N = std::min(1000, (int)std::ceil(2 * L * tc * 0.9));
    std::vector<double> u, w;
    gauss_legendre(N, -L, L, u, w);
    intti::TGridSpec<double> spec;
    spec.mapping = intti::TMapping::LinLog;
    spec.t_c = tc;
    spec.tail_order = 0; // leading Losilla delta tail (pi/t_c^2)
    auto grid = intti::make_tgrid(intti::coulomb(), spec);
    const int nt = grid.n();
    const double tail = grid.tail_coeff; // = pi/t_c^2

    std::printf("   t_c=%5.1f  N=%4d  nt=%3d\n", tc, N, nt);
    for (auto &q : qs) {
      const int la = q.A.l, lb = q.B.l, lc = q.C.l, ld = q.D.l;
      const int na = intti::ncart(la), nb = intti::ncart(lb);
      const int nc = intti::ncart(lc), nd = intti::ncart(ld);
      std::vector<double> blk((size_t)na * nb * nc * nd);
      intti::eri_quartet(intti::make_pair(q.A, q.B), intti::make_pair(q.C, q.D), oracle,
                         blk.data());
      const double pbra = q.A.alpha + q.B.alpha, pket = q.C.alpha + q.D.alpha;
      // per-axis density samples for each power (grid shared across x,y,z)
      std::vector<std::vector<double>> gbra(MP, std::vector<double>(N)),
          gket(MP, std::vector<double>(N));
      for (int m = 0; m < MP; ++m)
        for (int j = 0; j < N; ++j) {
          const double um = std::pow(u[j], m);
          gbra[m][j] = um * std::exp(-pbra * u[j] * u[j]);
          gket[m][j] = um * std::exp(-pket * u[j] * u[j]);
        }
      // A[mb][mk][it] (axis-independent) and the per-axis overlap O1[mb][mk]
      std::vector<double> A((size_t)MP * MP * nt, 0.0), O1((size_t)MP * MP, 0.0);
      for (int mb = 0; mb < MP; ++mb)
        for (int mk = 0; mk < MP; ++mk) {
          double o = 0;
          for (int j = 0; j < N; ++j) o += w[j] * gbra[mb][j] * gket[mk][j];
          O1[mb * MP + mk] = o;
#pragma omp parallel for schedule(dynamic)
          for (int it = 0; it < nt; ++it) {
            const double t2 = grid.t[it] * grid.t[it];
            double Aa = 0;
            for (int j = 0; j < N; ++j) {
              double inner = 0;
              const double uj = u[j];
              for (int k = 0; k < N; ++k) {
                const double du = uj - u[k];
                inner += w[k] * gket[mk][k] * std::exp(-t2 * du * du);
              }
              Aa += w[j] * gbra[mb][j] * inner;
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
              val += tail * O1[mbx * MP + mkx] * O1[mby * MP + mky] * O1[mbz * MP + mkz];
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
  }
  return 0;
}
