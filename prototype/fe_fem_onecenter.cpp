// M-FE step-2: genuine element-subdivided PIECEWISE-POLYNOMIAL FE representation.
//
// The 1D axis [-L,L] is split into elements; on each element the density factor
// rho^d(x) = x^m e^{-p x^2} is represented by a local polynomial (barycentric
// Lagrange through the element's Gauss-Legendre nodes). (pq|rs) factorizes by
// Cartesian dimension into A_d(t) = sum over ELEMENT PAIRS (eA,eB) of
//   int_eA int_eB P_eA(x) e^{-t^2(x-x')^2} P_eB(x') dx dx'.
//
// Element-pair structure (answers the two design questions):
//   * NEAR pairs (same/adjacent elements, x and x' can come within ~1/t): the
//     kernel is a knife's edge -> t-adapted inner quadrature x'=x-v/t so e^{-v^2}
//     is flat; P_eB is a POLYNOMIAL -> smooth in v for ALL t -> machine-exact
//     with fixed order, NO near/far t-split (unlike the Gaussian collocation).
//   * FAR pairs (well separated): the mapped v-range falls outside the e^{-v^2}
//     support -> the clamped range is empty -> the pair is screened (this is the
//     "different element factorizes/screens" case).
// Applying the t-adapted element-pair quadrature uniformly thus auto-screens the
// far pairs and is exact for the near ones. Validated to machine precision
// against eri_quartet.

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include <Kokkos_Core.hpp>

#include "intti/fock.hpp"
#include "intti/quartet.hpp"
#include "intti/tgrid.hpp"

using intti::PrimitiveShell;

static void gauss_legendre(int n, double a, double b, std::vector<double> &x,
                           std::vector<double> &w) {
  x.assign(n, 0.0);
  w.assign(n, 0.0);
  for (int i = 0; i < n; ++i) {
    double xi = std::cos(M_PI * (i + 0.75) / (n + 0.5)), dp = 0;
    for (int it = 0; it < 100; ++it) {
      double p0 = 1, p1 = xi;
      for (int k = 2; k <= n; ++k) {
        double p2 = ((2 * k - 1) * xi * p1 - (k - 1) * p0) / k;
        p0 = p1; p1 = p2;
      }
      dp = n * (xi * p1 - p0) / (xi * xi - 1);
      double dx = -p1 / dp; xi += dx;
      if (std::abs(dx) < 1e-15) break;
    }
    x[i] = 0.5 * (a + b) + 0.5 * (b - a) * xi;
    w[i] = (b - a) / ((1 - xi * xi) * dp * dp);
  }
}
static double gauss_moment(int m, double a) {
  if (m % 2) return 0.0;
  double df = 1.0;
  for (int k = m - 1; k > 0; k -= 2) df *= k;
  return df * std::sqrt(M_PI) / (std::pow(2.0, m / 2) * std::pow(a, (m + 1) / 2.0));
}

// 1D piecewise-polynomial FE grid: ne elements on [-L,L], reference GL nodes rn
// (in [-1,1]) with quadrature weights rw and barycentric weights bw.
struct FEGrid {
  int ne, np;
  double L;
  std::vector<double> be;        // element boundaries (ne+1)
  std::vector<double> rn, rw, bw; // reference nodes/weights (np)
  double cen(int e) const { return 0.5 * (be[e] + be[e + 1]); }
  double hw(int e) const { return 0.5 * (be[e + 1] - be[e]); }
  double node(int e, int k) const { return cen(e) + hw(e) * rn[k]; }
};
static FEGrid make_fegrid(int ne, int np, double L) {
  FEGrid g;
  g.ne = ne; g.np = np; g.L = L;
  g.be.resize(ne + 1);
  for (int e = 0; e <= ne; ++e) g.be[e] = -L + 2 * L * e / ne;
  gauss_legendre(np, -1, 1, g.rn, g.rw);
  g.bw.assign(np, 1.0);
  for (int k = 0; k < np; ++k)
    for (int j = 0; j < np; ++j)
      if (j != k) g.bw[k] /= (g.rn[k] - g.rn[j]);
  return g;
}
// barycentric Lagrange interpolation on element e at reference coord xi in [-1,1]
static double lag(const FEGrid &g, const double *f, double xi) {
  double num = 0, den = 0;
  for (int k = 0; k < g.np; ++k) {
    double d = xi - g.rn[k];
    if (std::abs(d) < 1e-13) return f[k];
    double t = g.bw[k] / d;
    num += t * f[k]; den += t;
  }
  return num / den;
}

int run();
int main() {
  Kokkos::initialize();
  int rc = run();
  Kokkos::finalize();
  return rc;
}

int run() {
  const double L = 8.0;
  const int ne = 20, np = 14; // 20 elements, degree-13 polynomials per element
  const int nv = 48;          // inner v-quadrature order
  const int MP = 5;           // per-axis powers 0..4
  auto g = make_fegrid(ne, np, L);
  std::vector<double> rnv, rwv;
  gauss_legendre(nv, -1, 1, rnv, rwv); // reference inner v-nodes

  // ---- overlap: FE composite quadrature of the piecewise-polynomial density ----
  std::printf("== overlap: piecewise-polynomial FE vs analytic ==\n");
  {
    struct Prim { double a; int l[3]; };
    std::vector<Prim> ps = {{1.3, {0, 0, 0}}, {0.7, {1, 0, 0}}, {0.4, {2, 0, 0}}};
    double wa = 0, wr = 0, sc = 0;
    for (auto &A : ps)
      for (auto &B : ps) {
        const double p = A.a + B.a;
        double grid = 1, exact = 1;
        for (int d = 0; d < 3; ++d) {
          const int m = A.l[d] + B.l[d];
          double s = 0;
          for (int e = 0; e < ne; ++e)
            for (int k = 0; k < np; ++k) {
              const double x = g.node(e, k);
              s += g.hw(e) * g.rw[k] * std::pow(x, m) * std::exp(-p * x * x);
            }
          grid *= s; exact *= gauss_moment(m, p);
        }
        wa = std::max(wa, std::abs(grid - exact)); sc = std::max(sc, std::abs(exact));
        if (std::abs(exact) > 1e-14) wr = std::max(wr, std::abs(grid - exact) / std::abs(exact));
      }
    std::printf("   max abs %.3e (scale %.3e), max rel (nonzero) %.3e\n", wa, sc, wr);
  }

  // ---- (pq|rs): FE element-pair loop + t-adapted quadrature (no split) ----
  std::printf("\n== (pq|rs): piecewise-poly FE element-pair t-adapted quadrature vs eri_quartet ==\n");
  std::printf("   %d elements x degree %d, Mobius t, inner GL(%d); far pairs auto-screened\n",
              ne, np - 1, nv);
  auto tgrid = intti::make_tgrid(intti::coulomb());
  const int nt = tgrid.n();
  auto shp = [&](double a, int l) { return PrimitiveShell<double>{a, {0, 0, 0}, l}; };
  struct Quartet { const char *name; PrimitiveShell<double> A, B, C, D; };
  std::vector<Quartet> qs = {
      {"(ss|ss)", shp(1.3, 0), shp(0.8, 0), shp(1.1, 0), shp(0.6, 0)},
      {"(pp|ss)", shp(1.3, 1), shp(0.8, 1), shp(1.1, 0), shp(0.6, 0)},
      {"(pp|pp)", shp(1.0, 1), shp(0.7, 1), shp(0.9, 1), shp(0.6, 1)},
  };
  long near_pairs = 0, far_pairs = 0;
  for (auto &q : qs) {
    const int la = q.A.l, lb = q.B.l, lc = q.C.l, ld = q.D.l;
    const int na = intti::ncart(la), nb = intti::ncart(lb);
    const int nc = intti::ncart(lc), nd = intti::ncart(ld);
    std::vector<double> blk((size_t)na * nb * nc * nd);
    intti::eri_quartet(intti::make_pair(q.A, q.B), intti::make_pair(q.C, q.D), tgrid, blk.data());
    const double pbra = q.A.alpha + q.B.alpha, pket = q.C.alpha + q.D.alpha;
    // FE dofs: density nodal values per element, per power
    std::vector<double> fbra((size_t)ne * MP * np), fket((size_t)ne * MP * np);
    for (int e = 0; e < ne; ++e)
      for (int m = 0; m < MP; ++m)
        for (int k = 0; k < np; ++k) {
          const double x = g.node(e, k);
          fbra[((size_t)e * MP + m) * np + k] = std::pow(x, m) * std::exp(-pbra * x * x);
          fket[((size_t)e * MP + m) * np + k] = std::pow(x, m) * std::exp(-pket * x * x);
        }
    // A[mb][mk][t] via element-pair t-adapted quadrature
    std::vector<double> A((size_t)MP * MP * nt, 0.0);
    for (int mb = 0; mb < MP; ++mb)
      for (int mk = 0; mk < MP; ++mk) {
#pragma omp parallel for schedule(dynamic) reduction(+ : near_pairs, far_pairs)
        for (int it = 0; it < nt; ++it) {
          const double t = tgrid.t[it];
          double Aa = 0;
          for (int eA = 0; eA < ne; ++eA) {
            const double *fbe = &fbra[((size_t)eA * MP + mb) * np];
            for (int ka = 0; ka < np; ++ka) {
              const double x = g.node(eA, ka), wx = g.hw(eA) * g.rw[ka] * fbe[ka];
              if (wx == 0) continue;
              double inner = 0;
              for (int eB = 0; eB < ne; ++eB) {
                // v = t(x-x'); x' in element eB -> v in [t(x-be[eB+1]), t(x-be[eB])]
                double lo = t * (x - g.be[eB + 1]), hi = t * (x - g.be[eB]);
                lo = std::max(lo, -8.0); hi = std::min(hi, 8.0);
                if (hi <= lo) { if (mb == 0 && mk == 0) far_pairs++; continue; } // screened
                if (mb == 0 && mk == 0) near_pairs++;
                const double *fke = &fket[((size_t)eB * MP + mk) * np];
                const double vc = 0.5 * (lo + hi), vh = 0.5 * (hi - lo);
                double s = 0;
                for (int gI = 0; gI < nv; ++gI) {
                  const double v = vc + vh * rnv[gI];
                  const double xp = x - v / t; // in eB by construction
                  const double xi = (xp - g.cen(eB)) / g.hw(eB);
                  s += vh * rwv[gI] * lag(g, fke, xi) * std::exp(-v * v);
                }
                inner += s / t;
              }
              Aa += wx * inner;
            }
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
            for (int it = 0; it < nt; ++it) val += tgrid.w[it] * Ax[it] * Ay[it] * Az[it];
            const double ref = blk[((size_t)ka * nb + kb) * nc * nd + (size_t)kc * nd + kd];
            worst = std::max(worst, std::abs(val - ref));
            scale = std::max(scale, std::abs(ref));
          }
        }
      }
    }
    std::printf("      %-8s  max|fe-eri| = %.3e   rel = %.3e\n", q.name, worst,
                worst / (scale + 1e-300));
  }
  std::printf("   element-pair census (per axis, mb=mk=0, all t): near=%ld far(screened)=%ld\n",
              near_pairs, far_pairs);
  return 0;
}
