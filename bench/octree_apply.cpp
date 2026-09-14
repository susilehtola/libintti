// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

// Applying the Coulomb operator on the adaptive octree, box to box.
//
// This is the operator primitive every version of the multiresolution apply
// needs, and the piece the point counts in bench/octree_count.cpp said nothing
// about. The kernel is separable at each t node,
//     e^{-t^2 |x-y|^2} = prod_d e^{-t^2 (x_d - y_d)^2},
// so the interaction of a source box with a target box is a tensor product of
// three (p x p) matrices acting on the source's p^3 coefficients -- 3 p^4 work
// instead of p^6, and no two-scale relation is needed because the sources and
// targets are both quadrature representations.
//
//     M^(d)_ab = w^S_b exp(-t^2 (x^T_a - y^S_b)^2)
//     V^T_abc += sum_ijk M^(x)_ai M^(y)_bj M^(z)_ck rho^S_ijk
//
// WHAT THIS ESTABLISHED. The apply is correct -- validated to ~3e-6 relative
// against nuclear.hpp::potential_on_points, an independent analytic oracle, with
// the residual set by the domain truncation rather than by the apply (it does
// not move when the order goes from 4 to 6). But it is only correct for part of
// the t grid, and the boundary is a single dimensionless number. Sweeping the
// number of t nodes, with apply and oracle sharing the SAME truncated grid so
// the t-quadrature error cancels and only the spatial discretisation is left:
//
//     t_max     0.073   0.33   0.85    1.9    4.2   10.7   42.7   5754
//     t h_min    0.06   0.29   0.75   1.68   3.73   9.45   37.7   5078
//     rel err   4e-06  3e-06  5e-06  5e-05  7e-04  2e-02  2e-01     77
//
// The error is a function of t h and nothing else: a box-box apply is accurate
// while the kernel width 1/t is comparable to or larger than the box, and fails
// completely once the kernel is narrower than the box it is being integrated
// over -- which is a quadrature statement, not a subtle one.
//
// SO THE HIERARCHY IS NOT AN OPTIMISATION, IT IS REQUIRED FOR CORRECTNESS. Each
// t node has to be applied on boxes of size h ~ 1/t. That is the multiresolution
// principle arrived at from the other end, and it carries a consequence worth
// noting: at large t the required boxes are FINER THAN THE DENSITY'S OWN SCALE,
// so the tree has to be refined for the OPERATOR and not only for the function.
// A tree built to resolve rho, which is what bench/octree_count.cpp measures, is
// not by itself enough.
//
// WHAT IS NOT DONE HERE is the scheduling: choosing a level per t node and
// coupling levels, i.e. the non-standard form proper. This benchmark is the
// primitive that form is built from, plus the criterion any implementation of it
// has to satisfy.
//
// Usage: octree_apply [order] [s_crit] [nt] [eps]

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <Kokkos_Core.hpp>

#include "intti/fegrid.hpp"
#include "intti/fock.hpp"
#include "intti/kernel.hpp"
#include "intti/nuclear.hpp"
#include "intti/tgrid.hpp"

namespace {

struct Box {
  double lo[3], h;   // corner and side
  int level;
  std::vector<double> rho; // p^3 values at the box's Gauss points
};

struct Gauss1D {
  std::vector<double> x, w; // on [0,1]
};

Gauss1D gauss_unit(int p) {
  Gauss1D g;
  intti::detail::fe_gauss_legendre<double>(p, 0.0, 1.0, g.x, g.w);
  return g;
}

// the pair Gaussians of the density, for the refinement predicate
struct G3 {
  double alpha, c[3];
};

std::vector<G3> pair_gaussians(const intti::ShellBasis<double> &b) {
  std::vector<G3> g;
  for (const auto &si : b.shells)
    for (const auto &sj : b.shells) {
      G3 e;
      e.alpha = si.alpha + sj.alpha;
      for (int d = 0; d < 3; ++d)
        e.c[d] = (si.alpha * si.center[d] + sj.alpha * sj.center[d]) / e.alpha;
      g.push_back(e);
    }
  return g;
}

void build(const std::vector<G3> &gs, const double lo[3], double h, double eps, int p,
           double s_crit, int depth, int maxdepth, std::vector<Box> &out) {
  const double cut = -std::log(eps);
  std::vector<G3> live;
  double smax = 0;
  for (const auto &g : gs) {
    double d2 = 0;
    for (int d = 0; d < 3; ++d) {
      const double dd = std::max({lo[d] - g.c[d], g.c[d] - (lo[d] + h), 0.0});
      d2 += dd * dd;
    }
    if (g.alpha * d2 > cut) continue;
    live.push_back(g);
    smax = std::max(smax, 0.5 * h * std::sqrt(2 * g.alpha));
  }
  if (live.empty()) return;
  if (smax <= s_crit || depth >= maxdepth) {
    Box b;
    for (int d = 0; d < 3; ++d) b.lo[d] = lo[d];
    b.h = h;
    b.level = depth;
    out.push_back(std::move(b));
    return;
  }
  const double hh = 0.5 * h;
  for (int k = 0; k < 8; ++k) {
    double l2[3];
    for (int d = 0; d < 3; ++d) l2[d] = lo[d] + (((k >> d) & 1) ? hh : 0.0);
    build(live, l2, hh, eps, p, s_crit, depth + 1, maxdepth, out);
  }
}

} // namespace

int main(int argc, char **argv) {
  const int p = argc > 1 ? std::atoi(argv[1]) : 4;
  const double s_crit = argc > 2 ? std::atof(argv[2]) : 1.3;
  const int nt_want = argc > 3 ? std::atoi(argv[3]) : 16;
  const double eps = argc > 4 ? std::atof(argv[4]) : 1e-2;

  Kokkos::initialize(argc, argv);
  {
    // small all-Gaussian test density: the analytic potential is the oracle
    auto basis = intti::make_basis<double>({{1.2, {0.0, 0.0, 0.0}, 0},
                                            {0.45, {0.0, 0.0, 0.0}, 0},
                                            {0.9, {1.6, 0.4, -0.3}, 0}});
    const int nao = basis.nao;
    std::vector<double> D((std::size_t)nao * nao);
    for (int i = 0; i < nao; ++i)
      for (int j = 0; j < nao; ++j) D[(std::size_t)i * nao + j] = 0.3 + 0.2 * std::cos(1.7 * i + 0.9 * j);
    for (int i = 0; i < nao; ++i)
      for (int j = i + 1; j < nao; ++j) {
        const double a = 0.5 * (D[(std::size_t)i * nao + j] + D[(std::size_t)j * nao + i]);
        D[(std::size_t)i * nao + j] = D[(std::size_t)j * nao + i] = a;
      }

    const auto gs = pair_gaussians(basis);
    double lo[3], hi[3];
    for (int d = 0; d < 3; ++d) { lo[d] = gs[0].c[d]; hi[d] = lo[d]; }
    for (const auto &g : gs) {
      const double w = std::sqrt(-std::log(eps) / g.alpha);
      for (int d = 0; d < 3; ++d) {
        lo[d] = std::min(lo[d], g.c[d] - w);
        hi[d] = std::max(hi[d], g.c[d] + w);
      }
    }
    double side = 0;
    for (int d = 0; d < 3; ++d) side = std::max(side, hi[d] - lo[d]);
    for (int d = 0; d < 3; ++d) lo[d] = 0.5 * (lo[d] + hi[d]) - 0.5 * side;

    std::vector<Box> boxes;
    build(gs, lo, side, eps, p, s_crit, 0, 16, boxes);

    // density at every box's Gauss points
    const auto g1 = gauss_unit(p);
    const int p3 = p * p * p;
    std::vector<std::array<double, 3>> allpts;
    for (auto &b : boxes) {
      b.rho.assign(p3, 0.0);
      for (int i = 0; i < p; ++i)
        for (int j = 0; j < p; ++j)
          for (int k = 0; k < p; ++k) {
            const double x = b.lo[0] + b.h * g1.x[i];
            const double y = b.lo[1] + b.h * g1.x[j];
            const double z = b.lo[2] + b.h * g1.x[k];
            allpts.push_back({x, y, z});
          }
    }
    // rho(g) = sum_ab D_ab chi_a chi_b, evaluated directly
    {
      std::size_t n = 0;
      for (auto &b : boxes)
        for (int i = 0; i < p3; ++i, ++n) {
          const auto &q = allpts[n];
          double r = 0;
          for (int a = 0; a < nao; ++a) {
            const auto &sa = basis.shells[a];
            double ca = 0;
            for (int d = 0; d < 3; ++d) ca += (q[d] - sa.center[d]) * (q[d] - sa.center[d]);
            const double va = std::exp(-sa.alpha * ca);
            for (int bb = 0; bb < nao; ++bb) {
              const auto &sb = basis.shells[bb];
              double cb = 0;
              for (int d = 0; d < 3; ++d) cb += (q[d] - sb.center[d]) * (q[d] - sb.center[d]);
              r += D[(std::size_t)a * nao + bb] * va * std::exp(-sb.alpha * cb);
            }
          }
          b.rho[i] = r;
        }
    }

    // DIAGNOSTIC: does the tree hold the whole density? The potential is long
    // ranged, so a domain that clips the tail loses charge and V is wrong
    // everywhere, not just near the edge.
    {
      double qtree = 0;
      std::size_t n = 0;
      for (const auto &b : boxes)
        for (int i = 0; i < p; ++i)
          for (int j = 0; j < p; ++j)
            for (int k = 0; k < p; ++k, ++n)
              qtree += b.h * b.h * b.h * g1.w[i] * g1.w[j] * g1.w[k] * b.rho[(i * p + j) * p + k];
      double qex = 0;
      for (int a = 0; a < nao; ++a)
        for (int bb = 0; bb < nao; ++bb) {
          const auto &sa = basis.shells[a];
          const auto &sb = basis.shells[bb];
          const double pp = sa.alpha + sb.alpha, mu = sa.alpha * sb.alpha / pp;
          double r2 = 0;
          for (int d = 0; d < 3; ++d)
            r2 += (sa.center[d] - sb.center[d]) * (sa.center[d] - sb.center[d]);
          qex += D[(std::size_t)a * nao + bb] *
                 std::pow(M_PI / pp, 1.5) * std::exp(-mu * r2);
        }
      std::printf("  charge: tree %.10f  exact %.10f  (rel %.2e)\n", qtree, qex,
                  std::abs(qtree - qex) / std::abs(qex));
    }

    // Sweep the t range: apply and oracle use the SAME truncated grid, so the
    // t-quadrature error cancels and what is left is the spatial discretisation
    // alone. This is how to find out which nodes the box-box apply can handle.
    auto tgf = intti::make_tgrid(intti::coulomb());
    const int ntf = tgf.n();
    double hmin = 1e300, hmax = 0;
    for (const auto &b : boxes) { hmin = std::min(hmin, b.h); hmax = std::max(hmax, b.h); }
    std::printf("octree: %zu leaves, order %d (%zu points), domain %.1f bohr\n",
                boxes.size(), p, allpts.size(), side);
    std::printf("  box sizes %.4f to %.4f bohr; full t grid has %d nodes, t up to %.3g\n",
                hmin, hmax, ntf, tgf.t[ntf - 1]);
    std::printf("  %5s %10s %10s %12s %12s\n", "nodes", "t_max", "1/t_max", "rel error", "t*h_min");
    for (int K : {8, 16, 24, 32, 40, 48, 56, ntf}) {
      if (K > ntf) continue;
      intti::TGrid<double> tk = tgf;
      tk.t.resize(K); tk.w.resize(K); tk.tail_coeff = 0.0;
      tk.t_dev = intti::detail::to_device(tk.t, "tk");
      tk.w_dev = intti::detail::to_device(tk.w, "tk_w");
      std::vector<double> V(allpts.size(), 0.0);
      const double rcut = std::sqrt(-std::log(1e-12));
      std::vector<double> Mx((std::size_t)p*p), My((std::size_t)p*p), Mz((std::size_t)p*p);
      std::vector<double> T1((std::size_t)p*p*p), T2((std::size_t)p*p*p);
      for (int it = 0; it < K; ++it) {
        const double t = tk.t[it], wt = tk.w[it];
        const double reach = rcut / t;
        for (std::size_t ti = 0; ti < boxes.size(); ++ti) {
          const auto &TB = boxes[ti];
          for (std::size_t si = 0; si < boxes.size(); ++si) {
            const auto &SB = boxes[si];
            double d2 = 0;
            for (int d = 0; d < 3; ++d) {
              const double gap = std::max({SB.lo[d]-(TB.lo[d]+TB.h), TB.lo[d]-(SB.lo[d]+SB.h), 0.0});
              d2 += gap*gap;
            }
            if (d2 > reach*reach) continue;
            for (int d = 0; d < 3; ++d) {
              double *M = d==0?Mx.data():(d==1?My.data():Mz.data());
              for (int aa = 0; aa < p; ++aa) {
                const double xa = TB.lo[d] + TB.h*g1.x[aa];
                for (int bb2 = 0; bb2 < p; ++bb2) {
                  const double yb = SB.lo[d] + SB.h*g1.x[bb2];
                  const double u = xa - yb;
                  M[aa*p+bb2] = SB.h*g1.w[bb2]*std::exp(-t*t*u*u);
                }
              }
            }
            for (int aa=0;aa<p;++aa) for (int j=0;j<p;++j) for (int k=0;k<p;++k) {
              double sm=0; for (int i=0;i<p;++i) sm += Mx[aa*p+i]*SB.rho[(i*p+j)*p+k];
              T1[(aa*p+j)*p+k]=sm; }
            for (int aa=0;aa<p;++aa) for (int bb2=0;bb2<p;++bb2) for (int k=0;k<p;++k) {
              double sm=0; for (int j=0;j<p;++j) sm += My[bb2*p+j]*T1[(aa*p+j)*p+k];
              T2[(aa*p+bb2)*p+k]=sm; }
            double *Vt = V.data() + ti*p3;
            for (int aa=0;aa<p;++aa) for (int bb2=0;bb2<p;++bb2) for (int c=0;c<p;++c) {
              double sm=0; for (int k=0;k<p;++k) sm += Mz[c*p+k]*T2[(aa*p+bb2)*p+k];
              Vt[(aa*p+bb2)*p+c] += wt*sm; }
          }
        }
      }
      const auto Vref = intti::potential_on_points(basis, D.data(), allpts, tk, 0.0, 0.0);
      double worst=0, scale=0;
      for (std::size_t i=0;i<V.size();++i){ scale=std::max(scale,std::abs(Vref[i]));
        worst=std::max(worst,std::abs(V[i]-Vref[i])); }
      std::printf("  %5d %10.3g %10.3g %12.3e %12.2f\n", K, tk.t[K-1], 1.0/tk.t[K-1],
                  worst/scale, tk.t[K-1]*hmin);
      std::fflush(stdout);
    }
  }
  Kokkos::finalize();
  return 0;
}
