// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

// How big would an ADAPTIVE 3D grid be, against the tensor product?
//
// The tensor-product mesh has to resolve every atom's cusp on every axis, so N
// grows linearly in the number of atoms and the point count cubically. An
// octree refines in VOLUME instead: an atom's cusp costs a geometric cascade of
// small boxes around that atom and nothing anywhere else. This counts the
// difference before any of the machinery to use such a grid is written, because
// the count is what decides whether the machinery is worth writing.
//
// Grid construction is usually the bottleneck of finite-element electronic
// structure, because the refinement indicator needs the solution and you pay a
// solve-estimate-refine cycle. That does not apply here: the density is a sum of
// Gaussian products known in closed form BEFORE anything is solved, so the
// indicator is a priori. And because a Gaussian factorises, the error of a
// tensor-product polynomial on a box is bounded by the sum of the three
// one-dimensional errors along its edges -- which is the test fegrid.hpp already
// has. Refinement is therefore a cheap local predicate, evaluated once.
//
// Usage: octree_count <basis-dump> [eps] [order]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "basis_io.hpp"
#include "intti/fegrid.hpp"
#include "intti/gridri.hpp"

namespace {

struct G3 {
  double alpha, c[3];
};

// pair-product Gaussians of the basis, deduplicated
std::vector<G3> pair_gaussians(const intti::ContractedBasis<double> &b) {
  std::vector<G3> g;
  for (const auto &si : b.shells)
    for (const auto &sj : b.shells)
      for (double ai : si.alpha)
        for (double aj : sj.alpha) {
          G3 e;
          e.alpha = ai + aj;
          for (int d = 0; d < 3; ++d)
            e.c[d] = (ai * si.center[d] + aj * sj.center[d]) / e.alpha;
          g.push_back(e);
        }
  std::sort(g.begin(), g.end(), [](const G3 &a, const G3 &b2) {
    if (a.alpha != b2.alpha) return a.alpha < b2.alpha;
    for (int d = 0; d < 3; ++d)
      if (a.c[d] != b2.c[d]) return a.c[d] < b2.c[d];
    return false;
  });
  g.erase(std::unique(g.begin(), g.end(),
                      [](const G3 &a, const G3 &b2) {
                        if (std::abs(a.alpha - b2.alpha) > 1e-12 * a.alpha) return false;
                        for (int d = 0; d < 3; ++d)
                          if (std::abs(a.c[d] - b2.c[d]) > 1e-9) return false;
                        return true;
                      }),
          g.end());
  return g;
}

long nleaf = 0, maxdepth = 0;
double npoints = 0;

long nprobe_calls = 0;
double s_accept = 0, s_reject = 1e300; // h*sqrt(2a) at accept / at reject

// `gs` is the list that reached the PARENT. A Gaussian that does not reach a box
// cannot reach its children, so the list only shrinks going down and each level
// filters the previous one instead of the global set. Rescanning all of them at
// every box is what made this take a minute and a half; it is the whole reason
// finite-element grid construction gets a reputation, and here it is avoidable
// because the indicator never needs a solve.
/// s_crit > 0 selects the CLOSED-FORM indicator: refine while the box spans more
/// than s_crit Gaussian widths, h sqrt(2 a). The probing indicator turns out to
/// be a function of that group alone -- its accept and reject bands nearly touch
/// (1.052 against 0.819 at order 6; 1.976 against 1.838 at order 10) -- so the
/// 16-probe interpolation test it runs per axis per Gaussian per degree buys
/// nothing over one comparison.
double s_crit_global = 0;

void refine(const std::vector<G3> &gs, const double lo[3], const double hi[3], double eps,
            int p, int pdeg, int depth, int maxdepth_allowed) {
  const double cut = -std::log(eps) + pdeg;
  std::vector<G3> live;
  live.reserve(gs.size());
  for (const auto &g : gs) {
    double d2 = 0;
    for (int d = 0; d < 3; ++d) {
      const double dd = std::max({lo[d] - g.c[d], g.c[d] - hi[d], 0.0});
      d2 += dd * dd;
    }
    if (g.alpha * d2 <= cut) live.push_back(g);
  }
  if (live.empty()) return; // nothing lives here: no leaf, no points
  ++nprobe_calls;
  double smax = 0;
  const double hbox = 0.5 * (hi[0] - lo[0]);
  for (const auto &g : live) smax = std::max(smax, hbox * std::sqrt(2 * g.alpha));
  if (s_crit_global > 0) {
    if (smax <= s_crit_global || depth >= maxdepth_allowed) {
      s_accept = std::max(s_accept, smax);
      ++nleaf;
      maxdepth = std::max(maxdepth, (long)depth);
      npoints += double(p) * p * p;
      return;
    }
    s_reject = std::min(s_reject, smax);
    double mid2[3];
    for (int d = 0; d < 3; ++d) mid2[d] = 0.5 * (lo[d] + hi[d]);
    for (int k = 0; k < 8; ++k) {
      double l2[3], h2[3];
      for (int d = 0; d < 3; ++d) {
        const bool up = (k >> d) & 1;
        l2[d] = up ? mid2[d] : lo[d];
        h2[d] = up ? hi[d] : mid2[d];
      }
      refine(live, l2, h2, eps, p, pdeg, depth + 1, maxdepth_allowed);
    }
    return;
  }
  std::vector<intti::FEGaussian1D<double>> ax[3];
  for (const auto &g : live)
    for (int d = 0; d < 3; ++d) ax[d].push_back({g.alpha, g.c[d]});
  // a Gaussian factorises, so the tensor-product error on the box is bounded by
  // the sum of the one-dimensional errors along its edges
  double err = 0;
  for (int d = 0; d < 3; ++d)
    err += intti::detail::fe_hp_elem_err(ax[d], lo[d], hi[d], p, pdeg, 16);
  // what the probing indicator is effectively enforcing: s = h sqrt(2a) is the
  // number of Gaussian widths across the half-box, the only dimensionless group
  // a barycentric interpolant of a Gaussian can depend on
  if (err <= eps || depth >= maxdepth_allowed) {
    s_accept = std::max(s_accept, smax);
    ++nleaf;
    maxdepth = std::max(maxdepth, (long)depth);
    npoints += double(p) * p * p;
    return;
  }
  s_reject = std::min(s_reject, smax);
  double mid[3];
  for (int d = 0; d < 3; ++d) mid[d] = 0.5 * (lo[d] + hi[d]);
  for (int k = 0; k < 8; ++k) {
    double l2[3], h2[3];
    for (int d = 0; d < 3; ++d) {
      const bool up = (k >> d) & 1;
      l2[d] = up ? mid[d] : lo[d];
      h2[d] = up ? hi[d] : mid[d];
    }
    refine(live, l2, h2, eps, p, pdeg, depth + 1, maxdepth_allowed);
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <basis-dump> [eps] [order]\n", argv[0]);
    return 1;
  }
  const std::string path = argv[1];
  const double eps = argc > 2 ? std::atof(argv[2]) : 1e-2;
  const int p = argc > 3 ? std::atoi(argv[3]) : 6;
  s_crit_global = argc > 4 ? std::atof(argv[4]) : 0.0;

  Kokkos::initialize(argc, argv);
  {
    const auto basis = intti_bench::load_basis(path);
    const auto gs = pair_gaussians(basis);
    int lmax = 0;
    for (const auto &s : basis.shells) lmax = std::max(lmax, s.l);
    const int pdeg = 2 * lmax;

    // the tensor grid, for comparison
    const auto tg = intti::grid_for_basis(basis, eps);
    const double tensor_pts = double(tg.N) * tg.N * tg.N;

    // cubic domain covering every Gaussian to eps
    double lo[3], hi[3];
    for (int d = 0; d < 3; ++d) {
      lo[d] = gs.front().c[d];
      hi[d] = lo[d];
    }
    for (const auto &g : gs) {
      const double w = std::sqrt((-std::log(eps) + pdeg) / g.alpha);
      for (int d = 0; d < 3; ++d) {
        lo[d] = std::min(lo[d], g.c[d] - w);
        hi[d] = std::max(hi[d], g.c[d] + w);
      }
    }
    double side = 0;
    for (int d = 0; d < 3; ++d) side = std::max(side, hi[d] - lo[d]);
    for (int d = 0; d < 3; ++d) {
      const double c = 0.5 * (lo[d] + hi[d]);
      lo[d] = c - 0.5 * side;
      hi[d] = c + 0.5 * side;
    }

    const auto t0 = std::chrono::steady_clock::now();
    refine(gs, lo, hi, eps, std::max(p, pdeg + 2), pdeg, 0, 24);
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();

    std::printf("%s\n", path.c_str());
    std::printf("  nshell=%d nao=%d lmax=%d  %zu distinct pair gaussians  domain %.1f bohr\n",
                (int)basis.shells.size(), basis.nao, lmax, gs.size(), side);
    std::printf("  tensor grid : N=%d/axis        %.3e points\n", tg.N, tensor_pts);
    std::printf("  octree      : %ld leaves, depth %ld, order %d  %.3e points\n", nleaf,
                maxdepth, std::max(p, pdeg + 2), npoints);
    std::printf("  ratio       : %.1fx fewer points   (tree built in %.0f ms, %ld boxes tested)\n",
                tensor_pts / npoints, ms, nprobe_calls);
    std::printf("  indicator   : accepted up to s = h*sqrt(2a) = %.3f, refined from %.3f\n",
                s_accept, s_reject);
  }
  Kokkos::finalize();
  return 0;
}
