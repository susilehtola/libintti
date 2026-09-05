// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
//
// Prototype: per-axis 1D hp-adaptive mesh generator for the tensorial FE grid
// (roadmap M-FE "grid construction" design note). Given a set of 1D Gaussians
// {exp(-alpha_i (x - X_i)^2)} that appear on one Cartesian axis (the per-axis
// factors of the molecule's primitives, possibly incl. the DAGE t-superposition
// / STO transform), build a 1D finite-element mesh -- element boundaries + a
// per-element polynomial order -- that interpolates EVERY such Gaussian to a
// target accuracy eps with as few degrees of freedom as possible.
//
// Why this is cheap and hp-natural: a 1D Gaussian is entire, so barycentric-
// Lagrange interpolation through Gauss-Legendre nodes on an element converges
// EXPONENTIALLY in the order p (spectral). So the optimal mesh is few elements
// at moderate-to-high p; h-refinement only separates disparate widths/centres.
// The 3D grid is the tensor product of three such 1D meshes; the whole
// construction is three independent 1D problems (O(nprim) 1D work), which is
// also exactly the per-axis structure the DAGE convolution consumes.
//
// Validated against: (a) the achieved interpolation error over ALL Gaussians on
// a dense probe set is <= a small multiple of eps; (b) the hp mesh uses far
// fewer DOFs than a uniform-order uniform-element mesh reaching the same error.
//
// Build (standalone):
//   g++ -std=c++20 -O3 -I include -I include/intti -isystem /usr/include/kokkos \
//       prototype/fe_hp_mesh.cpp -o /tmp/fe_hp_mesh -lkokkoscore && /tmp/fe_hp_mesh

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "intti/fegrid.hpp" // intti::detail::fe_gauss_legendre

struct Gauss1D {
  double alpha, center;
};

// Barycentric-Lagrange value at xt of the polynomial through (nodes, vals).
static double bary(const std::vector<double> &nodes, const std::vector<double> &w,
                   const std::vector<double> &vals, double xt) {
  double num = 0, den = 0;
  for (std::size_t k = 0; k < nodes.size(); ++k) {
    double d = xt - nodes[k];
    if (std::fabs(d) < 1e-14) return vals[k];
    double t = w[k] / d;
    num += t * vals[k];
    den += t;
  }
  return num / den;
}

// Worst absolute interpolation error of any Gaussian on element [a,b] at order
// p (barycentric through p Gauss-Legendre nodes), probed at nprobe points.
static double element_error(const std::vector<Gauss1D> &gs, double a, double b,
                            int p, int nprobe = 64) {
  std::vector<double> nodes, gw;
  intti::detail::fe_gauss_legendre<double>(p, a, b, nodes, gw);
  // barycentric weights for these (non-uniform) nodes
  std::vector<double> w(p, 1.0);
  for (int k = 0; k < p; ++k)
    for (int j = 0; j < p; ++j)
      if (j != k) w[k] /= (nodes[k] - nodes[j]);
  double worst = 0;
  std::vector<double> vals(p);
  for (const auto &g : gs) {
    for (int k = 0; k < p; ++k) {
      const double dx = nodes[k] - g.center;
      vals[k] = std::exp(-g.alpha * dx * dx);
    }
    for (int m = 0; m <= nprobe; ++m) {
      const double xt = a + (b - a) * m / nprobe;
      const double dx = xt - g.center;
      const double exact = std::exp(-g.alpha * dx * dx);
      worst = std::max(worst, std::fabs(bary(nodes, w, vals, xt) - exact));
    }
  }
  return worst;
}

// Build the hp mesh by recursive refinement: on each element try increasing p up
// to pmax; if none reaches eps, bisect (h) and recurse. Emits boundaries+orders.
static void refine(const std::vector<Gauss1D> &gs, double a, double b, double eps,
                   int pmin, int pmax, std::vector<double> &bnd, std::vector<int> &ord) {
  for (int p = pmin; p <= pmax; ++p) {
    if (element_error(gs, a, b, p) <= eps) {
      bnd.push_back(b);
      ord.push_back(p);
      return;
    }
  }
  const double m = 0.5 * (a + b);
  refine(gs, a, m, eps, pmin, pmax, bnd, ord);
  refine(gs, m, b, eps, pmin, pmax, bnd, ord);
}

int main() {
  // A deliberately stiff 1D primitive set: exponents over 6 orders of magnitude
  // (tight core to diffuse), a few centres (atoms along the axis).
  std::vector<double> alphas = {1e4, 1e3, 1e2, 30.0, 5.0, 1.0, 0.2, 0.03};
  std::vector<double> centres = {0.0, 1.4, -1.1};
  std::vector<Gauss1D> gs;
  for (double c : centres)
    for (double a : alphas) gs.push_back({a, c});

  const double eps = 1e-8;
  // domain: cover every Gaussian down to amplitude eps
  double xlo = 1e30, xhi = -1e30;
  for (const auto &g : gs) {
    const double w = std::sqrt(-std::log(eps) / g.alpha);
    xlo = std::min(xlo, g.center - w);
    xhi = std::max(xhi, g.center + w);
  }

  std::vector<double> bnd;
  std::vector<int> ord;
  bnd.push_back(xlo);
  // seed element boundaries at the centres so refinement starts scale-aware
  std::vector<double> seeds = centres;
  seeds.push_back(xlo);
  seeds.push_back(xhi);
  std::sort(seeds.begin(), seeds.end());
  seeds.erase(std::unique(seeds.begin(), seeds.end()), seeds.end());
  std::vector<double> ebnd{xlo};
  std::vector<int> eord;
  for (std::size_t s = 0; s + 1 < seeds.size(); ++s)
    refine(gs, seeds[s], seeds[s + 1], eps, 4, 16, ebnd, eord);

  const int nelem = static_cast<int>(eord.size());
  int dofs = 0;
  for (int p : eord) dofs += p;

  // verify: worst error of any Gaussian over the whole mesh
  double worst = 0;
  for (int e = 0; e < nelem; ++e)
    worst = std::max(worst, element_error(gs, ebnd[e], ebnd[e + 1], eord[e], 128));

  std::printf("hp mesh over [%.3f, %.3f] for %zu Gaussians, eps=%.0e\n", xlo, xhi,
              gs.size(), eps);
  std::printf("  elements = %d, total DOFs = %d, worst error = %.2e\n", nelem, dofs, worst);

  // compare: uniform mesh (fixed order, uniform elements) for the same accuracy
  int uni_dofs = -1;
  for (int p = 8; p <= 16 && uni_dofs < 0; p += 2) {
    for (int ne = nelem; ne <= 4000; ne = ne * 3 / 2 + 1) {
      double w2 = 0;
      const double h = (xhi - xlo) / ne;
      for (int e = 0; e < ne; ++e)
        w2 = std::max(w2, element_error(gs, xlo + e * h, xlo + (e + 1) * h, p, 32));
      if (w2 <= eps) {
        uni_dofs = ne * p;
        std::printf("  uniform match: order %d x %d elements = %d DOFs (error %.2e)\n",
                    p, ne, uni_dofs, w2);
        break;
      }
    }
  }
  if (uni_dofs > 0)
    std::printf("  hp/uniform DOF ratio = %.2fx fewer\n", double(uni_dofs) / dofs);

  const bool ok = worst <= 20 * eps;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
