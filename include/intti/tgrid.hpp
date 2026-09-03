// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <Kokkos_Core.hpp>

#include "kernel.hpp"
#include "math.hpp"

namespace intti {

/// t-grid mappings.
///
/// Mobius: t = s u/(1-u) with Gauss-Legendre nodes u in (0,1). Covers
/// [0, inf) without truncation, so no tail correction is needed. The ERI
/// integrand saturates (theta -> rho) and decays algebraically as t^-3,
/// which the rational map integrates natively: machine precision by
/// N ~ 48-64 nodes for exponents spanning [1e-2, 1e6] (default).
///
/// LinLog: composite linear [0, t_lin] + logarithmic [t_lin, t_c]
/// Gauss-Legendre panels (Jusélius and Sundholm, J. Chem. Phys. 126, 094101
/// (2007)), truncated at t_c with the delta-function tail correction of
/// Losilla, Mehine and Sundholm, Mol. Phys. 110, 2569 (2012):
/// exp(-t^2 r^2) -> (pi^(3/2)/t^3) delta^3(r), so the tail contributes
/// tail_coeff = pi/t_c^2 times the corresponding delta-type integral (the
/// four-orbital overlap for an ERI quartet), turning the O(1/t_c^2)
/// truncation error into O(1/t_c^4). Use this mapping when t_c is imposed
/// externally, e.g. by the spatial resolution of a real-space grid.
/// ExpSum: the Beylkin-Monzon / sinc quadrature for 1/r. Substituting t = e^s in
/// 1/r = (2/sqrt pi) int e^{-t^2 r^2} dt gives (2/sqrt pi) int e^{s - e^{2s} r^2}
/// ds; the trapezoidal rule on the (analytic, doubly-exponentially decaying)
/// integrand converges exponentially and is near-optimal for 1/r over a finite
/// range -- log-spaced nodes t_k = e^{s_k} with weight (2/sqrt pi) h t_k. Node
/// count grows only ~log(range)*log(1/eps), unlike the Mobius map's superlinear
/// tail growth. For [0, inf) kernels with tail decay -- Coulomb and Yukawa (whose
/// e^{-kappa^2/4t^2} factor even kills the slow small-t tail). erf/erfc have a
/// hard boundary at omega (only O(h^2) for the sinc rule) -- use Mobius there.
enum class TMapping { Mobius, LinLog, ExpSum };

template <class Real = double> struct TGridSpec {
  TMapping mapping{TMapping::Mobius};
  // Mobius parameters
  int n{64};  ///< Gauss-Legendre nodes
  Real s{2};  ///< map scale
  // LinLog parameters
  Real t_lin{3}; ///< switch point between the linear and logarithmic panels
  int n_lin{50}; ///< nodes on the linear panel
  int n_log{80}; ///< nodes on the logarithmic panel
  Real t_c{60};  ///< truncation point of infinite-range kernels
  // ExpSum (Beylkin-Monzon / sinc) parameters
  Real es_tmin{Real(1e-2)}; ///< smallest quadrature node t
  Real es_tmax{Real(1e4)};  ///< largest quadrature node t
  Real es_h{Real(0.35)};    ///< trapezoidal spacing in s = log t
};

/// Quadrature grid in t. The weights include the overall 2/sqrt(pi) factor
/// and the kernel t-weight (e.g. exp(-kappa^2/(4t^2)) for Yukawa), so
///   kernel(r) ~= sum_i w_i exp(-t_i^2 r^2)  (+ delta-function tail).
template <class Real = double> struct TGrid {
  std::vector<Real> t; ///< quadrature nodes (host)
  std::vector<Real> w; ///< quadrature weights (host, see above)
  Real t_c{0};         ///< truncation point (0 for untruncated mappings)
  Real tail_coeff{0};  ///< pi/t_c^2 for truncated infinite-range kernels, else 0
  /// device copies, populated for builtin floating-point Real
  Kokkos::View<Real *> t_dev, w_dev;

  int n() const { return static_cast<int>(t.size()); }
};

/// Gauss-Legendre nodes and weights on [a, b], in the precision of Real.
template <class Real>
void gauss_legendre(int n, Real a, Real b, Real *x, Real *w) {
  const Real pi = pi_v<Real>();
  const Real tol = 16 * std::numeric_limits<Real>::epsilon();
  const Real xm = (b + a) / 2;
  const Real xl = (b - a) / 2;
  for (int i = 0; i < (n + 1) / 2; ++i) {
    // Newton iteration on P_n from the Tricomi initial guess
    Real z = cos_(pi * (Real(i) + Real(0.75)) / (Real(n) + Real(0.5)));
    Real pp{};
    for (int it = 0; it < 200; ++it) {
      Real p0 = 1, p1 = 0;
      for (int j = 0; j < n; ++j) {
        const Real p2 = p1;
        p1 = p0;
        p0 = ((2 * j + 1) * z * p1 - j * p2) / (j + 1);
      }
      pp = n * (z * p0 - p1) / (z * z - 1);
      const Real dz = p0 / pp;
      z -= dz;
      if (abs_(dz) < tol) break;
    }
    x[i] = xm - xl * z;
    x[n - 1 - i] = xm + xl * z;
    w[i] = w[n - 1 - i] = 2 * xl / ((1 - z * z) * pp * pp);
  }
}

/// LinLog panel sizes for a given truncation point and target accuracy,
/// from the M1 convergence study (heuristic: ~27 log-panel nodes per unit of
/// ln t at eps ~ 1e-10, scaled with the accuracy demand).
template <class Real = double>
TGridSpec<Real> linlog_for(Real t_c, Real target_eps = Real(1e-10)) {
  auto ceil_int = [](Real x) {
    const int i = static_cast<int>(x);
    return x > Real(i) ? i + 1 : i;
  };
  TGridSpec<Real> spec;
  spec.mapping = TMapping::LinLog;
  spec.t_lin = 2;
  spec.t_c = t_c;
  const Real demand = -log_(target_eps) / log_(Real(10)) / 10; // 1.0 at eps = 1e-10
  const Real f = demand < Real(0.5) ? Real(0.5) : demand;
  spec.n_lin = ceil_int(50 * f);
  const Real span = t_c > spec.t_lin ? log_(Real(t_c / spec.t_lin)) : Real(0);
  int nl = ceil_int(27 * f * span);
  spec.n_log = nl < 20 ? 20 : nl;
  return spec;
}

/// Mobius TGridSpec sized to a basis's Gaussian exponent span, so one shared
/// grid resolves every pair. Anchored to the calibrated default (n=64, s=2
/// covers exponents [1e-2, 1e6], i.e. 8 decades): the node count grows ~8 per
/// decade of exponent span, and the map scale s tracks the geometric-mean
/// exponent relative to the reference (G0 = 1e2). Opt-in -- make_tgrid's default
/// spec is unchanged. `pad_decades` widens the effective range on each side for
/// the geometry factor (well-separated diffuse pairs need slightly smaller t).
template <class Real>
TGridSpec<Real> mobius_spec_for_range(Real alpha_min, Real alpha_max, double pad_decades = 1.0) {
  TGridSpec<Real> spec; // mapping defaults to Mobius
  const double amin = static_cast<double>(alpha_min), amax = static_cast<double>(alpha_max);
  const double lo = std::log10(amin) - pad_decades, hi = std::log10(amax) + pad_decades;
  // A single Mobius map needs superlinearly more nodes for a wider range (its
  // tails thin out), so anchor to the proven default -- 64 nodes at 10 padded
  // decades -- and grow as (decades/10)^2.5.
  const double decades = std::max(10.0, hi - lo);
  spec.n = std::max(64, static_cast<int>(std::ceil(64.0 * std::pow(decades / 10.0, 2.5))));
  const double G = std::pow(10.0, 0.5 * (lo + hi)); // padded geometric-mean exponent
  spec.s = static_cast<Real>(2.0 * std::sqrt(G / 1e2));
  return spec;
}

/// Beylkin-Monzon / sinc (ExpSum) TGridSpec for the Coulomb kernel sized to a
/// basis's exponent span. Log-spaced nodes cover t ~ [sqrt(alpha_min),
/// sqrt(alpha_max)] with margins for the integrand tails; `h` is the trapezoidal
/// spacing (smaller -> more accurate, exponentially). Node count grows only
/// logarithmically with the range, so it beats the Mobius map at wide spans.
template <class Real>
TGridSpec<Real> exp_sum_spec_for_range(Real alpha_min, Real alpha_max, Real h = Real(0.25)) {
  TGridSpec<Real> spec;
  spec.mapping = TMapping::ExpSum;
  // The small-t (large-r) side has a slow e^s tail, so it needs a wide margin;
  // the large-t (small-r) side cuts off super-exponentially, so a modest one.
  spec.es_tmin = static_cast<Real>(std::sqrt(static_cast<double>(alpha_min)) * 1e-9);
  spec.es_tmax = static_cast<Real>(std::sqrt(static_cast<double>(alpha_max)) * 30.0);
  spec.es_h = h;
  return spec;
}

template <class Real = double>
TGrid<Real> make_tgrid(const Kernel<Real> &kernel, const TGridSpec<Real> &spec = {}) {
  const Real pi = pi_v<Real>();
  const Real pref = 2 / sqrt_(pi);

  TGrid<Real> grid;

  if (spec.mapping == TMapping::Mobius) {
    std::vector<Real> u(spec.n), wu(spec.n);
    switch (kernel.type) {
    case KernelType::Erf: {
      // finite range [0, omega]: plain Gauss-Legendre, no map needed
      if (kernel.omega <= Real(0))
        throw std::invalid_argument("erf kernel needs omega > 0");
      gauss_legendre(spec.n, Real(0), kernel.omega, u.data(), wu.data());
      grid.t = u;
      grid.w = wu;
      break;
    }
    case KernelType::Coulomb:
    case KernelType::Erfc:
    case KernelType::Yukawa: {
      const Real shift = kernel.type == KernelType::Erfc ? kernel.omega : Real(0);
      if (kernel.type == KernelType::Erfc && kernel.omega <= Real(0))
        throw std::invalid_argument("erfc kernel needs omega > 0");
      if (kernel.type == KernelType::Yukawa && kernel.kappa <= Real(0))
        throw std::invalid_argument("Yukawa kernel needs kappa > 0");
      gauss_legendre(spec.n, Real(0), Real(1), u.data(), wu.data());
      grid.t.resize(spec.n);
      grid.w.resize(spec.n);
      for (int i = 0; i < spec.n; ++i) {
        const Real om1 = 1 - u[i];
        grid.t[i] = shift + spec.s * u[i] / om1;
        grid.w[i] = wu[i] * spec.s / (om1 * om1);
      }
      break;
    }
    }
    // untruncated (or exactly finite): no tail correction
    grid.t_c = Real(0);
    grid.tail_coeff = Real(0);
  } else if (spec.mapping == TMapping::ExpSum) { // Beylkin-Monzon / sinc
    // The trapezoidal (sinc) rule is exponentially convergent only when the
    // integrand decays to zero at both ends. Coulomb and Yukawa are [0, inf)
    // with tail decay (Yukawa's e^{-kappa^2/4t^2} factor even kills the slow
    // small-t tail); erf ([0, omega]) and erfc ([omega, inf)) have a hard
    // boundary at omega where the integrand is nonzero, so the sinc rule is
    // only O(h^2) there -- use the Mobius mapping (Gauss-Legendre) for those.
    if (kernel.type != KernelType::Coulomb && kernel.type != KernelType::Yukawa)
      throw std::invalid_argument("ExpSum supports Coulomb and Yukawa; use Mobius for erf/erfc");
    if (kernel.type == KernelType::Yukawa && kernel.kappa <= Real(0))
      throw std::invalid_argument("Yukawa kernel needs kappa > 0");
    if (spec.es_tmin <= Real(0) || spec.es_tmax <= spec.es_tmin || spec.es_h <= Real(0))
      throw std::invalid_argument("ExpSum needs 0 < es_tmin < es_tmax and es_h > 0");
    const Real slo = log_(spec.es_tmin), shi = log_(spec.es_tmax);
    int nn = static_cast<int>((shi - slo) / spec.es_h + Real(0.5));
    if (nn < 1) nn = 1;
    const Real hh = (shi - slo) / Real(nn); // land the endpoints exactly on nodes
    grid.t.reserve(nn + 1);
    grid.w.reserve(nn + 1);
    for (int k = 0; k <= nn; ++k) {
      const Real t = exp_(slo + Real(k) * hh);
      grid.t.push_back(t);
      Real wk = hh * t;                     // trapezoidal step times dt = t ds
      if (k == 0 || k == nn) wk *= Real(0.5); // trapezoidal endpoints
      grid.w.push_back(wk);
    }
    grid.t_c = Real(0);
    grid.tail_coeff = Real(0);
  } else { // TMapping::LinLog
    Real t0 = 0, t1 = spec.t_c;
    bool tail = true;
    switch (kernel.type) {
    case KernelType::Coulomb:
      break;
    case KernelType::Erf:
      if (kernel.omega <= Real(0))
        throw std::invalid_argument("erf kernel needs omega > 0");
      t1 = kernel.omega;
      tail = false;
      break;
    case KernelType::Erfc:
      if (kernel.omega <= Real(0))
        throw std::invalid_argument("erfc kernel needs omega > 0");
      t0 = kernel.omega;
      break;
    case KernelType::Yukawa:
      if (kernel.kappa <= Real(0))
        throw std::invalid_argument("Yukawa kernel needs kappa > 0");
      break;
    }
    if (t1 <= t0) throw std::invalid_argument("empty t integration range");

    // linear panel [t0, min(t_lin, t1)], logarithmic panel [max(t_lin, t0), t1]
    const Real lin_hi = spec.t_lin < t1 ? spec.t_lin : t1;
    if (lin_hi > t0) {
      std::vector<Real> x(spec.n_lin), w(spec.n_lin);
      gauss_legendre(spec.n_lin, t0, lin_hi, x.data(), w.data());
      grid.t.insert(grid.t.end(), x.begin(), x.end());
      grid.w.insert(grid.w.end(), w.begin(), w.end());
    }
    const Real log_lo = spec.t_lin > t0 ? spec.t_lin : t0;
    if (t1 > log_lo) {
      if (log_lo <= Real(0))
        throw std::invalid_argument("logarithmic panel needs t_lin > 0");
      std::vector<Real> uu(spec.n_log), w(spec.n_log);
      gauss_legendre(spec.n_log, log_(log_lo), log_(t1), uu.data(), w.data());
      for (int i = 0; i < spec.n_log; ++i) {
        const Real t = exp_(uu[i]);
        grid.t.push_back(t);
        grid.w.push_back(w[i] * t);
      }
    }
    grid.t_c = t1;
    grid.tail_coeff = tail ? pi / (t1 * t1) : Real(0);
  }

  // overall 2/sqrt(pi) and the kernel t-weight
  for (std::size_t i = 0; i < grid.t.size(); ++i) {
    grid.w[i] *= pref;
    if (kernel.type == KernelType::Yukawa)
      grid.w[i] *= exp_(Real(-kernel.kappa * kernel.kappa / (4 * grid.t[i] * grid.t[i])));
  }

  // device copies for builtin floating-point types
  if constexpr (kokkos_scalar_v<Real>) {
    grid.t_dev = Kokkos::View<Real *>("intti::tgrid::t", grid.t.size());
    grid.w_dev = Kokkos::View<Real *>("intti::tgrid::w", grid.w.size());
    auto th = Kokkos::create_mirror_view(grid.t_dev);
    auto wh = Kokkos::create_mirror_view(grid.w_dev);
    for (std::size_t i = 0; i < grid.t.size(); ++i) {
      th(i) = grid.t[i];
      wh(i) = grid.w[i];
    }
    Kokkos::deep_copy(grid.t_dev, th);
    Kokkos::deep_copy(grid.w_dev, wh);
  }
  return grid;
}

} // namespace intti
