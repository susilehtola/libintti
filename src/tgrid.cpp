// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola

#include "intti/tgrid.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace intti {

void gauss_legendre(int n, double a, double b, double *x, double *w) {
  const double xm = 0.5 * (b + a);
  const double xl = 0.5 * (b - a);
  for (int i = 0; i < (n + 1) / 2; ++i) {
    // Newton iteration on P_n from the Tricomi initial guess
    double z = std::cos(M_PI * (i + 0.75) / (n + 0.5));
    double pp = 0.0;
    for (int it = 0; it < 100; ++it) {
      double p0 = 1.0, p1 = 0.0;
      for (int j = 0; j < n; ++j) {
        const double p2 = p1;
        p1 = p0;
        p0 = ((2 * j + 1) * z * p1 - j * p2) / (j + 1);
      }
      pp = n * (z * p0 - p1) / (z * z - 1.0);
      const double dz = p0 / pp;
      z -= dz;
      if (std::abs(dz) < 1e-15) break;
    }
    x[i] = xm - xl * z;
    x[n - 1 - i] = xm + xl * z;
    w[i] = w[n - 1 - i] = 2.0 * xl / ((1.0 - z * z) * pp * pp);
  }
}

TGrid make_tgrid(const Kernel &kernel, const TGridSpec &spec) {
  // integration range in t
  double t0 = 0.0, t1 = spec.t_c;
  bool tail = true;
  switch (kernel.type) {
  case KernelType::Coulomb:
    break;
  case KernelType::Erf:
    if (kernel.omega <= 0.0) throw std::invalid_argument("erf kernel needs omega > 0");
    t1 = kernel.omega;
    tail = false;
    break;
  case KernelType::Erfc:
    if (kernel.omega <= 0.0) throw std::invalid_argument("erfc kernel needs omega > 0");
    t0 = kernel.omega;
    break;
  case KernelType::Yukawa:
    if (kernel.kappa <= 0.0) throw std::invalid_argument("Yukawa kernel needs kappa > 0");
    break;
  }
  if (t1 <= t0) throw std::invalid_argument("empty t integration range");

  // linear panel [t0, min(t_lin, t1)], logarithmic panel [max(t_lin, t0), t1]
  std::vector<double> tn, wn;
  const double lin_hi = std::min(spec.t_lin, t1);
  if (lin_hi > t0) {
    std::vector<double> x(spec.n_lin), w(spec.n_lin);
    gauss_legendre(spec.n_lin, t0, lin_hi, x.data(), w.data());
    tn.insert(tn.end(), x.begin(), x.end());
    wn.insert(wn.end(), w.begin(), w.end());
  }
  const double log_lo = std::max(spec.t_lin, t0);
  if (t1 > log_lo) {
    if (log_lo <= 0.0) throw std::invalid_argument("logarithmic panel needs t_lin > 0");
    std::vector<double> u(spec.n_log), w(spec.n_log);
    gauss_legendre(spec.n_log, std::log(log_lo), std::log(t1), u.data(), w.data());
    for (int i = 0; i < spec.n_log; ++i) {
      const double t = std::exp(u[i]);
      tn.push_back(t);
      wn.push_back(w[i] * t);
    }
  }

  // overall 2/sqrt(pi) and the kernel t-weight
  const double pref = 2.0 / std::sqrt(M_PI);
  for (std::size_t i = 0; i < tn.size(); ++i) {
    wn[i] *= pref;
    if (kernel.type == KernelType::Yukawa)
      wn[i] *= std::exp(-kernel.kappa * kernel.kappa / (4.0 * tn[i] * tn[i]));
  }

  TGrid grid;
  grid.t = Kokkos::View<double *>("intti::tgrid::t", tn.size());
  grid.w = Kokkos::View<double *>("intti::tgrid::w", wn.size());
  auto th = Kokkos::create_mirror_view(grid.t);
  auto wh = Kokkos::create_mirror_view(grid.w);
  for (std::size_t i = 0; i < tn.size(); ++i) {
    th(i) = tn[i];
    wh(i) = wn[i];
  }
  Kokkos::deep_copy(grid.t, th);
  Kokkos::deep_copy(grid.w, wh);
  grid.t_c = t1;
  grid.tail_coeff = tail ? M_PI / (t1 * t1) : 0.0;
  return grid;
}

} // namespace intti
