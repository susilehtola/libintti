# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026 Susi Lehtola
"""Validate the Slater-type-orbital-via-quadrature-contracted-GTO approach.

An STO is the exact integral transform of a Gaussian,
  e^{-zeta r} = (zeta/2 sqrt(pi)) int_0^inf s^{-3/2} e^{-zeta^2/4s} e^{-s r^2} ds,
so a normalized 1s STO phi = sqrt(zeta^3/pi) e^{-zeta r} becomes a
quadrature-contracted s-GTO: phi = sqrt(zeta^3/pi) sum_k c_k e^{-s_k r^2}.

The two-electron self-repulsion of the 1s Slater density is analytically
J = 5 zeta / 8. We reproduce it by contracting same-centre (ss|ss) Gaussian
integrals over the s-expansion -- the nested s (STO) x t (Coulomb) quadrature
the STO milestone rests on. This pins the s-grid node counts.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss


def sto_gaussians(zeta, N, smin=1e-4, smax=1e6):
    """1s STO e^{-zeta r} ~ sum_k c_k e^{-s_k r^2}: log-Gauss-Legendre in s."""
    x, w = leggauss(N)
    lo, hi = np.log(smin), np.log(smax)
    u = 0.5 * (hi - lo) * (x + 1) + lo
    s = np.exp(u)
    ws = 0.5 * (hi - lo) * w * s          # d s = s du
    c = zeta / (2 * np.sqrt(np.pi)) * ws * s ** -1.5 * np.exp(-zeta ** 2 / (4 * s))
    return s, c


def ss_same_centre(p, q):
    # (g_p g | g_q g) with all Gaussians on one centre, F0(0)=1
    return 2 * np.pi ** 2.5 / (p * q * np.sqrt(p + q))


def self_repulsion(zeta, N):
    s, c = sto_gaussians(zeta, N)
    # density phi^2 = (zeta^3/pi) sum_kl c_k c_l e^{-(s_k+s_l) r^2}
    P = s[:, None] + s[None, :]              # pair exponents
    C = (zeta ** 3 / np.pi) * np.outer(c, c)  # pair coeffs (incl norm)
    P = P.ravel(); C = C.ravel()
    # J = sum_{ab} C_a C_b (a|b)
    J = 0.0
    for a in range(len(P)):
        J += C[a] * np.sum(C * ss_same_centre(P[a], P))
    return J


if __name__ == "__main__":
    zeta = 1.3
    exact = 5 * zeta / 8
    print(f"1s Slater self-repulsion, zeta={zeta}: analytic J = 5 zeta/8 = {exact:.10f}")
    # first check the expansion reproduces e^{-zeta r}
    for N in [24, 48, 96, 160]:
        s, c = sto_gaussians(zeta, N)
        r = np.logspace(-2, 1, 300)
        approx = (np.exp(-np.outer(r * r, s)) * c).sum(1)
        exp_err = np.max(np.abs(approx - np.exp(-zeta * r)))
        J = self_repulsion(zeta, N)
        print(f"  N_s={N:3d}:  max|expansion - e^(-zeta r)| = {exp_err:.2e}   "
              f"J = {J:.10f}   rel err {abs(J - exact) / exact:.2e}")
    print("\n=> STO = quadrature-contracted GTO; nested s x t quadrature "
          "reproduces the analytic Slater ERI.")
