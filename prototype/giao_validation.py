# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026 Susi Lehtola
"""Analytic reference for GIAO (London orbital) ERIs.

A London orbital omega_a = exp(-i/2 [B x (R_a - O)].r) chi_a makes the pair
product omega_a^* omega_b a plane-wave-modulated Gaussian, i.e. a Gaussian
with a COMPLEX centre and an unchanged REAL exponent:

    P~ = P - i K/(2p),   K = (1/2) B x (R_b - R_a),
    prefactor  exp(-i K.P - K^2/(4p)).

The analytic ERI is then the ordinary McMurchie-Davidson expression evaluated
with complex centres -- which requires the Boys function of a COMPLEX
argument, computed here with mpmath. libintti's t quadrature needs no Boys
function at all; this script exists purely to give it something to be checked
against.

Prints reference values consumed by tests/test_giao.cpp.
"""

import numpy as np
import mpmath as mp

mp.mp.dps = 30


def boys_complex(m, T):
    """F_m(T) = int_0^1 u^{2m} exp(-T u^2) du, analytic in complex T."""
    T = mp.mpc(T)
    if abs(T) < mp.mpf("1e-18"):
        return mp.mpf(1) / (2 * m + 1)
    # F_m(T) = gamma(m+1/2) * P(m+1/2, T) / (2 T^{m+1/2}) with the lower
    # incomplete gamma; mpmath's gammainc handles complex arguments.
    return mp.gammainc(m + mp.mpf("0.5"), 0, T) / (2 * T ** (m + mp.mpf("0.5")))


def e_coeffs(la, lb, p, PA, PB, K):
    """MD expansion coefficients; PA, PB, K may be complex, p is real."""
    E = np.zeros((la + 1, lb + 1, la + lb + 2), dtype=complex)
    E[0, 0, 0] = K
    for i in range(la):
        for t in range(i + 2):
            E[i + 1, 0, t] = ((E[i, 0, t - 1] / (2 * p) if t > 0 else 0)
                              + PA * E[i, 0, t] + (t + 1) * E[i, 0, t + 1])
    for i in range(la + 1):
        for j in range(lb):
            for t in range(i + j + 2):
                E[i, j + 1, t] = ((E[i, j, t - 1] / (2 * p) if t > 0 else 0)
                                  + PB * E[i, j, t] + (t + 1) * E[i, j, t + 1])
    return E


def giao_pair(za, A, zb, B_, Bfield):
    """(p, complex centre P~, per-direction complex prefactor K~)."""
    A, B_, Bfield = map(np.asarray, (A, B_, Bfield))
    p = za + zb
    P = (za * A + zb * B_) / p
    K = 0.5 * np.cross(Bfield, B_ - A)          # gauge-origin independent
    Pt = P - 1j * K / (2 * p)
    mu = za * zb / p
    Kt = np.empty(3, dtype=complex)
    for d in range(3):
        Kt[d] = (np.exp(-mu * (A[d] - B_[d]) ** 2)
                 * np.exp(-K[d] ** 2 / (4 * p))
                 * np.exp(-1j * K[d] * P[d]))
    return p, Pt, Kt


def eri_giao(shells, Bfield, comps):
    """(ab|cd) for London orbitals; shells = [(z, centre, l), ...] x 4."""
    (za, A, la), (zb, B_, lb), (zc, C, lc), (zd, D_, ld) = shells
    a3, b3, c3, d3 = comps
    p, Pt, Kb = giao_pair(za, A, zb, B_, Bfield)
    q, Qt, Kk = giao_pair(zc, C, zd, D_, Bfield)
    rho = p * q / (p + q)
    PQ = Pt - Qt                                   # complex displacement
    T = rho * np.sum(PQ * PQ)                      # complex Boys argument

    N = sum(a3) + sum(b3) + sum(c3) + sum(d3)
    F = [complex(boys_complex(n, T)) for n in range(N + 1)]
    R = np.zeros((N + 1,) * 4, dtype=complex)
    for n in range(N + 1):
        R[n, 0, 0, 0] = (-2 * rho) ** n * F[n]
    for n in range(N - 1, -1, -1):
        for t in range(N - n + 1):
            for u in range(N - n - t + 1):
                for v in range(N - n - t - u + 1):
                    if t + 1 <= N - n:
                        R[n, t + 1, u, v] = ((t * R[n + 1, t - 1, u, v] if t else 0)
                                             + PQ[0] * R[n + 1, t, u, v])
                    if u + 1 <= N - n:
                        R[n, t, u + 1, v] = ((u * R[n + 1, t, u - 1, v] if u else 0)
                                             + PQ[1] * R[n + 1, t, u, v])
                    if v + 1 <= N - n:
                        R[n, t, u, v + 1] = ((v * R[n + 1, t, u, v - 1] if v else 0)
                                             + PQ[2] * R[n + 1, t, u, v])

    Eb = [e_coeffs(a3[d], b3[d], p, Pt[d] - A[d], Pt[d] - B_[d], Kb[d]) for d in range(3)]
    Ek = [e_coeffs(c3[d], d3[d], q, Qt[d] - C[d], Qt[d] - D_[d], Kk[d]) for d in range(3)]
    val = 0j
    for t in range(a3[0] + b3[0] + 1):
        for u in range(a3[1] + b3[1] + 1):
            for v in range(a3[2] + b3[2] + 1):
                eb = (Eb[0][a3[0], b3[0], t] * Eb[1][a3[1], b3[1], u]
                      * Eb[2][a3[2], b3[2], v])
                if eb == 0:
                    continue
                for tt in range(c3[0] + d3[0] + 1):
                    for uu in range(c3[1] + d3[1] + 1):
                        for vv in range(c3[2] + d3[2] + 1):
                            ek = (Ek[0][c3[0], d3[0], tt] * Ek[1][c3[1], d3[1], uu]
                                  * Ek[2][c3[2], d3[2], vv])
                            if ek == 0:
                                continue
                            val += (eb * ek * (-1) ** (tt + uu + vv)
                                    * R[0, t + tt, u + uu, v + vv])
    return 2 * np.pi ** 2.5 / (p * q * np.sqrt(p + q)) * val


# geometry shared with tests/test_giao.cpp
SH = [(0.9, [0.0, 0.1, -0.3], None),
      (1.3, [0.5, -0.2, 0.4], None),
      (1.1, [1.0, 0.8, 0.0], None),
      (0.6, [-0.4, 0.3, 1.1], None)]

CASES = [
    ("ssss", [(0, 0, 0)] * 4),
    ("psss", [(1, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]),
    ("ppss", [(1, 0, 0), (0, 1, 0), (0, 0, 0), (0, 0, 0)]),
    ("pppp", [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0)]),
    ("ddpp", [(2, 0, 0), (0, 1, 1), (1, 0, 0), (0, 1, 0)]),
]

if __name__ == "__main__":
    for Bz in (0.0, 0.1, 1.0):
        Bf = [0.0, 0.0, Bz]
        print(f"// B = (0, 0, {Bz})")
        for name, comps in CASES:
            shells = [(z, c, None) for (z, c, _) in SH]
            v = eri_giao(shells, Bf, comps)
            print(f"    {{{Bz}, \"{name}\", {{{v.real:.17e}, {v.imag:.17e}}}}},")
        print()
    # gauge-origin independence is manifest: K depends only on R_b - R_a
    print("// K = (1/2) B x (R_b - R_a) -> gauge-origin independent by construction")
