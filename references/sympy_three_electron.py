# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026 Susi Lehtola
"""Symbolic derivation of the one-centre three-electron references used as test
oracles (tests/test_threeel.cpp). The six-index integral is
  G_{abcdef} = <a(1)b(2)c(3)| r12^{-1} r13^{-1} |d(1)e(2)f(3)>,
electron 1 = (a,d), 2 = (b,e), 3 = (c,f). Two cases are derived from the actual
integral with SymPy: the Coulomb-Coulomb one-centre value 4 zeta/3, and the
Gaussian-geminal closed form that the (t,s)-quadrature engine reproduces.
"""
import sympy as sp

r, zeta = sp.symbols("r zeta", positive=True)
pi = sp.pi


def check(name, got, want):
    got, want = sp.simplify(got), sp.simplify(want)
    ok = sp.simplify(got - want) == 0
    print(f"  {name:44s} = {got}   [{'ok' if ok else 'MISMATCH want ' + str(want)}]")
    assert ok, f"{name}: derived {got} != expected {want}"


print("Three-electron one-centre symbolic references")

# --- Coulomb-Coulomb, all six normalised s-GTOs on one centre ---------------
# The basis functions are Gaussians chi = (2z/pi)^{3/4} e^{-z r^2}. Each
# electron's pair density chi_a chi_d = (2z/pi)^{3/2} e^{-2z r^2} = rho_G is a
# unit-charge Gaussian (exponent alpha = 2z). Electrons 2 and 3 couple to 1
# only, so integrate them out into the Gaussian Coulomb potential
# V_G(r) = erf(sqrt(2z) r)/r, leaving G = int rho_G(r) V_G(r)^2 d^3r.
z = zeta
rho_G = (2 * z / pi)**sp.Rational(3, 2) * sp.exp(-2 * z * r**2)
V_G = sp.erf(sp.sqrt(2 * z) * r) / r
G_coulomb = sp.integrate(4 * pi * rho_G * V_G**2 * r**2, (r, 0, sp.oo))
check("G_aaaaaa (Coulomb-Coulomb)  4 zeta/3", G_coulomb, 4 * zeta / 3)

# --- Gaussian-geminal, all six identical s-GTOs on one centre ---------------
# Operators f12 = e^{-gamma r12^2}, f13 = e^{-delta r13^2}. Pair density per
# electron is N0^2 e^{-alpha r^2}, alpha = 2z. The 9D Gaussian integral
# factorises over Cartesian dimensions into three identical (x1,x2,x3) blocks;
# each block is a 3-variable Gaussian, evaluate it via its quadratic form
#   E = (a+g+d) x1^2 + (a+g) x2^2 + (a+d) x3^2 - 2g x1 x2 - 2d x1 x3 = X^T A X,
# so the block integral is pi^{3/2}/sqrt(det A). This is the engine's M-core.
al, gam, dlt = sp.symbols("alpha gamma delta", positive=True)
A = sp.Matrix([[al + gam + dlt, -gam, -dlt],
               [-gam, al + gam, 0],
               [-dlt, 0, al + dlt]])
block = pi**sp.Rational(3, 2) / sp.sqrt(A.det())
LQ = al * gam / (al + gam)   # Mehine's Lambda_Q = alpha t^2/(t^2+alpha), t^2 -> gamma
LS = al * dlt / (al + dlt)
block_closed = pi**sp.Rational(3, 2) / sp.sqrt((al + gam) * (al + dlt) * (al + LQ + LS))
check("Gaussian-geminal block  pi^{3/2}/sqrt(detA)", block, block_closed)
# full 3D one-centre integral (per unit N0^6): the engine's M-function core
G_gauss = block**3
check("G one-centre (per N0^6)  pi^{9/2}/[...]^{3/2}", G_gauss,
      pi**sp.Rational(9, 2) / ((al + gam) * (al + dlt) * (al + LQ + LS))**sp.Rational(3, 2))

print("all three-electron references verified.")
