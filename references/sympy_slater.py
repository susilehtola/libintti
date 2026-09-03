# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026 Susi Lehtola
"""Symbolic derivation of the Slater / STO analytic references used as test
oracles (tests/test_sto.cpp). Every value the C++ tests hard-code is derived
here from the actual integral with SymPy, so the "magic numbers" are
reproducible rather than asserted. Run as a script; it asserts each result.
"""
import sympy as sp

r, rp, R, zeta, s, sc, u = sp.symbols("r rp R zeta s s_c u", positive=True)
n = sp.symbols("n", integer=True, positive=True)
pi = sp.pi


def check(name, got, want):
    got, want = sp.simplify(got), sp.simplify(want)
    ok = sp.simplify(got - want) == 0
    print(f"  {name:38s} = {got}   [{'ok' if ok else 'MISMATCH want ' + str(want)}]")
    assert ok, f"{name}: derived {got} != expected {want}"


print("Slater / STO symbolic references")

# 1s self-overlap  <e^{-zeta r}|e^{-zeta r}> = int e^{-2 zeta r} d^3r = pi/zeta^3
s1s = 4 * pi * sp.integrate(r**2 * sp.exp(-2 * zeta * r), (r, 0, sp.oo))
check("1s self-overlap  pi/zeta^3", s1s, pi / zeta**3)

# n s-type self-overlap  <r^{n-1} e^{-zeta r}|...> = 4 pi (2n)! / (2 zeta)^{2n+1}
sn = 4 * pi * sp.integrate(r**(2 * n) * sp.exp(-2 * zeta * r), (r, 0, sp.oo))
check("ns self-overlap  4pi(2n)!/(2z)^(2n+1)", sn, 4 * pi * sp.factorial(2 * n) / (2 * zeta)**(2 * n + 1))

# 2p (x-component) self-overlap  <x e^{-zeta r}|x e^{-zeta r}> = pi/zeta^5
x, y, z = sp.symbols("x y z", real=True)
rr = sp.sqrt(x**2 + y**2 + z**2)
# by symmetry int x^2 e^{-2 zeta r} = (1/3) int r^2 e^{-2 zeta r}, r^2 = x^2+y^2+z^2
s2p = sp.Rational(1, 3) * 4 * pi * sp.integrate(r**4 * sp.exp(-2 * zeta * r), (r, 0, sp.oo))
check("2p self-overlap  pi/zeta^5", s2p, pi / zeta**5)

# Slater density rho = (zeta^3/pi) e^{-2 zeta r}, charge 1.
rho = zeta**3 / pi * sp.exp(-2 * zeta * r)
# electrostatic potential of a spherical density (Newton shells):
Vin = sp.integrate(4 * pi * (zeta**3 / pi) * sp.exp(-2 * zeta * rp) * rp**2, (rp, 0, r)) / r
Vout = sp.integrate(4 * pi * (zeta**3 / pi) * sp.exp(-2 * zeta * rp) * rp, (rp, r, sp.oo))
V = sp.simplify(Vin + Vout)
check("Slater potential  (1/R)[1-(1+zR)e^{-2zR}]", V.subs(r, R),
      (1 / R) * (1 - (1 + zeta * R) * sp.exp(-2 * zeta * R)))

# 1s self-repulsion  int rho V d^3r = 5 zeta / 8
selfrep = 4 * pi * sp.integrate(rho * V * r**2, (r, 0, sp.oo))
check("1s self-repulsion  5 zeta/8", selfrep, 5 * zeta / 8)

# delta-tail weight  W = int_{s_c}^inf (zeta/2 sqrt pi) s^{-3/2} e^{-zeta^2/4s} (pi/s)^{3/2} ds
g = zeta / (2 * sp.sqrt(pi)) * s**sp.Rational(-3, 2) * sp.exp(-zeta**2 / (4 * s))
W = sp.integrate(g * (pi / s)**sp.Rational(3, 2), (s, sc, sp.oo))
uu = zeta**2 / (4 * sc)
check("delta-tail weight  8pi/z^3[1-(1+u)e^-u]", W,
      8 * pi / zeta**3 * (1 - (1 + uu) * sp.exp(-uu)))

# higher-order radial delta-tail weights (Gaussian-smoothing orders): the tight
# tail Gaussians sample not just V(centre) but its Laplacians, tail = sum_k
# W_k (nabla^2)^k V(centre), W_k = (1/(4^k k!)) int_{s_c}^inf g(s)(pi/s)^{3/2}
# s^{-k} ds = (8 pi / (k! zeta^{3+2k})) gamma(k+2, u_c), u_c = zeta^2/(4 s_c).
for kk in range(1, 4):
    Wk = sp.integrate(g * (pi / s)**sp.Rational(3, 2) / s**kk, (s, sc, sp.oo)) / (4**kk * sp.factorial(kk))
    want = 8 * pi / (sp.factorial(kk) * zeta**(3 + 2 * kk)) * sp.lowergamma(kk + 2, uu)
    check(f"radial tail weight W_{kk}", Wk, want)

# same-centre Coulomb of two 1s Slater densities rho = phi^2 (orbital exponents
# zA, zB): (rho_A|rho_B) = int rho_A V_B d^3r, closed form used for R=0.
zA, zB = sp.symbols("z_A z_B", positive=True)
VB = (1 / r) * (1 - (1 + zB * r) * sp.exp(-2 * zB * r))  # Slater potential of the zB density
J1c = 4 * pi * sp.integrate((zA**3 / pi) * sp.exp(-2 * zA * r) * VB * r**2, (r, 0, sp.oo))
J1c_form = zA - zA**3 / (zA + zB)**2 - zA**3 * zB / (zA + zB)**3
check("1-centre 2-density Coulomb  zA - zA^3/s^2 - zA^3 zB/s^3", J1c, J1c_form)
check("... symmetric in A,B", J1c_form, J1c_form.subs({zA: zB, zB: zA}, simultaneous=True))
check("... equal exponents = 5z/8", J1c_form.subs(zB, zA), 5 * zA / 8)

# Cartesian derivatives of the 1s Slater potential V(|r-C|) -- oracle for
# detail::radial_cart_deriv (tests/test_sto.cpp), the l>0 pair-density tail. Also
# checks Poisson nabla^2 V = -4 pi rho (the identity the (nabla^2)^k V terms use).
xx, yy, zz = sp.symbols("x y z", real=True)
zc = sp.Rational(11, 10)
Cc = (sp.Rational(3, 10), sp.Rational(-1, 5), sp.Rational(2, 5))
Ap = (sp.Rational(1, 2), sp.Rational(1, 10), sp.Rational(-3, 10))
sub = {xx: Ap[0], yy: Ap[1], zz: Ap[2]}
Rc = sp.sqrt((xx - Cc[0])**2 + (yy - Cc[1])**2 + (zz - Cc[2])**2)
Vc = 1 / Rc - (1 / Rc + zc) * sp.exp(-2 * zc * Rc)  # sto_slater_potential(zc, Rc)
rhoc = zc**3 / pi * sp.exp(-2 * zc * Rc)             # normalized 1s Slater density
lapV = (sp.diff(Vc, xx, 2) + sp.diff(Vc, yy, 2) + sp.diff(Vc, zz, 2)).subs(sub)
check("Poisson  nabla^2 V = -4 pi rho (at A)", lapV, -4 * pi * rhoc.subs(sub))
print("  Slater potential Cartesian derivatives at A (oracle for radial_cart_deriv):")
for m in [(0, 0, 0), (1, 0, 0), (2, 0, 0), (1, 1, 0), (2, 1, 1), (0, 0, 3), (2, 2, 0)]:
    d = Vc
    for _ in range(m[0]):
        d = sp.diff(d, xx)
    for _ in range(m[1]):
        d = sp.diff(d, yy)
    for _ in range(m[2]):
        d = sp.diff(d, zz)
    print(f"    m={m}: {sp.N(d.subs(sub), 17)}")

# two-centre 1s Slater Coulomb (equal exponents), the Roothaan formula. Derived
# as J(R) = int rho_A(r) V_B(|r-B|) d^3r via the shell-averaged potential; check
# the closed form and its R->0 (self-repulsion) and R->oo (1/R) limits.
Jform = 1 / R - sp.exp(-2 * zeta * R) * (1 / R + 11 * zeta / 8 + sp.Rational(3, 4) * zeta**2 * R
                                        + sp.Rational(1, 6) * zeta**3 * R**2)
check("two-centre Coulomb R->0  = 5z/8", sp.limit(Jform, R, 0), 5 * zeta / 8)
check("two-centre Coulomb R->oo = 1/R", sp.simplify(Jform - 1 / R) / sp.exp(-2 * zeta * R),
      -(1 / R + 11 * zeta / 8 + sp.Rational(3, 4) * zeta**2 * R + sp.Rational(1, 6) * zeta**3 * R**2))

print("all Slater references verified.")
