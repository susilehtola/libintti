# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026 Susi Lehtola
"""Symbolic derivation of the higher-order Losilla delta-tail corrections for
the truncated Coulomb t-quadrature (tests/test_tail.cpp).

The Coulomb kernel is 1/r = (2/sqrt(pi)) int_0^inf e^{-t^2 r^2} dt. Truncating
the explicit quadrature at t_c leaves the tail

    R(t_c) = (2/sqrt(pi)) int_{t_c}^inf G(t) dt,
    G(t)   = <rho_ab | e^{-t^2 r12^2} | rho_cd>   (the smoothed pair overlap).

For large t the Gaussian e^{-t^2 u^2} is a sharp smoothing kernel; expanding the
ket density under it in Gaussian moments gives

    G(t) = (sqrt(pi)/t)^3 * sum_{k>=0} M_k / (4^k k! t^{2k}),
    M_k  = int rho_ab (nabla^2)^k rho_cd   (Laplacian-overlap moments),

so integrating the tail term by term,

    R(t_c) = pi * sum_{k>=0} M_k / (4^k k! (k+1) t_c^{2k+2}).

k=0 is Losilla's leading delta tail pi*S/t_c^2 (S = M_0 the four-orbital
overlap); k=1 adds pi*L/(8 t_c^4) (L = M_1); k=2 adds pi*Q/(96 t_c^6). Each
extra order removes one power of 1/t_c^2 from the residual, so the explicit
[0, t_c] grid can be truncated much harder for the same accuracy.

This script derives the series coefficients and the M_k moments symbolically,
asserts each, and prints the oracle values hard-coded in the C++ test.
"""
import sympy as sp

pi = sp.pi


def check(name, got, want):
    # rewrite erfc->erf so erf(a)+erfc(a) collapses to 1 before comparing
    got, want = got.rewrite(sp.erf), want.rewrite(sp.erf)
    got, want = sp.simplify(got), sp.simplify(want)
    ok = sp.simplify(got - want) == 0
    print(f"  {name:44s} = {got}   [{'ok' if ok else 'MISMATCH want ' + str(want)}]")
    assert ok, f"{name}: derived {got} != expected {want}"


print("Higher-order delta-tail (Losilla) symbolic references\n")

# ---------------------------------------------------------------------------
# 1. Series coefficients from the Gaussian-smoothing expansion.
#    The k-th moment coefficient in G(t) is c_k = 1/(4^k k!); the tail
#    integral multiplies it by 1/(k+1) and moves 2/sqrt(pi)*pi^{3/2}=2*pi to
#    the front, giving the tail coefficient pi/(4^k k! (k+1)) / t_c^{2k+2}.
# ---------------------------------------------------------------------------
print("[1] series coefficients c_k (in G) and tail coeff a_k")
k = sp.symbols("k", nonnegative=True, integer=True)
# Isotropic Gaussian moment <(u.grad)^{2k}>/(2k)! under e^{-t^2 u^2} in 3D:
#   <u_i1...u_i2k> = (1/(2 t^2))^k * (sum over (2k-1)!! pairings of deltas),
#   contracted with grad^{2k} -> (2k-1)!! (nabla^2)^k, and (2k-1)!!=(2k)!/(2^k k!)
c_k = sp.Rational(1, 1) / sp.factorial(2 * k) * sp.factorial(2 * k) / (2**k * sp.factorial(k)) / 2**k
check("c_k = 1/(4^k k!)", c_k, 1 / (4**k * sp.factorial(k)))
# tail: (2/sqrt(pi)) * pi^{3/2} * c_k * int_{t_c}^inf t^{-(3+2k)} dt
tc = sp.symbols("t_c", positive=True)
a_k = (2 / sp.sqrt(pi)) * pi**sp.Rational(3, 2) * c_k * sp.integrate(
    sp.symbols("t", positive=True)**(-(3 + 2 * k)), (sp.symbols("t", positive=True), tc, sp.oo))
check("tail a_k = pi/(4^k k!(k+1) t_c^{2k+2})", a_k,
      pi / (4**k * sp.factorial(k) * (k + 1) * tc**(2 * k + 2)))
for kk, want in [(0, pi / tc**2), (1, pi / (8 * tc**4)), (2, pi / (96 * tc**6))]:
    check(f"  a_{kk}", a_k.subs(k, kk), want)

# ---------------------------------------------------------------------------
# 2. Verify the expansion against the EXACT smoothed overlap G(t) for a
#    concrete (ss|ss) system: G(t) = pi^3 / D^{3/2} exp(-t^2 pq R^2 / D),
#    D = pq + t^2 (p+q). Expand at large t and match sum_k M_k/(4^k k! t^{2k}).
#    Concrete rational exponents keep the 3D integrals and the 1/t series fast;
#    the coefficient structure in [1] is what is general.
# ---------------------------------------------------------------------------
print("\n[2] (ss|ss) large-t expansion vs Laplacian moments M_k  (p=13/10, q=4/5, R=11/10)")
pv, qv, Rv = sp.Rational(13, 10), sp.Rational(4, 5), sp.Rational(11, 10)
t = sp.symbols("t", positive=True)
rho_v = pv * qv / (pv + qv)
D = pv * qv + t**2 * (pv + qv)
G = pi**3 / D**sp.Rational(3, 2) * sp.exp(-t**2 * pv * qv * Rv**2 / D)

# Laplacian-overlap moments M_k = int e^{-p(r-P)^2} (nabla^2)^k e^{-q(r-Q)^2} d^3r,
# |P-Q| = R on the x axis (P at origin, Q at (R,0,0)); the y,z integrals are plain.
x, y, z = sp.symbols("x y z", real=True)
bra = sp.exp(-pv * (x**2 + y**2 + z**2))
ket = sp.exp(-qv * ((x - Rv)**2 + y**2 + z**2))


def lap(f):
    return sp.diff(f, x, 2) + sp.diff(f, y, 2) + sp.diff(f, z, 2)


def moment(f):
    return sp.integrate(sp.integrate(sp.integrate(f, (x, -sp.oo, sp.oo)),
                                     (y, -sp.oo, sp.oo)), (z, -sp.oo, sp.oo))


M = {}
kd = ket
for kk in range(3):
    M[kk] = sp.simplify(moment(bra * kd))
    kd = sp.expand(lap(kd))
# closed forms (independently derived): S=(pi/(p+q))^{3/2} e^{-rho R^2};
# L = S (pq/(p+q))(4 rho R^2 - 6).
S_cf = (pi / (pv + qv))**sp.Rational(3, 2) * sp.exp(-rho_v * Rv**2)
L_cf = S_cf * (pv * qv / (pv + qv)) * (4 * rho_v * Rv**2 - 6)
check("M_0 = S closed form", M[0], S_cf)
check("M_1 = L closed form", M[1], L_cf)

# match the large-t series of G(t) to sum_k M_k /(4^k k! t^{2k}) * (sqrt(pi)/t)^3
print("  matching G(t) series coefficients to M_k / (4^k k!):")
lead = (sp.sqrt(pi) / t)**3
ser = sp.series(G / lead, t, sp.oo, 8).removeO()
for kk in range(3):
    coeff = ser.coeff(t, -2 * kk)
    check(f"  [t^-{2*kk}] coeff = M_{kk}/(4^{kk} {kk}!)", coeff,
          M[kk] / (4**kk * sp.factorial(kk)))

# ---------------------------------------------------------------------------
# 3. Oracle numbers for the C++ test (tests/test_tail.cpp): the same concrete
#    (ss|ss) quartet's S, L, Q and the exact ERI. The C++ test builds truncated
#    grids with K-term tails and asserts the 1/t_c^{2K+2} residual scaling.
# ---------------------------------------------------------------------------
print("\n[3] concrete oracle (ss|ss): p=1.3, q=0.8, R=1.1")
xarg = rho_v * Rv**2
F0 = sp.sqrt(pi / xarg) * sp.erf(sp.sqrt(xarg)) / 2
eri = 2 * pi**sp.Rational(5, 2) / (pv * qv * sp.sqrt(pv + qv)) * F0
for nm, v in [("S = M_0", M[0]), ("L = M_1", M[1]), ("Q = M_2", M[2]),
              ("exact (ss|ss)", eri)]:
    print(f"    {nm:16s} = {sp.N(v, 17)}")

# ---------------------------------------------------------------------------
# 4. (nabla^2)^k of a GTO product component at a point -- oracle for the
#    GTO x cloud higher-order tail (detail::pair_component_laplacians). The
#    product p_x(A) * s(B): rho(r) = (x-Ax) exp(-aa|r-A|^2 - ab|r-B|^2).
# ---------------------------------------------------------------------------
print("\n[4] (nabla^2)^k of GTO product p_x(A)*s(B) at a point (pair_component_laplacians)")
aa, ab = sp.Rational(9, 10), sp.Rational(13, 10)
A = (sp.Rational(1, 10), sp.Rational(-1, 5), sp.Rational(3, 10))
Bc = (sp.Rational(1, 2), sp.Rational(2, 5), sp.Rational(-1, 10))
rpt = (sp.Rational(1, 5), sp.Rational(3, 20), sp.Rational(1, 4))
rho_g = (x - A[0]) * sp.exp(-aa * ((x - A[0])**2 + (y - A[1])**2 + (z - A[2])**2)
                            - ab * ((x - Bc[0])**2 + (y - Bc[1])**2 + (z - Bc[2])**2))
cur = rho_g
for kk in range(3):
    v = cur.subs({x: rpt[0], y: rpt[1], z: rpt[2]})
    print(f"    (nabla^2)^{kk} rho = {sp.N(v, 17)}")
    cur = sp.expand(sp.diff(cur, x, 2) + sp.diff(cur, y, 2) + sp.diff(cur, z, 2))

print("\nAll higher-order delta-tail references verified.")
