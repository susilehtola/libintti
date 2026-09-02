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

# --- higher angular momentum, general geometry (independent l>0 oracle) ------
# The raw (unnormalised) Gaussian-geminal three-electron integral of six
# Cartesian Gaussians is a pure polynomial x Gaussian moment: it factorises over
# Cartesian dimensions into a 3-variable (x1 on electron 1, x2 on 2, x3 on 3)
# integral that SymPy evaluates exactly for arbitrary centres and l. This gives
# an l>0 reference independent of the finite-difference centre-derivative check
# the C++ suite uses. Operator node convention: weight * exp(-t^2 r^2); here
# op12 = exp(-gamma r12^2), op13 = exp(-delta r13^2).
Q = sp.Rational
_x = sp.symbols("x1 x2 x3", real=True)


def _wick(idx, Sigma):
    """E[prod_k Y_{idx[k]}] for a zero-mean Gaussian with covariance Sigma."""
    if len(idx) % 2:
        return sp.Integer(0)
    if not idx:
        return sp.Integer(1)
    a0, rest = idx[0], idx[1:]
    return sum(Sigma[a0, rest[k]] * _wick(rest[:k] + rest[k + 1:], Sigma)
               for k in range(len(rest)))


def _gauss_moment_1d(poly, expo):
    """int poly * exp(expo) over R^3, expo a quadratic in x1,x2,x3, done in
    closed form: exp = -X^T A X + 2 B^T X + c, integral = pi^{3/2}/sqrt(detA)
    exp(c + B^T A^{-1} B) E[poly], X ~ N(A^{-1}B, A^{-1}/2)."""
    P = sp.Poly(-expo, *_x)  # X^T A X - 2 B^T X - c
    A = sp.zeros(3)
    B = sp.zeros(3, 1)
    c = -P.coeff_monomial(1)
    for i in range(3):
        A[i, i] = P.coeff_monomial(_x[i]**2)
        B[i] = -P.coeff_monomial(_x[i]) / 2
        for j in range(i + 1, 3):
            A[i, j] = A[j, i] = P.coeff_monomial(_x[i] * _x[j]) / 2
    Ainv = A.inv()
    mu = Ainv * B
    Sigma = Ainv / 2
    base = pi**sp.Rational(3, 2) / sp.sqrt(A.det()) * sp.exp(c + (B.T * mu)[0])
    # E[poly]: shift x_i -> mu_i + y_i, expand, apply Wick to each monomial
    y = sp.symbols("y1 y2 y3", real=True)
    shifted = sp.expand(poly.subs({_x[i]: mu[i] + y[i] for i in range(3)}, simultaneous=True))
    Py = sp.Poly(shifted, *y)
    Epoly = sum(coef * _wick([i for i, p in enumerate(mono) for _ in range(p)], Sigma)
                for mono, coef in Py.terms())
    return base * Epoly


def raw_gaussian_geminal(gaussians, gamma, delta):
    """gaussians = list of 6 dicts {a: exponent, R: (x,y,z), l: (lx,ly,lz)} for
    (a,b,c,d,e,f); electron 1 = (a,d), 2 = (b,e), 3 = (c,f)."""
    a, b, c, d, e, f = gaussians
    x1, x2, x3 = _x
    G = sp.Integer(1)
    for m in range(3):  # Cartesian dimension (integral factorises over axes)
        expo = -(a["a"] * (x1 - a["R"][m])**2 + d["a"] * (x1 - d["R"][m])**2
                 + b["a"] * (x2 - b["R"][m])**2 + e["a"] * (x2 - e["R"][m])**2
                 + c["a"] * (x3 - c["R"][m])**2 + f["a"] * (x3 - f["R"][m])**2
                 + gamma * (x1 - x2)**2 + delta * (x1 - x3)**2)
        poly = ((x1 - a["R"][m])**a["l"][m] * (x1 - d["R"][m])**d["l"][m]
                * (x2 - b["R"][m])**b["l"][m] * (x2 - e["R"][m])**e["l"][m]
                * (x3 - c["R"][m])**c["l"][m] * (x3 - f["R"][m])**f["l"][m])
        # evaluate each axis block numerically (the closed form is exact; a big
        # symbolic simplify of the rational product is pointlessly slow)
        G *= sp.N(_gauss_moment_1d(sp.expand(poly), expo), 30)
    return G


# a p_x on the P density (a), p_y on the Q density (b), p_z on the S density (c);
# distinct exponents and centres so no symmetry hides an index error.
bas = [
    {"a": Q(9, 10), "R": (Q(1, 10), 0, 0), "l": (1, 0, 0)},          # a  (electron 1)
    {"a": Q(11, 10), "R": (Q(3, 10), 0, 0), "l": (0, 1, 0)},         # b  (electron 2)
    {"a": Q(1), "R": (0, Q(2, 5), 0), "l": (0, 0, 1)},               # c  (electron 3)
    {"a": Q(7, 10), "R": (0, Q(1, 5), 0), "l": (0, 0, 0)},           # d  (electron 1)
    {"a": Q(8, 10), "R": (0, 0, Q(1, 10)), "l": (0, 0, 0)},          # e  (electron 2)
    {"a": Q(6, 10), "R": (Q(1, 5), 0, 0), "l": (0, 0, 0)},           # f  (electron 3)
]
# the l>0 many-centre value used by tests/test_threeel.cpp (HigherL...): a p_x on
# the P density (a), p_y on Q (b), p_z on S (c). Evaluated numerically per axis
# via the exact moment integrator; references/te_reference.py reproduces it by an
# independent Gauss-Hermite quadrature.
Graw_p = raw_gaussian_geminal(bas, Q(1, 2), Q(7, 10))
print(f"  raw Gaussian-geminal l>0 (px,py,pz)          = {sp.N(Graw_p, 20)}")
print(f"  -> tests/test_threeel.cpp regression value {sp.N(Graw_p, 17)}")

print("all three-electron references verified.")
