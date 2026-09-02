# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026 Susi Lehtola
"""Independent reference values for the three-electron integral engine
(include/intti/threeel.hpp), covering l>0 and many-centre configurations for the
plain integral, the r12^2 moment, and centre derivatives.

The references are computed by DIRECT numerical quadrature (scipy), independent
of the engine's analytic multivariate-Gaussian-moment algorithm. Gaussian-geminal
operators make the six-fold integral factorise over the three Cartesian axes into
a 3-variable (x1 = electron 1, x2 = electron 2, x3 = electron 3) block that
tplquad evaluates directly; 1/r operators are not used here (the Coulomb path is
the same te_core with a t-grid node list, and is pinned separately by the
independent erf-reduction value in tests/test_threeel.cpp).

Run with no argument to self-verify against a second quadrature; run with
`--emit <path>` to (re)generate tests/threeel_reference.hpp.
"""
import sys
import numpy as np

# basis function = (alpha, (cx,cy,cz), (lx,ly,lz)); order (a,b,c,d,e,f),
# electron 1 = (a,d), electron 2 = (b,e), electron 3 = (c,f).


def norm(alpha, l):
    """te_norm: N0 * prod (2l-1)!!^{-1/2}, N0 = pi^{-3/4} 2^{L+3/4} a^{L/2+3/4}."""
    L = sum(l)
    N0 = np.pi**-0.75 * 2.0**(L + 0.75) * alpha**(L / 2 + 0.75)
    g = 1.0
    for li in l:
        df = 1.0
        k = 2 * li - 1
        while k > 0:
            df *= k
            k -= 2
        g /= np.sqrt(df)
    return N0 * g


_GH_X, _GH_W = np.polynomial.hermite.hermgauss(10)  # exact to degree 19


def _block(basis, gamma, delta, m, r2):
    """One Cartesian-axis block int poly(x1,x2,x3) exp(-quadratic) dx1dx2dx3 by
    whitened tensor Gauss-Hermite: change x = mu + A^{-1/2} u so the exponent is
    -u^T u (separable), then a tensor GH rule is exact for the polynomial."""
    a, b, c, d, e, f = basis
    # exponent quadratic form: x^T Q x - 2 s^T x + c0, Q from a..f centres + geminals
    ca, cb, cc = a[1][m], b[1][m], c[1][m]
    cd, ce, cf = d[1][m], e[1][m], f[1][m]
    Q = np.array([[a[0] + d[0] + gamma + delta, -gamma, -delta],
                  [-gamma, b[0] + e[0] + gamma, 0.0],
                  [-delta, 0.0, c[0] + f[0] + delta]])
    s = np.array([a[0] * ca + d[0] * cd, b[0] * cb + e[0] * ce, c[0] * cc + f[0] * cf])
    c0 = (a[0] * ca**2 + d[0] * cd**2 + b[0] * cb**2 + e[0] * ce**2 + c[0] * cc**2 + f[0] * cf**2)
    Qi = np.linalg.inv(Q)
    mu = Qi @ s
    base = np.exp(s @ mu - c0) / np.sqrt(np.linalg.det(Q))  # int exp(-(x-mu)^T Q (x-mu)) = pi^{3/2}/sqrt|Q|
    w, V = np.linalg.eigh(Q)
    half_inv = V @ np.diag(w**-0.5) @ V.T  # Q^{-1/2}, x = mu + half_inv u, exp -> -u^T u
    la = (a[2][m], b[2][m], c[2][m], d[2][m], e[2][m], f[2][m])
    tot = 0.0
    for i, ui in enumerate(_GH_X):
        for j, uj in enumerate(_GH_X):
            for k, uk in enumerate(_GH_X):
                x = mu + half_inv @ np.array([ui, uj, uk])
                x1, x2, x3 = x
                p = ((x1 - ca)**la[0] * (x1 - cd)**la[3] * (x2 - cb)**la[1]
                     * (x2 - ce)**la[4] * (x3 - cc)**la[2] * (x3 - cf)**la[5])
                if r2:
                    p *= (x1 - x2)**2
                tot += _GH_W[i] * _GH_W[j] * _GH_W[k] * p
    return base * tot


def raw(basis, gamma, delta, moment12=False):
    """Unnormalised three-electron Gaussian-geminal integral (or its r12^2
    moment): G = prod_m block_m; moment = sum_m block_m[r2] prod_{m'!=m} block_m'."""
    I = [_block(basis, gamma, delta, m, False) for m in range(3)]
    if not moment12:
        return I[0] * I[1] * I[2]
    J = [_block(basis, gamma, delta, m, True) for m in range(3)]
    return J[0] * I[1] * I[2] + I[0] * J[1] * I[2] + I[0] * I[1] * J[2]


def _N(basis):
    return np.prod([norm(g[0], g[2]) for g in basis])


def integral(basis, gamma, delta):
    return _N(basis) * raw(basis, gamma, delta)


def moment(basis, gamma, delta):
    return _N(basis) * raw(basis, gamma, delta, moment12=True)


def deriv(basis, which, dim, gamma, delta):
    """d/d(center) of the UNNORMALISED integral, 5-point central difference. The
    engine reproduces this analytically via the McMurchie-Davidson centre shift
    d/dA_x chi = 2 alpha chi_{l+1} - l chi_{l-1}, so the C++ test forms
    2 alpha raw(l+1) - l raw(l-1) and compares to this independent value."""
    def shifted(h):
        b = [list(g) for g in basis]
        cen = list(b[which][1]); cen[dim] += h; b[which][1] = tuple(cen)
        return raw([tuple(g) for g in b], gamma, delta)
    h = 1e-3
    return (-shifted(2 * h) + 8 * shifted(h) - 8 * shifted(-h) + shifted(-2 * h)) / (12 * h)


# ---- battery of configurations -------------------------------------------
S = (0, 0, 0)
CENT = [(0.0, 0.0, 0.0), (0.4, 0.0, 0.0), (0.0, 0.3, 0.5),
        (0.1, 0.0, 0.0), (0.4, 0.1, 0.0), (0.0, 0.3, 0.6)]
Z = [0.9, 1.1, 0.7, 1.3, 0.8, 1.0]


def cfg(ls):
    return [(Z[i], CENT[i], ls[i]) for i in range(6)]


CONFIGS = {
    "s_manycentre": cfg([S, S, S, S, S, S]),
    "px_on_a": cfg([(1, 0, 0), S, S, S, S, S]),
    "py_on_b": cfg([S, (0, 1, 0), S, S, S, S]),
    "pz_on_c": cfg([S, S, (0, 0, 1), S, S, S]),
    "px_on_d": cfg([S, S, S, (1, 0, 0), S, S]),
    "py_on_e": cfg([S, S, S, S, (0, 1, 0), S]),
    "pz_on_f": cfg([S, S, S, S, S, (0, 0, 1)]),
    "pxpypz_abc": cfg([(1, 0, 0), (0, 1, 0), (0, 0, 1), S, S, S]),
    "dxx_on_a": cfg([(2, 0, 0), S, S, S, S, S]),
    "dxy_on_a": cfg([(1, 1, 0), S, S, S, S, S]),
    "p_on_P_both": cfg([(1, 0, 0), S, S, (0, 1, 0), S, S]),
    "mixed_high": cfg([(1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0), S, S]),
}
GAMMA, DELTA = 0.7, 0.5
# centre derivatives to check: (config, which function 0..5, dim 0..2)
DERIVS = [("s_manycentre", 0, 0), ("s_manycentre", 1, 1), ("s_manycentre", 2, 2),
          ("px_on_a", 0, 0), ("pxpypz_abc", 3, 2), ("dxy_on_a", 0, 1)]


_GL_X, _GL_W = np.polynomial.legendre.leggauss(96)  # fixed rule on [-L, L]


def _gl_block(basis, gamma, delta, m, r2, L=8.0):
    """Independent second method: a fixed high-order Gauss-Legendre tensor rule
    on a box (numpy, vectorised, fast). Different weight function and nodes from
    the whitened Gauss-Hermite rule, so it anchors it without sharing math."""
    a, b, c, d, e, f = basis
    x = L * _GL_X
    w = L * _GL_W
    X1, X2, X3 = np.meshgrid(x, x, x, indexing="ij")
    W = np.einsum("i,j,k->ijk", w, w, w)
    E = (a[0] * (X1 - a[1][m])**2 + d[0] * (X1 - d[1][m])**2
         + b[0] * (X2 - b[1][m])**2 + e[0] * (X2 - e[1][m])**2
         + c[0] * (X3 - c[1][m])**2 + f[0] * (X3 - f[1][m])**2
         + gamma * (X1 - X2)**2 + delta * (X1 - X3)**2)
    p = ((X1 - a[1][m])**a[2][m] * (X1 - d[1][m])**d[2][m]
         * (X2 - b[1][m])**b[2][m] * (X2 - e[1][m])**e[2][m]
         * (X3 - c[1][m])**c[2][m] * (X3 - f[1][m])**f[2][m])
    if r2:
        p = p * (X1 - X2)**2
    return float(np.sum(W * p * np.exp(-E)))


def selfcheck():
    # anchor the fast whitened Gauss-Hermite rule against an independent fixed
    # Gauss-Legendre tensor rule, across plain and l>0 configs (integral + r12^2).
    checks = [("s_manycentre", 0, False), ("s_manycentre", 1, False), ("s_manycentre", 2, False),
              ("px_on_a", 0, False), ("px_on_a", 0, True), ("dxy_on_a", 0, False),
              ("pxpypz_abc", 2, False), ("mixed_high", 0, True)]
    for name, m, r2 in checks:
        b = CONFIGS[name]
        gh = _block(b, GAMMA, DELTA, m, r2)
        gl = _gl_block(b, GAMMA, DELTA, m, r2)
        assert abs(gh - gl) < 1e-9 * (abs(gl) + 1), (name, m, r2, gh, gl)
    print("self-check: Gauss-Hermite matches Gauss-Legendre (%d anchors)." % len(checks))


def emit(path):
    lines = ["// SPDX-License-Identifier: BSD-3-Clause",
             "// GENERATED by references/te_reference.py -- do not edit by hand.",
             "// Independent (scipy tplquad) three-electron reference values.",
             "#pragma once", "#include <vector>", "", "namespace te_ref {",
             "struct Fn { double alpha; double c[3]; int l[3]; };",
             "struct Case { const char *name; Fn f[6]; double integral; double moment; };",
             f"inline constexpr double gamma_op = {GAMMA!r};",
             f"inline constexpr double delta_op = {DELTA!r};", "",
             "inline const std::vector<Case> cases = {"]
    for name, b in CONFIGS.items():
        Ival = integral(b, GAMMA, DELTA)
        Mval = moment(b, GAMMA, DELTA)
        fns = ", ".join("{%r, {%r, %r, %r}, {%d, %d, %d}}" % (g[0], g[1][0], g[1][1], g[1][2],
                                                              g[2][0], g[2][1], g[2][2]) for g in b)
        lines.append('  {"%s", {%s}, %.17g, %.17g},' % (name, fns, Ival, Mval))
    lines += ["};", "",
              "struct DCase { const char *name; int which; int dim; double value; };",
              "inline const std::vector<DCase> dcases = {"]
    for name, which, dim in DERIVS:
        dv = deriv(CONFIGS[name], which, dim, GAMMA, DELTA)
        lines.append('  {"%s", %d, %d, %.17g},' % (name, which, dim, dv))
    lines += ["};", "", "} // namespace te_ref", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {path}: {len(CONFIGS)} integral/moment cases, {len(DERIVS)} derivative cases.")


if __name__ == "__main__":
    selfcheck()
    if len(sys.argv) == 3 and sys.argv[1] == "--emit":
        emit(sys.argv[2])
