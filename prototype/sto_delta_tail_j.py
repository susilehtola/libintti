# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026 Susi Lehtola
"""What it takes to fold the delta-tail into the STO J/K matrix build.

Finding (this script): the two-electron delta-tail must be applied at the
PAIR-DENSITY level (each product density rho = phi_mu phi_nu expanded and
truncated on its own s-grid), NOT by truncating the orbital primitives and
letting coulomb_build square them.

Why the orbital-level matrix build (sto_jk_build with a truncated grid) fails:
truncating each orbital at s_c and forming (phi^low)^2 is a DIFFERENT truncation
than truncating the pair density e^{-2 zeta r} at t_c. The orbital square drops
the "one orbital tight" pairs (pair exponent s_k + s_l with one node > s_c) in a
way the point-charge delta correction cannot repair on the diagonal; the four
leading correction terms plateau at ~1% with a systematic overshoot (the
bra-tail term double-counts the on-centre density self-potential).

Why the density level works -- even on-centre (R=0): treating rho = e^{-2 zeta r}
directly, a truncated s-grid plus the delta-tail weight reproduces the analytic
1s self-repulsion 5 zeta/8, the hardest cusp-cusp case:

    t_c = 8 :  low -5.6 %,  delta +0.64 %   (residual = tail-tail self-energy)
    t_c = 20:  low -1.1 %,  delta +0.01 %

So the on-centre cusp IS capturable by the delta-tail, at the density level.

Path to a production delta-tail J (a re-architecture, not a drop-in):
  * form pair densities rho_ab = phi_a phi_b, not orbital primitives;
    - same-centre pairs are single-centre Slater densities e^{-(z_a+z_b) r}:
      density-level delta-tail (this script), exact incl. the R=0 self term;
    - two-centre pairs are diffuse overlap distributions: no cusp-cusp, the
      tail is a point charge W_a phi_b(A) at each atom (the clean 2c case);
  * (ab|cd) = pair-pair Coulomb of the (delta-tailed) pair densities;
  * the residual tail-tail self-energy (t_c small) needs the closed-form
    Coulomb self-energy of the tail density, still to be derived.
This is the honest scope; the current tree keeps Schwarz screening as the
correct exact cost control and does not ship an approximate delta-tail J.
"""
import numpy as np
import math
from numpy.polynomial.legendre import leggauss


def sto_g(zeta, ns, smin=1e-4, smax=1e6):
    x, w = leggauss(ns)
    lo, hi = np.log(smin), np.log(smax)
    u = 0.5 * (hi - lo) * (x + 1) + lo
    s = np.exp(u)
    ws = 0.5 * (hi - lo) * w * s
    c = zeta / (2 * np.sqrt(np.pi)) * ws * s ** -1.5 * np.exp(-zeta ** 2 / (4 * s))
    return s, c


def boys0(x):
    return 1 - x / 3 if x < 1e-13 else 0.5 * math.sqrt(math.pi / x) * math.erf(math.sqrt(x))


def ss(p, q, R2):
    return 2 * math.pi ** 2.5 / (p * q * math.sqrt(p + q)) * boys0(p * q / (p + q) * R2)


def wtail(zeta, sc):
    u = zeta ** 2 / (4 * sc)
    return 8 * math.pi / zeta ** 3 * (1 - (1 + u) * math.exp(-u))


def self_repulsion(zeta, tc, ns_low):
    """1s self-repulsion (rho|rho) at R=0 by density-level delta-tail."""
    CA = zeta ** 3 / math.pi
    tl, dl = sto_g(2 * zeta, ns_low, 1e-4, tc)
    low = sum(CA * dl[k] * CA * dl[m] * ss(tl[k], tl[m], 0.0)
              for k in range(len(tl)) for m in range(len(tl)))
    Q = CA * wtail(2 * zeta, tc)
    return low + 2 * Q * zeta   # + tail-tail self (dropped): the residual


if __name__ == "__main__":
    zeta = 1.3
    exact = 5 * zeta / 8
    print(f"1s self-repulsion, zeta={zeta}: analytic 5 zeta/8 = {exact:.8f}")
    for tc in (4.0, 8.0, 20.0):
        val = self_repulsion(zeta, tc, 24)
        print(f"  density-level delta-tail t_c={tc:4.0f}: {val:.8f}  "
              f"rel err {(val - exact) / exact:+.2e}")
    print("=> the on-centre cusp-cusp is capturable at the DENSITY level; a "
          "production delta-tail J must be built from pair densities.")
