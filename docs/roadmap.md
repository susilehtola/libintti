# libintti roadmap

Status: M1–M9 delivered (quartets, batching, Cholesky, PSC/grid interactions,
GIAOs, quad precision, libcint façade — ERIs match PySCF to 2×10⁻¹⁴, an RHF
driven entirely by libintti matches PySCF to 1.4×10⁻¹³ Ha). The **RI +
property + derivative + local-hybrid program** (matrix-level API; see the
approved plan) is under way and mostly delivered:

- **M10 done** — 1e property matrices (overlap, kinetic, multipoles, and
  angular momentum ⟨μ|(r−O)×∇|ν⟩ vs PySCF `int1e_cg_irxp`), `oneel.hpp`.
- **M11 done** — screened nuclear attraction + Coulomb-potential collocation, `nuclear.hpp`.
- **M12 done** — 2-/3-center Coulomb (ghost-shell), `ncenter.hpp`.
- **M13 done** — RI J/K + occupation-driven RI-K, `ri.hpp`.
- **M14 done** — local & locally range-separated exchange energy density, `localhybrid.hpp`.
- **M16 (in progress)** — derivative integrals, `deriv.hpp`: one-electron
  gradient set complete (overlap/kinetic/nuclear, vs PySCF `int1e_ip*` and
  finite difference); two-electron gradient (`erigrad.hpp`) and arbitrary-order
  geometric derivatives (`geoderiv.hpp`) done. GIAO ∂/∂B started (`giao.hpp`):
  complex overlap S(B) in a finite field and the analytic dS/dB at B=0, vs
  PySCF `int1e_igovlp` (2.6e-16) and finite difference of S(B). Remaining:
  RI/2e gradients into the Fock build and Hessians. GIAO dV/dB done: complex
  nuclear V(B) and the analytic dV/dB (multiplicative operator, so phase-only
  like the overlap), vs PySCF int1e_ignuc (1.2e-14) and finite difference.
  GIAO dT/dB done: -1/2 nabla^2 also differentiates the London phase, so
  dT/dB carries a gradient term beyond the phase-weighted kinetic
  (kinetic_moment_matrices + gradient_matrices). The exact finite-field T(B)
  is built from the momentum form and validated against an independent
  real-space grid integral (libcint's int1e_igkin is a one-sided g convention
  and does not match the full symmetric derivative, so it is not used as the
  oracle here); the analytic dT/dB matches finite difference of T(B).
  Two-electron dJ/dB, dK/dB done (`giao2e.hpp`, matrix-level: density in,
  imaginary first-order J/K matrices out): 1/r12 is multiplicative so the
  field differentiates only the two pair phases, giving i times a combination
  of real position-weighted (bra-promoted) ERIs contracted with the density.
  Validated vs finite difference of the exact complex finite-field GIAO
  J(B)/K(B) built by direct quartet summation. Two-electron energy Hessian
  done (`erihess.hpp`): the MD centre-shift applied twice (l+/-2 same shell,
  l+/-1 x l+/-1 cross shells), contracted with the two-particle density into
  a (3 ns) x (3 ns) matrix; validated vs finite difference of the 2e gradient,
  symmetric, translationally invariant. One-electron overlap and kinetic
  Hessians, the nuclear-attraction electronic Hessian, and the
  nuclear-repulsion Hessian done (`geohess.hpp`): the order-2 geoderiv blocks
  routed per shell into a (3 ns) x (3 ns) matrix over the bra/ket-centre
  families (weighted by the energy-weighted density / density); the
  nuclear-attraction Hessian adds the operator-centre derivative via
  translational invariance (d/dR_C = -(d_A + d_B) per single-charge term, so
  no new integral is needed); all validated vs a finite-difference second
  difference. This completes the **integral/skeleton Hessian**: the
  frozen-density second-derivative contributions of every energy term.
  RI (density-fitting) Coulomb gradient done (`rigrad.hpp`): the standard DF-J
  force dE_J/dx = sum_P gamma_P dd_P/dx - 1/2 sum_PQ gamma_P gamma_Q dM_PQ/dx
  (gamma = M^{-1} d), assembled from derivatives of the ghost-augmented 2-/3-
  centre Coulomb integrals via the erigrad centre-shift (the zero-exponent
  ghost never moves); returns per-shell forces over the orbital and auxiliary
  bases. Validated vs finite difference of the ri_fit/ri_jk energy and
  translationally invariant. Remaining: the RI-K (exchange) gradient (double
  density contraction through a 3-index intermediate).

**Scope boundary.** libintti stops at the integral/matrix level: derivative
integrals and the skeleton (frozen-density) energy-derivative contractions
(gradient/Hessian of an energy term given a density; the first-order Fock
builders). The CPHF/CP-SCF orbital-response solve that turns a skeleton
Hessian or a first-order Fock into true frequencies / magnetic properties is
the host SCF code's responsibility (it is needed independently of who supplies
the integrals) and is deliberately **not** in libintti; libintti only feeds it
the matrix-level pieces it consumes.

**Ground rules (binding for any implementer):** every deliverable has a hard
oracle (PySCF or analytic) *and* an independent second check (finite
difference or an internal identity) — the RI eigenvector-transpose bug was
invisible to self-consistency alone. Never loosen a tolerance to proceed;
report and stop. Matrix-level API only (no per-quartet public entry point).
Follow the codebase idiom (header-only templates, Kokkos for float/double,
serial fallback for complex/quad/class scalars, BSD-3-Clause + SPDX).

## Remaining program milestones

- **M15 — NAO unification via fitting** (`nao.hpp`): fit NAO products to the
  GTO auxiliary set in the Coulomb metric (grid/PSC path builds the fit
  once), so all NAO integrals reduce to GTO 2-/3-center RI; grid path kept as
  the exact reference. Oracle: NAO-via-fit vs NAO-via-grid.

- **M-STO — Slater-type orbitals** (new, folded in 2026): an STO is the exact
  integral transform of a Gaussian,
  `e^{-ζr} = (ζ/2√π) ∫₀^∞ s^{-3/2} e^{-ζ²/4s} e^{-s r²} ds`,
  i.e. a **quadrature-contracted GTO** over an auxiliary radial variable s.
  An STO integral is then a nested s×t quadrature (an s-integral per STO,
  nested in the Coulomb t-integral). **Key structural reuse:** the large-s
  tail of the s-expansion is *delta-like* (the tight Gaussians approach
  δ³(r); verified — 100% of the truncation error sits at the cusp r<0.5), so
  it is the **same t→∞ delta-function tail correction of Losilla et al.** we
  already apply to the Coulomb quadrature (`tgrid.hpp`), now applied to the
  s-quadrature — the tight components need no brute quadrature and are
  spatially local (they screen). Reuses: `eri_quartets_accumulate`
  (STO = contracted GTO shell), the delta-tail correction, the Möbius/log
  grid mappings. Two complementary routes: (a) the integral-transform route
  above, exact in the node limit; (b) RI-fit STO products to GTOs like NAOs.
  Plan: Python-prototype the nested s×t + delta-tail against analytic Slater
  integrals (e.g. the 1s self-repulsion 5ζ/8) first, pin the s-grid and node
  counts, then implement a matrix-level STO basis on the existing engine.
  User is interested in STOs *widely* — treat as a first-class basis type
  alongside GTOs/NAOs, not a niche add-on.

- **M17 — spin-orbit** (`soc.hpp`): 1e SO off moment+t-quad; 2e SO
  (SOMF/Breit-Pauli) as the new operator.

- **M18 — GPU + performance engineering**: table-driven bit-packed Hermite
  recurrence and batched-GEMM contractions (libintX-style, architecture only
  — GPL, do not copy code), optional analytic Chebyshev-Boys fast kernel for
  plain-double Coulomb, team-scratch refactor of the stack-array kernels.
  **GPU runtime validation is BLOCKED on hardware** here (Intel-only dev box:
  no AMD GPU despite the ROCm modules, no CUDA, no SYCL, Kokkos OpenMP+Serial
  only) — do the refactor now (verifiable on the host backend), validate on a
  real GPU elsewhere. Also: minimax/Beylkin–Monzón t-grid and STO weight
  optimization via the arbitrary-precision core.

- **M-QUAD — optimal Losilla tuning** (elevated priority; the grid/local-hybrid
  and STO regimes both need it): use as small a truncation t_c as possible so
  the explicit t-quadrature covers only the expensive dense small-t region,
  pushing the cheap tail into analytic corrections. The delta term
  (pi*S/t_c^2, S the four-orbital overlap) is only the leading tail and gives
  1/t_c^4. Verified (prototype): adding the next moment,
  pi*L/(8 t_c^4) with L = int rho_ab lap(rho_cd), gives 1/t_c^6 (a 4x smaller
  t_c for the same accuracy). The tail is a series
  pi*S/t_c^2 + pi*L/(8 t_c^4) + ..., each order a cheap Laplacian-overlap
  moment off e_coeffs. The same series corrects the STO s-quadrature's
  tight-Gaussian (delta-like) tail, so this tooling serves STOs, local
  hybrids, and FEM/grid densities at once. Pair with minimax t-node placement
  for the explicit [0, t_c] range. Prototypes: prototype/sto_validation.py
  (STO = quadrature-contracted GTO, 5*zeta/8 to 7e-12) and the t_c study.

## Standing items

- No performance data exists; correctness only. Benchmark against
  libcint/libintX before optimizing.
- The local-hybrid **KS-potential** layer (with a user local mixing function
  g(r)) sits on M14 — turns the energy density into a working Fock
  contribution; note the several published gauge conventions differ.
- CI does not run the PySCF or MPI tests (not installed on runners); both are
  verified locally only.

## Reference: gen1int (one-electron response engine)

`/home/work/gen1int` (Bin Gao & Andreas Thorvaldsen, LGPL — GitLab
bingao/gen1int; learn architecture only, do not copy into our BSD-3-Clause
tree) is the
gold-standard **one-electron** integral/derivative engine for response theory
(used in Dalton/LSDalton/DIRAC). Same Hermite-Gaussian (McMurchie–Davidson)
substrate as us, but analytic Boys (`aux_boys_vec`), Fortran 90 + Python,
contracted Cartesian *and* spherical, London (LAO/GIAO) orbitals.

**Operator set (the target 1e property surface — mine this like libcint's
intor):** `INT_OVERLAP`, `INT_KIN_ENERGY`, `INT_POT_ENERGY` (nuclear),
`INT_ONE_HAMIL` (T+V), `INT_CART_MULTIPOLE`, `INT_SPHER_MULTIPOLE`,
`INT_ANGMOM` (angular momentum), `INT_PSO` (paramagnetic spin-orbit),
`INT_GAUSSIAN_POT` (finite-nucleus/effective potential), plus ECP. libintti
has S, T, V, Cartesian multipoles, angular momentum ⟨r×∇⟩, and their gradients
(M10/M16); missing: PSO, Gaussian/finite-nucleus potential, ECP, spherical
multipoles (c2s handles the transform).

**Key architectural lesson — arbitrary-order geometric derivatives via an
N-ary tree over atomic centers** (`gen1int_geom.F90`, `NaryTreeCreate`,
`order_geo`/`max_num_cent`). Rather than special-casing gradient then Hessian,
one recursion distributes the derivative order among the differentiated
centers (with N_alpha <= 2 typically). **Adopt this for M16:** build the
derivative layer as arbitrary-order from the start (gradient and Hessian are
orders 1 and 2), not ad hoc. gen1int also does magnetic-field and total
rotational-angular-momentum derivatives (LAO) — the 1e magnetic-response
integrals for NMR/magnetizability.

**Positioning:** gen1int is 1e-only; libintX is 2e-only (GPU). libintti's bet
is ONE t-quadrature MD engine for BOTH, scalar-templated (real/complex/quad),
so GIAO/complex/anisotropic/arbitrary-kernel come for free where the
analytic-Boys engines need new special functions. gen1int is 15 years mature
on 1e response — treat its operator list and N-ary-tree derivative
organization as the design target, its numbers as an oracle.
