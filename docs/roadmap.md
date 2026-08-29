# libintti roadmap

Status: M1–M9 delivered (quartets, batching, Cholesky, PSC/grid interactions,
GIAOs, quad precision, libcint façade — ERIs match PySCF to 2×10⁻¹⁴, an RHF
driven entirely by libintti matches PySCF to 1.4×10⁻¹³ Ha). The **RI +
property + derivative + local-hybrid program** (matrix-level API; see the
approved plan) is under way and mostly delivered:

- **M10 done** — 1e property matrices (overlap, kinetic, multipoles), `oneel.hpp`.
- **M11 done** — screened nuclear attraction + Coulomb-potential collocation, `nuclear.hpp`.
- **M12 done** — 2-/3-center Coulomb (ghost-shell), `ncenter.hpp`.
- **M13 done** — RI J/K + occupation-driven RI-K, `ri.hpp`.
- **M14 done** — local & locally range-separated exchange energy density, `localhybrid.hpp`.
- **M16 (in progress)** — derivative integrals, `deriv.hpp`: one-electron
  gradient set complete (overlap/kinetic/nuclear, vs PySCF `int1e_ip*` and
  finite difference). Remaining: two-electron/RI gradients, Hessians, ∂/∂B
  GIAO field derivatives (forces + frequencies = the black-box goal).

**Ground rules (binding for any implementer):** every deliverable has a hard
oracle (PySCF or analytic) *and* an independent second check (finite
difference or an internal identity) — the RI eigenvector-transpose bug was
invisible to self-consistency alone. Never loosen a tolerance to proceed;
report and stop. Matrix-level API only (no per-quartet public entry point).
Follow the codebase idiom (header-only templates, Kokkos for float/double,
serial fallback for complex/quad/class scalars, MPL-2.0 + SPDX).

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
