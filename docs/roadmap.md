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
  translationally invariant. RI-K (exchange) gradient done (same file): the
  double density contraction routes through the density-transformed 3-index
  H_{sn}^Q = sum_l D_sl (ln|Q) and its fit G = M^{-1} H, giving 3-centre and
  2-centre coefficient tensors that the same quartet_pos_grad helper contracts;
  validated vs finite difference of the ri_jk exchange energy. RI J and K
  gradients thus complete the RI Fock-derivative surface (skeleton; the caller
  adds CPHF). RI J and K Hessians done (`rigrad.hpp`): the envelope form
  d^2E/dxdy = (direct integral-Hessian terms with the gradient's coefficient
  tensors) + a response term in first-derivative residuals contracted through
  M^{-1} (r_x for J; the 3-index R_x with an orbital transpose for K). Two
  shared helpers -- quartet_pos_deriv_block (full first derivative) and
  quartet_pos_hess (two-position second derivative) on the ghost quartets --
  drive both. Validated vs finite difference of the RI gradients and symmetric.

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

- **M-3EL — three-electron integrals** (`threeel.hpp`): the t-quadrature
  applied twice. G_abcdef = <a(1)b(2)c(3)|r12^-1 r13^-1|d(1)e(2)f(3)> replaces
  both Coulomb operators by their Gaussian transform, giving a 2D (t,s)
  quadrature (Mehine, Losilla & Sundholm 2013 -- the same group's generalisation
  of the scheme libintti already uses; the scaled/Mobius grid and the delta tail
  carry over per dimension). At each (t,s) node the six-fold integral over the
  three electron coordinates is a product of Gaussians -- electron 1 (density
  P = a d) is shared, electron 2 (Q = b e) couples through t^2, electron 3
  (S = c f) through s^2 -- that factorises over the Cartesian axes into a
  3-variable (x1,x2,x3) Gaussian with quadratic form A (det A = (aP+LQ+LS)
  (aQ+t^2)(aS+s^2)) and a centre-offset source. The s-value is pi^{9/2}/
  det(A)^{3/2} exp(-Phi) with the FULL three-term exponent Phi = [aP LQ R_PQ^2
  + aP LS R_PS^2 + LQ LS R_QS^2]/(aP+LQ+LS); arbitrary angular momentum is a
  polynomial moment E[(x1-RP)^nP (x2-RQ)^nQ (x3-RS)^nS] of that Gaussian
  (Isserlis recurrence, `te_central_moments`), contracted with the Cartesian T
  (Gaussian-product) coefficients (`te_core`).

  *Correctness note.* An earlier version used the truncated exponent
  exp(-LQ R_PQ^2 - LS R_PS^2), which drops the aP/(aP+LQ+LS) scaling and the
  entire Q-S coupling term. It was wrong for every multi-centre geometry but
  passed its tests: the one-centre analytics have R^2 = 0, the "general s-type
  reference" had been generated with the same formula (circular), and the
  finite-difference l>0 checks compared the explicit integral to a difference of
  the same wrong s-integral (internally consistent). It was caught by adding a
  genuinely independent l>0 oracle. The engine was rebuilt on the correct
  three-variable Gaussian-moment form and is now pinned by independent
  references (below).

  Operators are node lists: Coulomb r^{-1} = (2/sqrt pi) int e^{-t^2 r^2} dt is
  the t-grid, a Gaussian geminal sum_k c_k e^{-g_k r^2} its fixed nodes, and
  f/r (`geminal_over_r_nodes`) the Coulomb grid with exponents shifted by g_k
  (g=0 recovers Coulomb). So the node-representable F12 family -- r^{-1}, f,
  f^2, f/r -- all go through one routine (`three_electron`). The r^2 moment
  (`three_electron_moment12`) inserts r12^2 = sum_dir (x1-x2)^2 into the same
  moment machinery, giving the linear operator r12 = r12^2 r12^{-1} (Coulomb
  nodes) and (grad_1 f12).(grad_1 f12) (Gaussian nodes); the cross moment
  (`three_electron_moment_cross`) inserts the vector dot product r12 . r13 =
  sum_dir (x1-x2)(x1-x3), i.e. (grad_1 f12).(grad_1 f13) with Gaussian nodes on
  both slots -- the F12 commutator / B-matrix ingredient. Matrix-level
  contraction (`three_electron_energy`): the fully-contracted 3-body energy
  E = sum_{abcdef} G_{abcdef} D_ad D_be D_cf (density in, scalar out; sextet
  internal, like J/K from quartets) -- the mean-field 3-body term
  transcorrelated / F12 methods build. The effective one-body reduction is done
  (`three_electron_fock`): F_pq = dE/dD_pq, the 3-body contribution to the Fock
  matrix (density in, matrix out) -- each sextet scatters into the three slots
  its pairs occupy. This is what a mean-field SCF consumes. Both contractions
  take a `kind` selector so they also drive the moment operators (r12^2, r12.r13)
  -- the F12 3-body correction energies and their Fock derivatives.

  Independent validation (no circular or self-consistency references): the
  one-centre G_aaaaaa = 4 zeta/3 and Gaussian-geminal analytics (closed forms,
  `references/sympy_three_electron.py`); a general multi-centre s-type Coulomb
  value from an erf-potential 3D reduction; an l>0 many-centre Gaussian-geminal
  value from a closed-form multivariate-normal moment (SymPy); a whole battery
  of l>0 / many-centre integrals, r12^2 moments AND centre derivatives
  (`references/te_reference.py` -> `tests/threeel_reference.hpp`, independent
  Gauss-Hermite quadrature anchored to scipy); and the 2-function 3-body energy
  vs the erf-reduction sextet sum. The Coulomb/f-over-r centre-derivative
  finite-difference tests remain as coarse cross-checks (grid-FD noise floor
  ~1e-6). Density screening of the sextet loop is done: `three_electron_energy`
  and `three_electron_fock` take a `screen` threshold (default 0 = exact) that
  prunes on the density-weighted pair overlap q_pq = |D_pq| o_pq (o_pq the
  pair-overlap magnitude), skipping a sextet when q_ad q_be q_cf < screen (max
  q)^3 -- the same pruned set for both, so the screened Fock stays the exact
  gradient of the screened energy. This folds the density and pair sparsity into
  the loop. The fully-folded exact form E = int rho_D V_D^2 (density rho_D, its
  Coulomb potential V_D) is done by RI (`three_electron_energy_ri`, `threeel_ri.hpp`):
  fit rho_D to an auxiliary basis (d = M^{-1} g, M=(P|Q) via `coulomb_2c`,
  g_P=(P|rho_D) via `coulomb_3c`), then E ~ sum_{PQR} d_P d_Q d_R T_{RPQ} with
  T_{RPQ} = int chi_R V_P V_Q the three-electron integral of three single
  auxiliary functions (ghost-partner trick, still Mehine-direct). O(naux^3) +
  O(nao^2 naux) vs O(nao^6), and exact when rho_D lies in span(aux) -- validated
  to numerical precision for s and for p x p -> d products, plus the approximate
  (incomplete-aux) case. The integrals are never RI'd; only the contraction is
  refolded. The RI-folded Fock is done too (`three_electron_fock_ri`): E3^RI is
  cubic in the fitted density d and d is linear in D, so F_{mu nu} = sum_Q (mu
  nu|Q) [M^+ G]_Q with G the auxiliary-space gradient (each aux integral T_{ijk}
  scattered into its three slots) -- O(naux^3), matching the energy, with
  sum F D = 3E. Validated against the direct raw Fock in the exact-aux cases (s
  and p) and by finite difference of E3^RI. The effective two-body reduction is
  done for the Coulomb (J) channel (`three_electron_effective_coulomb_ri`):
  contracting electron 3 with a density Dc gives an effective 2-electron operator
  Omega, and folding it into a J build with a density Db collapses to the
  one-body matrix <mu nu|V_Dc V_Db> = sum_{PQ} dc_P db_Q T_{mu nu,PQ} (both
  densities fitted), O(nao^2 naux^2) vs O(nao^6). Dc = Db = D is the electron-1
  slot <mu|V_D^2|nu> of the three-body Fock; a separate Db is the transcorrelated
  / response use. Validated against the direct raw contraction (s, s with Dc!=Db,
  and p x p -> d). The exchange (K) channel is done too
  (`three_electron_effective_exchange_ri`): K_{mu la} = sum_P dc_P sum_{nu si}
  Db_{nu si} W^P_{mu nu, la si}, W^P = int rho_{mu nu} rho_{la si} chi_P/r12/r13
  = three_electron_raw(mu,la,P,nu,si,ghost) -- only V_Dc fitted, Db carried
  through the density-matrix kernel, so O(nao^4 naux) and NOT symmetric in
  (mu,la) (only electron 1 is dressed). Validated against the direct raw exchange
  (s with Dc!=Db, and p). The three-electron / F12 matrix-level program is now
  complete: integrals (r^-1, f, f^2, f/r, arbitrary l, many-centre), moments
  (r12^2, r12.r13), and every contraction -- energy, Fock, RI-folded energy/Fock,
  and the effective two-body J and K -- all independently validated. The RI
  folding no longer needs an external auxiliary: `cholesky_product_aux` /
  `three_electron_energy_cd` generate it in-library from the two-step Cholesky
  decomposition of the ERI over the orbital pair space -- the CD selects the
  significant shell pairs (threshold tau), each materialised as product-shell
  Gaussians (l = 0..l_i+l_j at the product centre, exponent alpha_i+alpha_j) that
  span the pair block exactly, so the set spans rho_D to the CD threshold with
  O(rank) auxiliaries. Validated exact (tight tau) vs the direct sum, s and p.
  The _cd path is precision-generic end to end: it drives the precision-generic
  `pivoted_cholesky` (batched/serial by scalar) and a Jacobi metric solve
  (`detail::jacobi_eigh`) in place of LAPACK, so `three_electron_energy_cd` runs
  at long double and __float128 -- validated against the direct sum at both.

- **Precision-generic Cholesky** (`cholesky.hpp`): the one-step pivoted Cholesky
  (ab|cd) ~= sum_J L_ab^J L_cd^J is LAPACK-free (pivoted recurrence + sqrt), so
  `pivoted_cholesky` runs at float/double/long double (the batched-integral
  scalars). two_step_cholesky's static_assert is relaxed to any such scalar and
  its step 2 (dense S^{-1/2}, host LAPACK) is if-constexpr-guarded to
  float/double, throwing for long double (which uses the one_step path).
  Validated: long-double pivoted CD reconstructs the ERI to threshold. For
  __float128 / MPFR (which the Kokkos batch path rejects), a serial pivoted
  Cholesky drives the single-quartet eri_quartet() host path directly (no
  PairTable, no LAPACK); `pivoted_cholesky` over a shell-pair list dispatches to
  the batched path for kokkos scalars and the serial path otherwise -- one entry,
  all precisions. Validated: the serial path matches the batched at double, and
  reconstructs the ERI far below the double floor at __float128. The consumer is
  precision-generic too: `cholesky_jk`'s only BLAS use (the K-build matmuls) goes
  through `detail::matmul_nn`, which is BLAS for float/double and a triple loop
  otherwise, and a ShellBasis `pivoted_cholesky` sets the pair bookkeeping for
  any precision. So CD -> J/K runs end-to-end at long double and __float128
  (validated against the exact four-index J/K, below the double floor). The GPU
  path stays Kokkos (only float/double/long double are device scalars); no
  mplapack/Eigen -- extended precision is CPU-only by nature (no __float128 GPU).

- **Adaptive t-grid range** (`tgrid.hpp`): the default Mobius grid (n=64, s=2)
  is calibrated for exponents [1e-2, 1e6]; real basis sets exceed this (heavy
  cores ~1e7-1e9, doubly-diffuse/Rydberg ~1e-3-1e-4). `mobius_spec_for_range`
  (and `adaptive_mobius_spec(basis)`) sizes one shared grid from the basis's
  exponent span, anchored to the proven default and growing the node count as
  (decades/10)^2.5 (a single Mobius map's tails thin out superlinearly with the
  span), with s tracking the geometric-mean exponent. Opt-in; the default spec
  is unchanged. Validated vs analytic (ss|ss) across [1e-5, 1e9] (14 decades) to
  <1e-7 where the default is off by ~4e-3.

  The Beylkin-Monzon / sinc mapping is done (`TMapping::ExpSum`,
  `exp_sum_spec_for_range`): substituting t = e^s in the Gaussian resolution and
  applying the trapezoidal rule gives log-spaced nodes t_k = e^{s_k} with weight
  (2/sqrt pi) h t_k, exponentially convergent. Its node count grows ~log(range),
  not the Mobius map's (decades/10)^2.5, so it is more efficient at wide spans:
  over [1e-5, 1e9] (14 decades) it matches accuracy (6.6e-9) at 161 nodes vs the
  Mobius 208, and the gap widens with the range. The small-t (large-r) side has a
  slow e^s tail needing a wide low-t margin (the large-t side cuts off
  super-exponentially). Validated vs analytic (ss|ss). The range-separated
  extension is done: ExpSum also covers the Yukawa (screened Coulomb) kernel,
  whose e^{-kappa^2/4t^2} factor kills the slow small-t tail, so both ends decay
  and the sinc rule is cleanly exponential (validated: Yukawa kernel reproduced
  to 1e-9). erf/erfc have a hard boundary at omega where the integrand is nonzero
  -- the trapezoidal rule is only O(h^2) there -- so ExpSum refuses them; Mobius
  (Gauss-Legendre, spectral on the finite/shifted range) is the tool for those.

- **M15 — NAO support: OUT OF SCOPE (dropped).** libintti does everything
  natively in GTOs; the STO path earns its place because it is the *integral
  transform* (Gaussian resolution of e^{-zeta r}) plus the delta-tail
  correction, which is more exact than a plain Gaussian fit of a Slater. NAO
  integrals, by contrast, would be a fitting approximation (NAO products onto a
  GTO aux, or NAOs onto GTO expansions) with residuals that are hard to control
  for cusped / strictly-confined NAOs -- an approximation the library declines to
  bless as a first-class feature. Anyone wanting it can build it on top of the
  GTO/STO machinery themselves. So there is no `nao.hpp`.

- **M-STO — Slater-type orbitals** (minimal STOs done, `sto.hpp`): a minimal
  STO (n = l+1), r^l Y_lm e^{-ζr}, is a contracted Cartesian GTO shell of
  angular momentum l with the s-expansion exponents/coefficients, so every
  existing matrix builder works once the primitive shells are contracted.
  `sto_gaussians` builds the log-Gauss-Legendre s-expansion, `expand_sto`
  emits the primitive ShellBasis + contraction map, `contract_to_sto` folds a
  primitive AO matrix to the STO basis. Validated vs analytic Slater results:
  the s-expansion reproduces e^{-ζr} to 1e-9, the 1s/2p self-overlaps hit
  π/ζ³ and π/ζ⁵ to 1e-8, and the 1s self-repulsion hits 5ζ/8 to 1e-6.
  Higher n done: r^m e^{-ζr} = (-d/dζ)^m e^{-ζr} is the same Gaussian
  contraction with ζ-differentiated coefficients (a Hermite-like P_m
  polynomial), so any n>=l+1 is a contracted GTO shell; validated to 1e-6 vs
  the analytic s-STO self-overlap 4π(2n)!/(2ζ)^{2n+1} for n=1,2,3. Delta-tail
  acceleration done: truncating the s-grid at s_c leaves a delta-like tail
  whose weight int_{s_c}^inf g(s,ζ)(π/s)^{3/2} ds = (8π/ζ³)[1-(1+u)e^{-u}],
  u=ζ²/4s_c, is added as V(centre) times that weight (the Losilla tail, now on
  the s-quadrature) -- recovering the exact int e^{-ζr}d³r from a heavily
  truncated grid. The correction is wired into an STO overlap builder
  (`sto_overlap_delta`, 1s): each orbital's s-integral is split at s_c, the
  low-s block is an ordinary contracted-GTO overlap, and the delta tail adds
  W_partner times the low-s value at the partner centre (tail-tail vanishes for
  distinct centres; the diagonal is the analytic self-overlap) -- reproducing
  the full-grid overlap to 1e-5 from a coarse 32-node truncated grid. Extended
  to l>0: the angular polynomial vanishes at the centre, so the tail acts as
  *derivatives* of delta -- an off-centre partner contributes the component
  weight (a lower-incomplete-gamma, C (pi zeta/2)(4/zeta^2)^{N+2} gamma(N+2,u))
  times the parity-order (a_d mod 2) derivatives of the other orbital's low-s
  part at the partner centre; the diagonal block comes from a dense
  single-centre grid. Mixed s/p reproduces the full grid to 1e-4.
  Two-electron delta-tail done for the two-centre density Coulomb repulsion
  (`sto_coulomb_2c_delta`): rho_A's s-grid is truncated at t_c and its tight
  tail charge Q_tail contributes Q_tail * V_B(A), the tail sitting at A and
  sampling rho_B's smooth analytic Slater potential there. The full builder
  (`sto_coulomb_2c`) matches the analytic Roothaan two-centre 1s Coulomb to
  1e-6; the delta-tail recovers it to ~4e-4 from a coarse 20-node truncated
  rho_A grid (accuracy controllable via t_c). General STO J/K builds done
  (`sto_jk_build`): every STO AO is a contracted GTO, so the STO ERI is a
  contraction of primitive ERIs -- push the density to the primitives
  (C^T D C), run the ordinary coulomb_build/exchange_build, contract back
  (C J C^T); expand_density_to_prim is the reverse of contract_to_sto.
  Validated: the single-1s self-ERI J=K=5 pi^2/8 zeta^5 and the two-centre J
  vs sto_coulomb_2c (hence the analytic Roothaan value). The practical cost
  control for the STO J/K build is Schwarz screening: the tight tail primitives
  are spatially local, so tau>0 prunes their negligible inter-centre integrals
  exactly-to-tolerance (verified: screened == unscreened). Folding the
  *delta-tail* into the matrix build does NOT compose from the clean 2-centre
  correction -- the off-diagonal two-centre bra products and the several
  tail-combination terms (bra tail x density tail, on-centre tail-tail) leave
  leading-order corrections ~1-3% off on a coarse grid with a systematic
  overshoot. Root cause found (`prototype/sto_delta_tail_j.py`): the delta-tail
  must be applied at the PAIR-DENSITY level, not by truncating orbital
  primitives -- (phi^low)^2 is a different truncation than truncating the pair
  density e^{-2 zeta r}, and drops the "one orbital tight" pairs the point
  charge cannot repair. At the density level it works even on-centre (R=0): the
  1s self-repulsion 5 zeta/8 is reproduced to 1.4e-4 at t_c=20 and 6e-3 at
  t_c=8. A production delta-tail J is therefore a re-architecture around pair
  densities (single-centre Slater for same-atom pairs, point-charge tails for
  two-centre pairs) plus the closed-form tail-tail self-energy; open. An STO is the exact
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
  grid mappings. The route is the integral transform (exact in the node limit)
  with the delta-tail correction -- NOT a plain Gaussian fit of the Slater, which
  is what makes it more exact and is the reason STOs are in scope where NAOs are
  not (see M15). Plan: Python-prototype the nested s×t + delta-tail against
  analytic Slater integrals (e.g. the 1s self-repulsion 5ζ/8) first, pin the
  s-grid and node counts, then implement a matrix-level STO basis on the existing
  engine. User is interested in STOs *widely* -- a first-class basis type
  alongside GTOs, not a niche add-on.

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
