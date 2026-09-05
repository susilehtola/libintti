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

- **M-TC — transcorrelated (and F12) integrals** (builds on M-3EL). The TC method
  similarity-transforms H by a Jastrow J = sum_{i<j} u(r_i,r_j):
  H_TC = e^{-J} H e^{J} = H - sum_{i<j} K(r_i,r_j) - sum_{i<j<k} L(r_i,r_j,r_k),
    K(r_i,r_j) = 1/2[nabla_i^2 u + nabla_j^2 u + (nabla_i u)^2 + (nabla_j u)^2]
                 + (nabla_i u . nabla_i + nabla_j u . nabla_j),
    L(r_i,r_j,r_k) = nabla_i u_ij . nabla_i u_ik + nabla_j u_ji . nabla_j u_jk
                     + nabla_k u_ki . nabla_k u_kj.
  (Haupt PhD thesis 2024, Eq 2.29-2.31; F12 uses the SAME family additively --
  r^-1, f, f^2, f/r, (grad f)^2, and the 3-electron B-matrix term.) Almost the
  whole integral list is ALREADY in threeel.hpp as operator node lists:
  - V = <pq|r12^-1|rs>: Coulomb (coulomb_nodes). done.
  - K's multiplicative pieces: 1/2 nabla^2 u and 1/2 (nabla u)^2 are geminal /
    moment node lists -- (nabla f)^2 is exactly three_electron_moment12; nabla^2 u
    of a Gaussian geminal is another geminal (4 g^2 r^2 - 6 g) node list. done.
  - L (three-body): each cross-gradient term (grad_i u_ij).(grad_i u_ik) is the
    cross moment three_electron_moment (moment=2); L is the sum of its three
    electron-pair permutations. done and independently validated.
  - The ONE genuinely new integral: K's NON-HERMITIAN term nabla_i u_ij . nabla_i.
    Integrating by parts moves the geminal gradient (x1-x2) off the operator onto
    the electron-1 orbitals, collapsing the (x1-x2)-weighted "moment" geminal to
    PLAIN geminal integrals with derivative-shifted orbitals -- no moment code:
      <pq| grad_1 u.grad_1 |rs>
        = - sum_k c_k sum_d [ G_k(d_d p, q; d_d r, s) + G_k(p, q; d_d^2 r, s) ]
    where G_k(..) = <..|e^{-g_k r12^2}|..> is the plain Gaussian-geminal 2e
    integral and d_d/d_d^2 are orbital derivatives (deriv.hpp center-shifts).
    Both ingredients now exist as first-class pieces: the geminal 2e integral is a
    `gaussian_geminal(g,c)` TGrid (tgrid.hpp) fed to eri_quartet / coulomb_build /
    exchange_build -- VALIDATED to 1e-16 vs the closed-form ss geminal integral
    (test_tgrid GaussianGeminalMatchesAnalytic). The same grid gives the geminal
    J/K matrices, so K's Hermitian pieces (1/2 nabla^2 u, 1/2 (nabla u)^2) are just
    geminal J/K builds. Remaining: assemble the non-Hermitian term from
    derivative-shifted shells (erigrad-style) over the geminal grid, matrix-level.
  So the milestone is: (1) the non-Hermitian nabla_u.nabla 2-body operator;
  (2) a matrix-level TC-Hamiltonian assembler producing the effective
  (non-Hermitian) 1e/2e integrals + the 3-body L contributions a TC-FCIQMC /
  TC-CC host consumes (density/orbitals in, integrals out -- never per-quartet).
  Jastrow representation: u as a Gaussian-geminal sum (standard STG-nG), or --
  better -- Slater-type u via the integral transform + delta tail we use for STOs
  (M-STO), which is MORE exact than a plain Gaussian fit (exact in the node limit
  plus the analytic tail) and handles the coalescence cusp analytically.
  Why our approach wins over the reference (Haupt/PyTCHInt, tchint): they evaluate
  EVERY TC matrix element -- the 3-electron L and the K terms -- by
  Treutler-Ahlrichs atom-centred numerical grids (Becke multicenter scheme), with
  real grid error they extrapolate away (lgrid 1-5 -> 1/n_grid->0). Ours is
  analytic: the Gaussian transform reduces the two operators to a smooth 2D (t,s)
  quadrature and does the 9D electron integral analytically (Gaussian moments /
  Isserlis), exponentially convergent (Mobius grid + higher-order delta tail) at
  arbitrary precision -- no spatial grid, no cusp-convergence problem, controlled
  error. The multiplicative TC ansatz caps at 3-electron integrals (vs F12's
  additive higher-body), which is exactly what the t-quadrature does natively.
  Oracle: the 3-electron cross moment is already validated independently; the new
  non-Hermitian term vs a closed-form / independent-quadrature reference and vs
  PyTCHInt's grid values (grid-converged); a small TC-SCF or xTC energy vs a
  published TC reference as the capstone.

- **M-MP -- multipole (far-field) integral evaluation** (`include/intti/multipole.hpp`).
  For well-separated bra/ket product-charge distributions (|P-Q| large vs their
  extents) the quartet collapses to a bilinear multipole contraction
      (ab|cd) ~= sum_{lm,l'm'} q^{ab}_{lm} T^{lm}_{l'm'}(R_PQ) q^{cd}_{l'm'},
  the FMM far-field. The moments q^{ab}_{lm} are exactly the solid-harmonic
  moment matrices moment.hpp already builds (M10), so the distant-pair ERI /
  nuclear term is a cheap O(L^4) contraction of existing moment vectors with an
  interaction tensor T(R) -- no primitive quartet. Highest-value target: the
  local-hybrid grid x pair nuclear-attraction over huge grid x pair sets
  (M11/M14), where distant grid points get the multipole shortcut and near ones
  the exact t-quadrature (Schwarz + distance decides near/far, screening already
  present). Design: (1) a well-separatedness criterion; (2) pair multipole
  moments (have); (3) the interaction-tensor contraction, matrix-level.
  Note -- unifies with M-HK: the SAME machinery is the Helmholtz far-field with
  T(R) swapped from the Coulomb tensor 1/R^{l+l'+1} to the modified spherical
  Bessel tensor I-hat_{l+1/2}, K-hat_{l+1/2} (Park 2017 Eq 6-7, 13-16). Do M-MP
  first: it stands alone and is the on-ramp to M-HK. Oracle: exact quartet in the
  far regime; a full FMM-accelerated J / nuclear build vs the dense reference.

- **M-HK -- Helmholtz-kernel (Green's-function) SCF route** (`include/intti/helmholtz.hpp`).
  An alternative to diagonalizing the Fock matrix: the bound-state orbital update
  is a Green's-function fixed point (Park 2017, JCTC 13, 654; Eq 2-5),
      psi_i^{(n+1)} = -2 (-nabla^2 + kappa_i^2)^{-1} V psi_i^{(n)}
                    = -2 int e^{-kappa_i|r-r'|}/(4 pi|r-r'|) V(r') psi_i^{(n)}(r') dr',
      kappa_i = sqrt(-2 eps_i).
  The Green's function is the bound-state Helmholtz kernel = a Yukawa (screened
  Coulomb) kernel, WHICH WE ALREADY CARRY: the ExpSum t-grid covers Yukawa
  (its e^{-kappa^2/4t^2} factor kills the slow small-t tail; validated to 1e-9,
  see the Adaptive t-grid note). The apply-then-project integrals
  <chi_mu|G_kappa V|chi_nu> are Yukawa-kernel (x) Coulomb (nuclear / J) products
  -- multi-t Gaussian integrals, the same machinery as the 3-electron / geminal
  integrals (threeel.hpp); the core primitive is the Yukawa potential of a GTO
  product (analytic).
  THE PER-ORBITAL SCALING CONCERN AND ITS RESOLUTION. Each occupied orbital has
  its own kappa_i, so naively the kernel is applied N_occ times -- a loop that
  could kill the method (and is only rescued in fully numerical codes by the
  finite kernel range 1/kappa ~ 1 bohr for valence plus localized orbitals ->
  each apply O(1) local, loop O(N)). The t-quadrature dodges it structurally:
  kappa enters ONLY as a scalar Gaussian weight e^{-kappa^2/4t^2} on the shared
  t-nodes, and V is orbital-independent, so
      M(kappa)_{mu nu} = <chi_mu|G_kappa V|chi_nu>
                       = sum_t w_t e^{-kappa^2/4t^2} K_{mu nu}(t)
  is built from ONE t-resolved tensor K_{mu nu}(t) common to every orbital;
  each M(kappa_i) is a cheap N_t-term (~20-40) re-weighting. So the cost is one
  Fock-like build + N_occ cheap re-weight-and-matvec updates -- comparable to a
  normal Fock build MINUS the diagonalization, not N_occ x anything expensive.
  This is BETTER than the grid convolution: those put the source inside the
  convolution (g = 2 V psi_i) and so redo the expensive integral per orbital,
  rescued only by locality; our matrix form pulls the source out (linearity:
  G_kappa V is a shared operator tensor), amortizing across orbitals -- and we
  STILL inherit the locality, because e^{-kappa^2/4t^2} truncating the small-t
  nodes is exactly a t-grid truncation (the LRSH primitive, M14). We get both
  escapes; the grid gets one. Residual cost: N_occ matvecs M(kappa_i)c_i (O(N^3)
  dense, O(N) with locality) -- the cost of forming the density anyway.
  Honest caveats on value. In a SMALL GTO basis the eigensolve is cheap, so the
  win is not avoiding diagonalization; the genuine advantages are (a) G_kappa
  inverts -nabla^2 exactly and we only project the RESULT -- the kinetic operator
  is never projected, so better-conditioned and potentially better cusp-region
  energies than Galerkin in the SAME basis; (b) all screened convolutions /
  contractions, no dense eigensolve -- a fit for the GPU-saturating matrix-level
  design; (c) FMM far-field (M-MP) + kappa-screening -> linear scaling. Payoff is
  in large / mixed GTO-STO / fully-numerical-adjacent bases, not def2-SVP water.
  Plan: (1) the Yukawa-potential-of-GTO primitive vs a closed form -- DONE
  (helmholtz.hpp): yukawa_grid builds the ExpSum Yukawa t-grid and, because the
  screened kernel differs from Coulomb only by the node weight e^{-kappa^2/4t^2},
  EVERY existing builder computes the Yukawa analogue unchanged when fed it
  (attraction_accumulate/nuclear_matrix -> Yukawa attraction, eri_quartet/
  coulomb_build/exchange_build -> Yukawa ERI/J/K). yukawa_attraction_matrix
  validated to ~1e-6 vs an independent real-space screened-Poisson radial oracle
  (test_helmholtz); (2) the M(kappa) two-kernel build -- DONE
  (helmholtz_nuclear_matrix): M_{mu nu}=<mu|G_kappa V|nu> as a Yukawa (x) Coulomb
  double quadrature via eri_quartet + the ghost trick (Yukawa kernel = 4pi
  G_kappa; the nuclear 1/r' unfolds on a Coulomb grid as a Gaussian at R_C).
  VALIDATED by the Green's-function fixed point M(kappa0)c0 = -1/2 S c0 at the
  H_core ground state -- residual basis-limited (1.9e-4 at 6 fns, 3.8e-5 at 12,
  tracking eps0->-0.5; the integral form uses the exact kinetic while Galerkin
  projects it). This de-risks the route. Remaining: fold the t-resolved tensor so
  one build serves all kappa_i (the per-orbital amortisation); (3) a Helmholtz-SCF
  proof-of-concept on H / He / H2 confirming the fixed point converges to the
  in-basis Galerkin/diagonalization energy (add J/K to V for many-electron), then
  measuring whether kinetic-exactness buys accuracy per basis function.
  (3) PROOF-OF-CONCEPT DONE for the H atom (references/hk_scf_poc.cpp): the pure
  integral-form eps* -- the self-consistent point where A(eps)=-2 S^{-1} M(kappa)
  has dominant eigenvalue 1, found by power iteration + secant, extracting the
  energy WITHOUT the projected kinetic -- converges to machine precision
  (lambda-1 ~ 1e-13) and agrees with Galerkin eps0 in the basis limit
  (eps*-eps0: 3.2e-3 at 4 fns -> 2.8e-5 at 8 fns; both -> -0.5). HONEST FINDING:
  kinetic-exactness does NOT give better accuracy per basis function -- eps* is
  marginally WORSE than the variational Galerkin eps0 for H. So the route's value
  is NOT per-basis accuracy but the structural properties (no diagonalisation,
  screened linear-scaling convolutions, GPU-friendly all-contraction SCF). Still
  remaining -- the many-electron case (fold J/K into V) is a THREE-kernel build:
  <mu|G_kappa J|nu> = sum_{rs} D_{rs} int^3 chi_mu(r) chi_nu(r') G_kappa(r-r')
  chi_rho(r'') chi_sigma(r'')/|r'-r''| -- Yukawa (r,r') AND Coulomb (r',r''), a
  threeel-topology integral (electron 2 central) with the density folded on
  (rho sigma). Buildable like M(kappa): unfold 1/|r'-r''| on a Coulomb grid so
  the (rho sigma) pair becomes a t-smeared Gaussian at P_{rs} (exponent
  p t^2/(p+t^2)), then eri_quartet(pair(mu,ghost), pair(nu,G_smear), yukawa_grid)
  summed over (rho sigma) pairs and t-nodes (fold the smear prefactor
  K_{rs}(pi/(p+t^2))^{3/2}; general-L smears carry a polynomial). The G_kappa J
  build (s-case) is VALIDATED (references/hk_j_poc.cpp): for a fixed density D,
  the eigenpair of F=T+V_nuc+J[D] satisfies M_eff(kappa0)c0 = -1/2 S c0 with
  M_eff = M_nuc + M_J[D] -- residual drops from 0.72 (M_nuc only, J missing) to
  3.2e-4 (6 fns) / 5.2e-6 (9 fns), basis-limited. EXCHANGE M_K also VALIDATED
  (references/hk_jk_poc.cpp): despite first appearances it is NOT a 4-point
  object -- (K chi_nu)(r) = sum_{rs} D_rs chi_rho(r) int chi_sigma chi_nu(r')/
  |r-r'|, so <mu|G_kappa K|nu> = sum_{rs} D_rs int^3 chi_mu(r'') G_kappa(r''-r)
  chi_rho(r) chi_sigma chi_nu(r')/|r-r'| is the SAME three-kernel threeel-topology
  as M_J, just contracted differently (rho on the Yukawa side, the (sigma,nu)
  pair smeared). The RHF-like fixed point M_eff(kappa0)c0 = -1/2 S c0 with
  M_eff = M_nuc + M_J - 1/2 M_K holds to the basis limit (4.1e-4 at 6 fns,
  2.4e-5 at 9). So ALL THREE integral pieces of the Helmholtz apply work; the
  route is conceptually de-risked end to end. Remaining is engineering, and it
  needs NO new integral code: the three-kernel (mu nu | Y_kappa (x) C | rho sigma)
  is exactly threeel.hpp's three_electron_kind with op12 = yukawa nodes, op13 =
  coulomb nodes, and the central electron on nu:
    (mu nu|Y_k (x) C|rho sigma) = three_electron_kind(nu, mu, rho, ghost, ghost,
                                    sigma, yukawa_nodes, coulomb_nodes),
  so general-L M_J = sum_{rho sigma} D_{rho sigma} (mu nu|...|rho sigma) and M_K
  the analogous contraction (rho on the Yukawa/central side). No smear-as-shell
  needed -- threeel handles general-L Gaussians natively (the s-case PoCs use the
  cheaper eri_quartet smear). DONE (helmholtz_jk_matrices): general-L M_J and M_K
  are library functions -- the density contraction over three_electron_raw_nodes
  with yukawa_grid nodes as op12 (via coulomb_nodes, since OpNode just carries
  {w,t}) and a per-AO CartGauss helper. Validated by the He effective-potential
  fixed point M_eff = M_nuc + M_J - 1/2 M_K (test_helmholtz ManyElectronJKFixedPoint)
  and vs the s-case smear PoCs to 1e-15. So the ENTIRE Helmholtz-apply integral
  machinery (yukawa_grid, helmholtz_nuclear_matrix, helmholtz_jk_matrices) is in
  the library. Remaining: the many-electron SCF DRIVER (application-level; the
  fixed-point checks already prove convergence); and efficiency -- the current
  J/K is O(nao^4) three-electron evals, so the t-resolved-tensor amortisation
  across the occupied kappa_i (only the Yukawa node weight e^{-kappa^2/4s^2}
  depends on kappa, so one (s,t)-resolved tensor serves every orbital) plus
  density-folding (as tc_gradu_grad_build did) are the perf steps.
  Oracle: closed-form Yukawa
  integrals; the converged HK energy vs a standard diagonalizing SCF in the same
  basis (must agree in-basis), and vs the basis-set-limit reference.

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
  -- the trapezoidal rule is only O(h^2) there -- so ExpSum refuses them.

  The double-exponential (tanh-sinh) mapping is done for erf/erfc
  (`TMapping::DoubleExp`, `de_spec_for_range`), the DFT-relevant range-separated
  kernels. tanh-sinh sends the omega boundary to a doubly-exponentially clustered
  endpoint, restoring spectral convergence across it. erf = int_0^omega integrates
  the finite interval directly (spectral: ~80 nodes to machine precision). erfc =
  int_omega^inf substitutes t = omega e^v to map the half-line onto v in [0,
  v_max]; in the log variable the Gaussian's transition at t ~ 1/r has an
  r-independent width, so one grid resolves every r (a linear map over [omega,
  t_max], or exp-sinh, spreads the far nodes too fast and under-resolves small r).
  erfc is the limiting case (its wide v-interval converges slower than erf's narrow
  one): de_spec_for_range sizes de_h=0.028 and de_tmax=12 sqrt(alpha_max) for the
  erfc grid, giving < 1e-9 over ~8 exponent decades at ~285 nodes (erf reuses the
  spec and is over-resolved to machine precision; an erf-only caller can pass a
  coarser de_h). Validated vs std::erf/erfc across r in [1e-3, 1e1].

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
  Higher-order (radial) delta tail done, any l (`sto_overlap_delta` radial_order,
  `detail::sto_tail_weight_comp_p` + the generalised `g1d`): the tight tail
  Gaussians sample not just phi_partner(centre) but its derivatives, so the tail
  composes the l>0 angular parity-derivatives with the radial Laplacians --
  sum_{1<=|p|<=K} W^{(p)} d^{dB+2p} phi_partner(centre), the partner
  differentiated to order dB_d+2p_d per axis (dB = a%2 the leading parity),
  W^{(p)} = [prod_d (2(N_d+p_d)-1)!!/(dB_d+2p_d)!] 8 pi 2^{N+k} gamma(N+k+2,u_c)/
  zeta^{3+2(N+k)}, N_d=(a_d+a_d%2)/2, k=|p|, u_c=zeta^2/(4 s_c). p=0 recovers the
  leading sto_tail_weight_comp; a=0 gives the pure 1s W_k (nabla^2)^k after the
  multinomial over p (references/sympy_slater.py verifies W_1..W_3 of the radial
  slice). Each order removes a 1/s_c power, so a much harder truncation reaches
  the same accuracy -- at s_c=6 the order-2 tail is ~100x tighter than the leading
  term for both 1s (1.5e-5 -> 1.3e-7) and mixed s/p (1.1e-4 -> 5.6e-7); s_c=6
  buys what s_c~20 needs at order 0. g1d generalised from parity {0,1} to any
  derivative order via the coefficient map Poly -> Poly' - 2 s u Poly.
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
  t_c=8. The pair-density-level building block is now done for 1s
  (`sto_coulomb_2c_delta_sym`). Same centre (R=0) is EXACT via the closed-form
  1-centre Coulomb of two Slater densities, (rho_A|rho_B) = zA - zA^3/s^2 -
  zA^3 zB/s^3, s = zA+zB (`detail::sto_coulomb_1c`, references/sympy_slater.py;
  reduces to 5 zeta/8), so the cusp-cusp case is trivial and the delta tail is
  used only for R>0, where every term is smooth. There BOTH densities are
  truncated at t_c: (low_A|low_B) + (tail_A|rho_B) + (rho_A|tail_B) - Q_A Q_B/R,
  the cross terms carried to any order via the Gaussian-smoothing series
  (tail_A|rho_B) = sum_k W_{A,k} (nabla^2)^k V_B(A) -- Poisson makes them closed
  form ((nabla^2)^k V_B = -4 pi (nabla^2)^{k-1} rho_B, and (nabla^2)^j e^{-kr} =
  (k^{2j} - 2j k^{2j-1}/r) e^{-kr}). Validated: R=0 exact, and two-centre order 2
  is ~1e4x tighter than order 0 (t_c=6: 1.8e-4 -> 1.4e-8), each order removing a
  1/t_c^2 factor. The l>0 pair-density tail's key enabler is done -- arbitrary
  Cartesian derivatives of the Slater potential (`detail::radial_cart_deriv` +
  `slater_pot_radial_derivs`), validated to 1e-12 vs a SymPy oracle -- so an l>0
  pair density's multipole tail interacting with a 1s density is
  sum_p W_ab^{(p)} d^{dP+2p} V_cd(A), reusing sto_tail_weight_comp_p. Remaining for
  a production matrix J: assemble that cross term (p-type-density Coulomb
  reference via the ghost/centre-differentiated ss Coulomb) and the l>0 x l>0 case
  (non-spherical V_ab), then the matrix assembly ((ab|cd) = pair-pair Coulomb
  scattered into J, with Schwarz screening still the exact cost control
  meanwhile). An STO is the exact
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

- **M-SHARK — BLAS-3 factorization, LKC architecture, digestion**
  (techniques from Neese, *J. Comput. Chem.* 2023, 44, 381, "The SHARK integral
  generation and digestion system"; ideas/architecture only, ORCA/SHARK is not
  open — do not copy code). Five items. Assessment after measuring: **#1 and #3
  DO NOT apply** to our t-quadrature J-engine (both optimize SHARK's
  explicit-integral MD architecture; ours is already integral-free -- see the
  per-item notes). #1 is at best a future GPU item gated on batched-GEMM infra.
  **#4** (Helgaker-Taylor derivatives) genuinely APPLIES but is a MODERATE win on
  the derivative surface (not the SCF hot path) for a substantial implementation
  (~1.8x on 2e gradients, more on Hessians -- see the #4 note). So the clear
  realizable win is **#2** (LKC -- maintainability, no perf risk); **#5** (general
  contraction) is needed for def2/ANO basis sets; **#4** is worth it only if
  frequency/geometry-opt throughput becomes a priority. Recommended order: 2, 5,
  then 4 if warranted. Meta-conclusion: SHARK's performance techniques target its
  explicit-integral MD architecture; our t-quadrature J-engine is already
  integral-free and efficient, so the transferable value is chiefly architectural
  (#2) rather than the headline speedups.

  1. **BLAS-3 (GEMM) factorization of the contraction** (SHARK Eq 21, 25-26) --
     the headline performance item, = the M18 "batched-GEMM." Recast the
     per-quartet Hermite contraction as matrix multiplications. It fits the
     universal t-grid unusually well: the shared t-nodes are a natural batch/
     matrix dimension, and the per-t-node Hermite integrals (`hermite_b`) play
     SHARK's R-matrix role, so phase G becomes G_d = fh_d^T B_d (per direction,
     ncomb x nf times nf x nt -> ncomb x nt) and the assembly is a t-weighted
     contraction of G_x G_y G_z. Replaces the hand-written loops in
     quartet.hpp/batch.hpp; drives peak vendor BLAS on any hardware.
     Validated (`prototype/gemm_factorization.cpp`): the GEMM path matches
     eri_quartet to 2.3e-16 across s/p/d/mixed. Benchmarked
     (`prototype/gemm_bench.cpp`, per-quartet single GEMMs, nt=64, this CPU box):
     GEMM LOSES at low L (ss 0.58x, ps 0.80x -- dgemm overhead on tiny matrices)
     and WINS 1.4-2.9x for l_tot >= 2 (ncomb >= 4), rising to 2.9x for ff --
     BUT that single-quartet micro-benchmark is misleading. The BATCH benchmark
     (`prototype/gemm_bench.cpp` batchbench, 1024 quartets, 6 threads) is the
     decisive one: the CURRENT phase-G threads over quartets with OpenMP, and a
     serial loop of per-quartet GEMMs is 4-8x SLOWER (hand/gemm 0.12-0.24x). The
     root cause is structural: in the t-quadrature theta_i = t^2 p q/D is
     QUARTET-SPECIFIC (via p,q), so B_d differs per quartet and the quartets do
     NOT merge into one large GEMM the way SHARK's shared-primitive structure
     does. A naive per-quartet GEMM drop-in therefore REGRESSES 4-8x on CPU by
     forfeiting the thread parallelism the hand loops already exploit.
     Conclusion / re-scope: the GEMM win requires a BATCHED GEMM (N independent
     small GEMMs run across all threads/SMs) -- KokkosKernels batched GEMM, MKL
     dgemm_batch, or cuBLAS/rocBLAS batched -- NOT a reference-BLAS drop-in. It is
     primarily a GPU lever (each small GEMM a device kernel; blocked on hardware
     here) and a win on MKL CPUs; with reference BLAS + OpenMP the current
     hand-loops are near-optimal and must not be replaced. So #1 is gated on
     batched-GEMM infrastructure; the hand loops stay the CPU default. This does
     not change #2/#3 (architecture and digestion), which are the near-term wins.
  2. **Loop/Kernel/Consumer (LKC) architecture** (SHARK Sec 3.10) -- one
     IntegralLoop, pluggable Kernels (operators) x Consumers (tasks: J, K,
     gradient, transformation), to consolidate the builder sprawl
     (jbuild/kbuild/coulomb_build/exchange_build/nuclear/ncenter/threeel). Our
     "operators = node lists" already unifies the KERNEL side more elegantly
     than SHARK's per-kernel functions, so the work is mainly the CONSUMER
     (digestion) abstraction. Do this first: it is the seam #1 and #3 slot into.
     STARTED (2026-09-05, include/intti/lkc.hpp): the CONSUMER seam for the
     DETERMINISTIC explicit-integral pair builders -- `detail::scatter_pair`
     (single AO matrix) and `detail::scatter_pair3` (three matrices) hoist the
     repeated "enumerate the (ka,kb) Cartesian components of pair (a,b),
     compute the row-major AO indices, write the block, mirror it (+1 sym / -1
     antisym / 0 none)" boilerplate behind a per-element value callback. First
     consumer: oneel.hpp, whose six builders (S, T, multipole, angular
     momentum, gradient, kinetic-moment) dropped ~11 duplicated scatter sites
     to value-lambdas; byte-exact (suite 206/206, PySCF 1e oracles unchanged).
     Deliberately NOT applied to the FUSED J/K engines (jbuild/kbuild) or the
     n^6 threeel builder: their density digestion is fused into the coupling
     and a generic consumer would de-fuse the hot path (= the #3 conclusion).
     Extended to ncenter's coulomb_2c (the (P|Q) symmetric block scatter is the
     same pattern; byte-exact). The remaining explicit-integral writes -- the
     3-center (mu nu|P) 3-index tensor (mu<->nu mirror, extra P stride) and the
     nuclear-attraction accumulate (+=, not =) -- need a tensor-stride and an
     accumulate variant of the consumer; low duplication (1-2 sites each), so
     deferred unless a third caller appears. The pervasive-duplication win
     (oneel, 11 sites) is captured; #2 as a clean non-regressing refactor is
     essentially complete. The remaining SHARK lever with real payoff is #5
     (general contraction, needed for def2/ANO), assessed next.
  3. **Digestion / permutational-redundancy** (SHARK Sec 3.6, Eq 29-37) --
     ANALYSED, DOES NOT APPLY to our J-engine. SHARK's digestion redundancy
     de-duplicates EXPLICIT integrals; jbuild is a J-engine that never forms
     them -- it fuses the Hermite-Coulomb coupling (the hermite_b B arrays) with
     the density contraction (t1/t2/jp stages). The bra-ket symmetry (p|q)=(q|p)
     makes only the B arrays shareable across (p,q)/(q,p); the dominant
     contraction is density-specific (dq(q)->jp(p) vs dq(p)->jp(q)) and cannot be
     shared. Measured (`prototype/jengine_digest_bench.cpp`): B is 0.8-2.5% of the
     per-node cost (contraction 97.5-99.2%), so a triangular+atomic-scatter
     version saves ~1-2% AT BEST and the atomics would erase it. So #3 is not
     worth doing -- the same conclusion as #1: SHARK's #1 (BLAS-3) and #3
     (digestion) both optimize its explicit-integral MD architecture, while our
     t-quadrature J-engine is already integral-free, past the regime they
     optimize. (Where #3 could ever apply -- the explicit-integral consumers
     schwarz/cholesky via eri_quartets -- those are not the Fock hot path.)
  4. **Helgaker-Taylor D,P derivative reformulation** (SHARK Eq 55-60) --
     ASSESSED: genuinely APPLIES (unlike #1/#3), a moderate win on the DERIVATIVE
     surface, substantial to implement. Shift to (D=R_A-R_B, P) so dOmega/dP just
     shifts the Hermite index (Lambda_t -> Lambda_{t+1}) and dOmega/dD uses a
     same-length E'-recursion; then dA = a/p dP + dD, dB = b/p dP - dD (no
     incremented sums, no promoted integrals). The 1e derivatives are ALREADY
     efficient (deriv.hpp computes the 1D overlap table once at the extended bra,
     then a cheap per-axis shift bra_shift -- essentially the same idea), so ~no
     win there. The 2e derivatives are the target: erigrad/erihess recompute FULL
     quartets at l+1 (and l-1) per shell position via eri_block4. Measured
     (`prototype/deriv_promote_bench.cpp`): an l+1 quartet costs 1.0-1.6x the base,
     so the gradient's ~4 positions x (promote+demote) is ~8x base per quartet;
     D,P would get all 3 components per position at ~base cost -> ~1.8x on the 2e
     gradient, and more on Hessians (l+2 promotes). BUT the derivative surface is
     per-geometry-step, not the per-SCF-iteration Fock hot path, so the practical
     payoff of a substantial E'-recursion + P-shift quartet-derivative
     implementation is moderate -- worth it mainly if frequency /
     geometry-optimisation throughput becomes a priority.
  5. **General & partial-general contraction** (SHARK Sec 3.3-3.4): "giant"
     E-matrices per atom+angular-momentum GEMM-contracted; PGC via
     decontract -> recontract. Needed for efficiency on real def2/ANO basis sets.
     STARTED (2026-09-05, include/intti/contracted.hpp). User decision: make the
     native matrix API NATIVELY generally contracted via CONTRACTION-AWARE
     BUILDERS (shared primitive intermediates), NOT a decontract/recontract
     C^T M C wrapper (which would forfeit the GC efficiency it exists to give).
     New representation: `ContractedShell` (nprim exponents + nprim x nctr
     coefficient block, one shell = many contracted functions sharing the
     primitive set) and `ContractedBasis` + `make_contracted_basis`. AO layout
     within a shell is contracted-outer, Cartesian-inner (c*ncart(l)+k, matching
     the libcint facade). Normalization convention (pinned to PySCF cart=True so
     the matrices compare directly to int1e_*_cart): the weight on the engine's
     UNNORMALIZED primitive is d_{cp} * cart_norm_pyscf(l, alpha_p). The builder
     evaluates each primitive-pair 1D table ONCE and reuses it across every
     contracted-function/Cartesian pair -- the shared-intermediate GC path.
     Increment 1: contraction-aware overlap_matrix + kinetic_matrix, validated
     (tests/test_contracted.cpp, suite 210/210) against (a) an independent
     closed-form contracted-Gaussian self-overlap and (b) a decontract ->
     primitive -> recontract reference off the analytically-validated primitive
     builders (isolating the contraction+normalization layer); the test basis
     uses a genuinely general contraction (s shell, 3 primitives -> 2 contracted
     functions). DONE: contracted multipole and nuclear (1e), then contracted J
     and K (the 2e payoff). J reuses the fused primitive J-engine on the
     contracted basis's primitive pairs, injecting contraction only in the
     density fold (D_eff = C^T D C, block-by-block, symmetry-folded) and the
     output gather (C J_eff C^T, accumulated) -- no nao_prim^2 matrix. K is
     MEMORY-LEAN too (user decision over a decontract/recontract wrapper):
     detail::exchange_build_contracted_impl runs the SAME per-primitive-quartet
     t-space core as exchange_build_impl on the contracted primitive pairs, but
     reads the ket density as an on-the-fly effective primitive density (summed
     over the two ket shells' contraction indices, never materialized) and
     atomic-scatters the primitive K block into the contracted K with the bra
     coefficients; the full primitive (a,b) loop + atomic accumulate gives the
     symmetric contracted K with no mirror, each primitive quartet evaluated
     once (shared-intermediate GC). J and K validated against the independent
     C (build(C^T D C)) C^T fused-engine references; suite 214/214. The native
     contracted Fock surface is now COMPLETE: S, T, V, J, K (+ multipoles).
     Remaining: contracted-K screening (currently unscreened), n-center (RI)
     contraction, and a PySCF cc-pVDZ cross-check of the whole native stack via
     an intti_dump driver.

  Note where we are AHEAD of SHARK and must not regress: range separation is a
  node-list (erf/erfc/Yukawa) with native position-dependent omega(r) for local
  hybrids -- SHARK's Boys-based route cannot easily do that; plus arbitrary
  precision and the higher-order delta tail. The factorization idea transfers,
  the Boys machinery does not.

- **M-QUAD — optimal Losilla tuning** (higher-order delta tail done; minimax
  node placement open). Use as small a truncation t_c as possible so the
  explicit t-quadrature covers only the expensive dense small-t region, pushing
  the cheap tail into analytic corrections. The **higher-order delta tail** is
  done (`tgrid.hpp`, `quartet.hpp`, `batch.hpp`): the truncated-quadrature tail
  is the full series

    R(t_c) = pi * sum_{k>=0} M_k / (4^k k! (k+1) t_c^{2k+2}),
    M_k = int rho_ab (nabla^2)^k rho_cd   (Laplacian-overlap moments),

  derived by expanding the ket density under the large-t Gaussian smoothing
  kernel e^{-t^2 r12^2} (references/sympy_tail.py verifies the coefficients and
  the M_k symbolically). k=0 is Losilla's leading delta pi*S/t_c^2 (the previous
  default, residual 1/t_c^4); each added order removes one 1/t_c^2 factor, so
  retaining orders 0..K leaves a residual O(1/t_c^{2K+4}). The moments cost
  almost nothing: a ket spatial derivative only raises the Hermite-integral
  index, so the per-axis order-m "derivative overlap" is D^{(2m)} = spref *
  sum_n fh_n B_{n+2m} -- the SAME fh coefficients the leading tail already
  builds, with B computed a few orders higher; M_k assembles from products of
  these per axis (multinomial over nabla^2 = sum_d d_d^2). Opt-in via
  `TGridSpec/TGrid::tail_order` (default 0 = unchanged; capped at TAIL_KMAX=4),
  precision-generic (serial __float128 and the Kokkos float/double/long double
  batch path give identical results). Validated (`tests/test_tail.cpp`): vs the
  exact (ss|ss) oracle the residual scales as 1/t_c^{2K+4} (doubling t_c drops
  the error by ~4^{K+2}: 16/64/256/1024 for K=0/1/2/3), and at a hard t_c=6 the
  order-3 tail reaches ~7e-12 where the leading tail is ~3e-5 -- a 4x smaller
  t_c (much cheaper explicit grid) for the same accuracy, exactly the goal.
  The matrix-level J and K builders carry it through: `jbuild.hpp`/`kbuild.hpp`
  fold the tail as one pseudo-node per Laplacian-order combination (a,b,c),
  a+b+c <= Ktail, each with weight b_{a+b+c}/(a!b!c!) and per-axis Hermite-index
  shifts 2a/2b/2c on the same contraction (the leading tail is the single
  (0,0,0) node, unchanged at tail_order=0). Validated (`tests/test_tail.cpp`):
  J and K over an s/p basis converge to the untruncated reference far faster at
  order 2 than order 0.

  Optimal order/truncation selection is done (`adaptive_linlog_tail_spec`, and
  the `ShellBasis` overload in `fock.hpp`). The series term_k ~ (-1)^k k^{-1/2}
  (rho/t_c^2)^k with rho = p q/(p+q) <= alpha_max, so it CONVERGES only for
  t_c^2 > rho: a hard floor t_c > sqrt(alpha_max) (the tightest pair), below
  which higher orders DIVERGE. The spec holds t_c a margin above the floor,
  models the order-K worst-pair residual as (rho/t_c^2)^{K+1}, and picks the
  (t_c, tail_order) that minimises total node count -- explicit LinLog nodes
  (which grow ~log t_c) plus the C(K+3,3) tail pseudo-nodes (cubic in K) -- to
  reach a target eps. So the cost trade (each order costs ~K^2/2 more
  pseudo-nodes but buys a smaller t_c) is resolved automatically, and the
  divergence floor is enforced (guards a real footgun: a small t_c + high order
  on a tight basis silently corrupts the tightest pairs). Validated
  (`tests/test_tail.cpp`): the worst-case tightest quartet (rho = alpha_max)
  meets eps = 1e-6 and 1e-10 across alpha_max in [1.5, 40] with t_c above the
  floor (the model is deliberately conservative -- ~2-3 orders of margin). The
  selector currently reuses `linlog_for` panel sizing, which is calibrated for
  the tested exponent regime; very tight all-electron cores (alpha_max ~ 1e6,
  floor t_c ~ 1e3) want separate validation of the explicit-grid resolution.

  The PSC/grid consumer (`product.hpp::interaction`) is audited: GTO x GTO
  routes through eri_quartet and so inherits the FULL higher-order tail
  automatically (tested); the grid-represented paths need the moments M_k = int
  rho_f (nabla^2)^k rho_g. GTO x cloud is now done
  (`detail::pair_component_laplacians`): M_k = sum_j w_j (nabla^2)^k rho_GTO(r_j),
  the Laplacian moved onto the analytic GTO and evaluated at the cloud points via
  the per-axis coefficient map Poly(y) -> Poly'(y) - 2 p y Poly(y); gto_cloud_sums
  returns the moment vector and interaction() folds sum_k a_k M_k. Validated
  (`tests/test_tail.cpp`): pair_component_laplacians vs a SymPy oracle
  (references/sympy_tail.py [4]) to 1e-11, and a GTO x point-cloud interaction
  converges to the exact sum_j w_j V_GTO(r_j) far faster at order 2 than order 0.
  cloud x cloud / coaxial PSC stay order-0 (their overlap is only estimated from
  the kernel value at t_c).

  Minimax [0, t_c] node placement was investigated and DEFERRED. The explicit
  part reproduces erf(t_c r)/r. Prototyped: (a) tanh-sinh on [0, t_c] needs MORE
  nodes than LinLog (Gauss-Legendre panels already match erf(t_c r)/r well);
  (b) least-squares-optimised weights on log-spaced nodes give ~2x fewer nodes
  at better accuracy (n=60 -> 9e-9 vs LinLog n=113 -> 1e-7) BUT the normal-
  equations solve is treacherously ill-conditioned -- a float128 Cholesky fails
  (non-positive pivot) as the node range widens, and Tikhonov regularisation
  caps accuracy at ~2e-6. A library-grade version needs a TSVD/regularised solve
  (e.g. jacobi_eigh at float128 with eigenvalue truncation) plus a LinLog
  fallback, for a bounded ~2x gain on nodes the tail has already made few. Given
  the tail already delivers the dominant savings and LinLog is at the optimal
  ~log(t_c)log(1/eps) rate, the value/effort is poor; deferred.

  The STO s-quadrature higher-order tail is done for any l (see M-STO): the l>0
  angular parity-derivatives compose with the radial Laplacians. Prototypes:
  prototype/sto_validation.py (STO = quadrature-contracted GTO, 5*zeta/8 to
  7e-12).

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

## libcint parity — integral-type coverage gaps (2026-09-05)

Where libintti stands against libcint's `intor` catalog. The nonrelativistic
scalar spine is largely there; the relativistic layer is entirely absent, and a
handful of nonrelativistic operators/derivatives are still missing. Matrix-level
API only throughout (no per-quartet public surface); each new family lands with
an independent oracle (PySCF `intor`, which wraps libcint, is the natural one).

**HAVE (nonrelativistic scalar spine).** Overlap S, kinetic T, nuclear
attraction V, arbitrary Cartesian multipoles, angular momentum <r x grad>;
Coulomb J and exchange K (direct + Cholesky + RI); 2-/3-centre Coulomb and
RI-J/K; GIAO first-order field derivatives of S/T/V/ERI; geometric gradients and
Hessians of S/T/V/ERI. Cart->real-spherical via c2s. Beyond libcint's usual
emphasis: range-separated erf/erfc/Yukawa ERIs native to the t-quadrature,
Gaussian geminals / transcorrelated, 3-electron integrals, Helmholtz/Yukawa,
local hybrids; all scalar-templated (real/complex/long double/__float128).

**MISSING -- relativistic layer (the whole category; not yet in any milestone).**
A dedicated Dirac/X2C integral layer. In rough dependency order:
1. **Spinor c2s** -- j-adapted (kappa, m_j) complex spinor transform, the
   analogue of c2s.hpp for the 2-component/4-component basis. The one genuinely
   new piece of machinery (everything else reuses the MD substrate). Prereq for
   all `_spinor` outputs.
2. **Small-component sigma.p 1e families** -- `int1e_spsp` (<sigma.p|sigma.p> =
   4 T for the kinetic-balance metric), `int1e_spnucsp` (<sigma.p|V|sigma.p>),
   `int1e_sprinvsp`. sigma.p on a Gaussian is a raise/lower (derivative) pattern
   we already compute; the new work is the Pauli 2x2 spin algebra + assembly.
3. **Small-component 2e** -- `int2e_spsp1` (SL), `int2e_sp1sp2`, and
   `int2e_spsp1spsp2` (SS|SS), i.e. the LL/SL/SS blocks of the Dirac-Coulomb ERI.
4. **Gaunt** (`int2e_ssp1ssp2` ...) and **Breit** (+ gauge `int2e_gauge_*`)
   two-electron interactions. Highest complexity; do last.
   Oracle: PySCF/DIRAC spinor integrals; validate LL block reduces to the
   nonrelativistic ERI and the sigma.p metric to 4T.

**MISSING -- nonrelativistic operators libcint has and we do not.**
- **Spin-orbit** (`INT_PSO`, 1e paramagnetic SO; 2e SOMF/Breit-Pauli) -- this is
  milestone M17 (`soc.hpp`), planned, NOT started ("can slip"). Overlaps with
  the sigma.p machinery above.
- **ECP** (effective core potentials). Split by architectural fit:
  * **Local part** `U_loc = sum_k d_k r^n e^{-zeta_k r^2}` -- GOOD fit: for even
    n a Gaussian-times-polynomial on the ECP centre, i.e. a 3-centre
    Gaussian-potential integral the MD substrate already does. Falls out of the
    finite-nucleus / INT_GAUSSIAN_POT work essentially for free.
  * **Semilocal projector part** `sum_l U_l(r)|lm><lm|` -- POOR fit, and the part
    that makes ECP worth having for heavy elements. Standard evaluation projects
    the Gaussians onto spherical harmonics about the ECP centre -> type-1/type-2
    radial integrals over modified spherical Bessel functions + angular Gaunt
    coefficients: a DISJOINT numerical method (Bessel radial quadrature + real-SH
    angular algebra), not a t-quadrature/MD reuse. It would be a bolt-on engine
    that breaks the "one t-quadrature MD engine" thesis. DECISION (2026-09-05):
    not a parity target. For heavy elements prefer the architecture-consistent
    all-electron relativistic (X2C/Dirac) route below -- sigma.p genuinely reuses
    the MD derivative substrate, ECP-semilocal does not. Revisit only if a
    concrete need arises where all-electron relativistic is too costly. (Same
    principle as the NAO decision: keep what the transform justifies, not a
    bolted-on disjoint method.)
- **Momentum-operator 1e** -- `int1e_pnucp` (<grad.V grad>) and `int1e_pnucxp`
  (<grad x V grad>, the PSO/SO seed). Used by X2C/DKH picture-change and SO.
- **Finite-nucleus / Gaussian-charge potential** (`INT_GAUSSIAN_POT`) -- a
  Gaussian nuclear charge model in place of the point charge. Small: swap the
  point-charge t-node kernel for the finite-width one (the tail machinery
  already handles a Gaussian smear).
- **Spherical multipoles** (`INT_SPHER_MULTIPOLE`) -- have Cartesian; c2s
  handles the transform, so this is a thin wrapper, not new integrals.

**PARTIAL -- derivative cross-product.** libcint exposes a large combinatorial
set of nabla (ip), GIAO (ig), and 1/r (rinv) derivative combinations
(`int2e_ip1ip2`, `int1e_ipiprinv`, `int1e_ipipnuc`, `int2e_ig1ip2`, ...). We
have geometric gradient + Hessian of S/T/V/ERI and GIAO first derivatives -- a
solid subset but not the full cross-product. The gen1int N-ary-tree approach
(arbitrary derivative order distributed over centres/fields) is the intended way
to close this generatively rather than case by case; adopt it in the M16
derivative layer so higher mixed orders come for free.

**Suggested ordering.** The nonrelativistic gaps are the cheap, high-value wins
and mostly reuse the existing substrate: finite-nucleus potential and spherical
multipoles (thin) -- and the finite-nucleus work also delivers **local ECP** for
free -- then `pnucp`/`pnucxp` + M17 spin-orbit (shared sigma.p algebra), then the
derivative cross-product via the N-ary tree. The full relativistic Dirac/X2C
layer is the one large standalone milestone: gated on the spinor c2s (item 1),
best sequenced LL -> SL -> SS -> Gaunt/Breit, each oracle-checked against its
nonrelativistic limit -- and it, not semilocal ECP, is libintti's heavy-element
story (see the ECP decision above).

## M-FE -- tensorial finite-element integral engine (2026-09-05)

A second integral route alongside the GTO/STO one: represent functions on a
tensorial (Cartesian-product) finite-element grid and evaluate every integral by
grid quadrature. Strategic payoff: it is OPERATOR-AGNOSTIC -- spin-orbit, Gaunt/
Breit, ECP, finite-nucleus, arbitrary range-separation all become "an operator
on the grid," so the whole libcint-parity breadth (above) and the relativistic
layer come by construction rather than as N analytic integral families. It also
IS the 3D grid solver we want for Helmholtz-SCF (M-HK): the FE machinery and the
orbital update (psi <- -2 G_kappa V psi) reuse our Yukawa/Helmholtz kernels.

**Attribution (the shared kernel identity is old -- credit it correctly).** The
whole family rests on the Gaussian/Laplace ("proper-time") transform of 1/r,
  1/r = (2/sqrt(pi)) \int_0^\infty e^{-r^2 t^2} dt
      = (2/sqrt(pi)) \int_0^\infty w^{-2} e^{-r^2/w^2} dw   (w = 1/t),
which is the same identity the Boys function and our t-quadrature use. It is
decades old: the plain Laplace transform; the Gaussian-transform molecular-
integral method of Shavitt & Karplus (1960s); White, Wilkins & Teter (1989,
eqn 25) as the finite-element realization; Losilla & Sundholm et al. (DAGE,
2010) as the tensorial-real-space-grid realization. DAGE/White contributed the
grid/FE discretization + accuracy/scaling engineering, NOT the kernel identity.
Document this lineage in the numerical-methods notes and citations.bib (do not
credit DAGE with the 1/r representation).

**Design decisions (user, 2026-09-05).**
1. **Tensorial (Cartesian-product) grid**, deliberately, over adaptive/
   unstructured FE. It over-resolves in the periodic copies / far regions, but
   its attraction is SIMPLICITY: the Coulomb kernel factorizes by Cartesian
   dimension exactly like GTO integrals (White eqn 22: operator matrix elements
   are 1D integrals x Kronecker deltas), so the whole separable-1D + delta-tail
   apparatus we already have applies unchanged.
2. **Two routes, permanently**: GTO/STO (the current engine) and FEM. They share
   the kernel machinery (tgrid.hpp t-quadrature, multipole.hpp T(R), the Losilla
   delta-tail, Yukawa/Helmholtz), differing only in the basis the integrand is
   discretized on.
3. A Gaussian is not a polynomial, but its **polynomial (FE) approximation is
   numerically exact** to any target by element order/refinement -- and gridding
   the smooth Gaussian basis is easy (~a dozen points along 1/sqrt(alpha) per
   function; the per-nucleus grid is the union of the per-function grids).
4. **No n^3 grid tensor is ever stored -- everything is ELEMENT-PAIR local.**
   The unit of work is a pair of Cartesian 3D boxes (elements), exactly like a
   GTO shell-pair. For an integral (pq|rs) the element pairs are screened by
   separation:
   * **far box pairs -> multipole** (reuse multipole.hpp T_{tuv}(R); the box's
     local polynomial density -> its moments, contracted through the exponent-
     free interaction tensor). This is the M-MP far-field, with a box in place of
     a Gaussian pair.
   * **near box pairs -> t-quadrature** (the separable Cartesian Coulomb above).
   * **large-t within a near pair -> multipole / moment again** (the short-range,
     nearly-local end of the t-integral is the Losilla delta-tail we already do:
     a local moment correction, not dense quadrature).
   So the FE engine is structurally the SAME screened pair loop as the GTO J/K
   build (shell-pair x shell-pair, multipole-far / quadrature-near), with box
   elements as the pairs -- which is why the storage is O(elements) working set,
   not O(n^3), and why it inherits our screening/tiling directly.

**Multi-center.** Handled by the tensorial choice (1): a global Cartesian box
grid (or per-nucleus tensorial grids) with the element-pair loop bridging
centers via the far/near split -- accepting the over-resolution that the product
structure costs, for the factorization simplicity.

5. **t-ADAPTED element-pair quadrature is the crux for machine precision**
   (user, 2026-09-05). The kernel matrix element over two elements,
   \int\int L_i(x) e^{-t^2(x-x')^2} L_j(x') dx dx', has a large-t "knife's edge":
   a fixed grid suited to the polynomials L_i L_j misses the sharp Gaussian --
   the classic pitfall of Sundholm-group naive implementations. Fix: substitute
   x' = x - v/t, so the inner integral becomes (1/t)\int L_j(x - v/t) e^{-v^2} dv
   with e^{-v^2} FLAT for all t; Gauss quadrature in v samples the (piecewise-
   polynomial) shape function at near-constant argument -> EXACT for any t at
   fixed order, no delta-tail needed. i.e. redistribute the quadrature *along the
   exponential*, not along the polynomial. For a piecewise-POLYNOMIAL density
   this one branch is t-uniform; for a Gaussian density (as in the collocation
   prototype) it fails at SMALL t (the density becomes the knife's edge in v), so
   a near/far t-split is used there: direct x'-grid for small t, t-adapted for
   large t (overlapping validity -> machine precision across all t).
6. **Large-t asymptotics = the Losilla delta-tail, exact for polynomials**
   (user, 2026-09-05). Steepest descent on the flat-kernel inner integral
   (Laplace point v=0, Taylor L_j about x, Gaussian moments) gives
     (1/t)\int L_j(x-v/t) e^{-v^2} dv = (sqrt(pi)/t) sum_m L_j^{(2m)}(x)/(4^m m! t^{2m})
     = (sqrt(pi)/t)[ L_j(x) + L_j''(x)/(4t^2) + L_j''''(x)/(32 t^4) + ... ].
   For a polynomial L_j this series TERMINATES -> an exact, finite closed form
   for the large-t regime (moments/derivatives of the shape function, no
   quadrature). The leading term is the delta/overlap; the higher terms are
   exactly the Losilla higher-order delta-tail series (M-QUAD, the 1/t_c^{2k}
   Laplacian-moment corrections) -- the FE side re-derives the same tail. So the
   two routes share not only the 1/r kernel identity but the tail series; the
   complete recipe is t-adapted quadrature for near/moderate t + the finite
   moment expansion for large t.

**Plan (one-center first -- self-contained, perfect oracle).**
1. Per-nucleus tensorial FE grid; represent the Gaussian basis on it; evaluate
   1e integrals and (pq|rs) on-grid via the element-pair loop; validate to
   machine precision against the ANALYTIC t-quadrature values we already produce
   (clean CPU oracle -- no external reference needed).
2. Multi-center element-pair bridging (the real research step): the far/near box
   split across nuclei on the global tensorial grid.
3. Fold in the Helmholtz grid solver (the M-HK capstone) on the same FE machinery.

**Status (2026-09-05): one-center step-1 COMPLETE and machine-precise**
(prototype/fe_onecenter.cpp, validated against eri_quartet + analytic).
- Overlap by tensorial grid quadrature: MACHINE-EXACT (1.8e-15) -- the Gaussian
  basis representation claim holds.
- (pq|rs) via tensorial grid + Mobius t-quadrature + near/far-split t-adapted
  inner quadrature (design point 5): MACHINE-EXACT vs eri_quartet -- (ss|ss)
  9.8e-15, (pp|ss) 1.1e-15, (pp|pp) 4.4e-15 -- with NO delta tail. (Parity-zero
  components like (ps|ss) are correct at ~1e-16.)
- Confirmed en route: naive fixed-grid + leading delta-tail is only ~1/t_c^2
  (8e-4->5e-5); the t-adapted element-pair quadrature beats the naive one
  100-600x at large t (t=256: 1.3e-3 vs 0.20) on polynomial elements.
**Step-2 DONE: element-subdivided piecewise-polynomial FE representation**
(prototype/fe_fem_onecenter.cpp). The axis is split into elements; on each the
density factor is a local polynomial (barycentric Lagrange through the element
Gauss-Legendre nodes). (pq|rs) is the ELEMENT-PAIR loop A_d(t) = sum_{eA,eB}
int_eA int_eB P_eA e^{-t^2(x-x')^2} P_eB, and answers the two structural
questions: NEAR pairs (same/adjacent elements, x,x' can come within ~1/t) use
the t-adapted inner quadrature -- and because P_eB is a POLYNOMIAL it is smooth
in v for ALL t, so a SINGLE t-adapted branch is machine-exact with no near/far
t-split (the split was only an artifact of the Gaussian collocation); FAR pairs
(well separated) have an empty clamped v-range and are auto-SCREENED (the
"different element factorizes/screens" case). Validated vs eri_quartet: overlap
9.7e-15, (ss|ss) 1.1e-14, (pp|pp) 1.1e-14 at 20 elements x degree 13, and
spectrally convergent under refinement (12x9 -> 20x13 tightens ~1e-12 -> 1e-14).
Element-pair census showed ~45% of pairs screened as far. NB the far pairs are
merely screened here; the efficient treatment multipole-factorizes them (design
point 4) -- the next optimization.
Next: (i) multi-centre element-pair bridging (the far/near box split across
nuclei, with multipole for far); (ii) the Helmholtz grid solver on the same FE
machinery (the piecewise-polynomial elements also give the cusp resolution the
solver needs).

**Exchange on the FE grid = co-density Coulomb builds (user, 2026-09-05).** There
is no cheap "exchange kernel" on a grid, but with D = occ_scale sum_i C_i C_i^T,
  K_pq = sum_rs (pr|qs) D_rs = occ_scale sum_i (p phi_i | q phi_i),
i.e. K is a SUM OVER OCCUPIED ORBITALS of Coulomb interactions between the
co-densities g_pi(r) = chi_p(r) phi_i(r): K_pq = occ_scale sum_i (g_pi | g_qi).
So exchange = nocc cheap grid-Coulomb builds, each producing an Nbf x Nbf matrix
(fits memory; accumulate over i). Screening: g_pi is confined to where chi_p
lives (local even though phi_i is delocalized), so negligible (p,i) co-densities
drop and the element-pair near/far split screens the grid Coulomb -- linear
scaling, best with LOCALIZED occupied orbitals. This is exactly seminumerical
exchange (Friesner pseudospectral / Neese chain-of-spheres COSX, one electron
numerical; here both via DAGE on the full grid) and the real-space twin of the
occ-driven ri_k_occ already in the library (K = sum_i W_i W_i^T). Caveats: cost
is nocc Coulomb builds (payoff = generality/scaling, not beating analytic K on
small GTO systems); and the co-densities inherit phi_i's NUCLEAR CUSP, so
near-nucleus grid resolution (refinement / bubbles) is the accuracy pressure
point -- exchange energy is more grid-sensitive than Coulomb. Fold into M-FE
once multi-centre Coulomb is in place; validate one-centre K via co-densities
against the analytic exchange_build.
PROTOTYPED (prototype/fe_exchange_onecenter.cpp, 2026-09-05): the co-density /
occupied-orbital ASSEMBLY K_pq = occ_scale sum_i (pi|qi), (pi|qi) = sum_rs C_ri
C_si (pr|qs), driven through the per-orbital loop on the machine-precise
separable grid (pr|qs) (step-1), reproduces exchange_build to 1.3e-15 (s-basis,
one centre). This validates the co-density formulation + occupied-orbital
assembly. The genuine seminumerical form is now also PROTOTYPED
(prototype/fe_exchange_dage.cpp, 2026-09-05): form g_qi = chi_q phi_i as ONE 3D
grid tensor and take a SINGLE 3D DAGE Coulomb solve per (q,i) (no O(Nbf^2) r,s
expansion), then K_pq = occ_scale sum_i int g_pi V_qi. Reproduces exchange_build
to rel 1.1e-4 at a coarse grid (5 elem x deg 7, N=40/axis; s-basis, 1 occupied),
representation-limited and spectrally convergent (the DAGE quadrature is
machine-exact per steps 1-3). This is the production seminumerical K on the grid
(pseudospectral/COSX-style), built on the step-3 3D DAGE; cost = nocc*Nbf DAGE
solves. Combined with the multi-centre DAGE, FE exchange for molecules is
de-risked. Remaining: accuracy/cost tuning (near-nucleus resolution for the
cusped co-densities), localized-orbital screening, and library-ization.

**Helmholtz/Yukawa apply on the FE grid VALIDATED** (prototype/fe_yukawa_dage.cpp,
2026-09-05) -- the last core operator for the real-space SCF solver (M-HK
capstone). G_kappa = e^{-kappa r}/(4 pi r) applied to a grid function is the SAME
3D t-adapted DAGE, only with the YUKAWA t-grid (the exp(-kappa^2/4t^2) weight is
already in make_tgrid(yukawa)). Validated (rho|e^{-kappa r}/r|rho) for a
non-separable density vs the analytic Yukawa from eri_quartet: rel 2.1e-7 at
6 elem x deg 8 (N=54/axis), kappa=1. So ALL grid Fock/solver operators now exist
and are validated on the FE grid: Coulomb J (DAGE), exchange K (co-density +
DAGE), and the Helmholtz orbital-update kernel G_kappa. The grid SCF driver
psi <- -2 G_kappa V psi (the previously efficiency-gated M-HK capstone) is now
an ASSEMBLY of these validated operators + convergence control (the cusped 1/r
potential needs near-nucleus FE refinement / bubbles for accuracy).

**Grid Helmholtz SOLVER STEP validated** (prototype/fe_helmholtz_solver.cpp,
2026-09-05): the fixed-point identity psi = -2 G_kappa(V psi) for an EXACT
eigenpair -- decisive test needing no SCF convergence control. On a shifted
harmonic well V=2a^2 r^2 - C (smooth, no cusp; exact ground state e^{-a r^2},
eps=3a-C<0), one grid Green's-function step (grid V-multiply + Yukawa-DAGE
apply, psi_new = -(1/2pi) YukawaDAGE(V psi)) reproduces the exact eigenfunction
to rel 1.2e-6 at 6 elem x deg 8 (N=54/axis). So the SOLVER core operator is
correct; the full real-space SCF is now the iteration assembly (energy update
e.g. Rayleigh/GF-power + damping/DIIS + near-nucleus refinement for cusped
potentials), validated against a GTO-basis diagonalization reference. The M-HK
capstone is unblocked: all its grid operators are built and validated.

**Grid Helmholtz SCF driver -- TEST HARNESS, not library** (prototype/
fe_scf_solver.cpp, 2026-09-05). SCOPE DECISION (user): libintti is an integrals
library -- it ships the grid OPERATORS (DAGE Coulomb, co-density K, the Helmholtz
apply G_kappa); the SCF DRIVER (the psi <- -2 G_kappa(V psi) loop, energy update,
convergence control) is application-level and lives only as a test/oracle, never
in a library header (like the RHF-via-libintti validation, never library code).
Result: the self-consistent Green's-function power iteration on a shifted
harmonic well (exact ground state e^{-r^2/2}, eps=-2.5), from a WRONG guess
e^{-0.8 r^2}, converges MONOTONICALLY to eps=-2.49960 (diff 4.0e-4 at the coarse
N=32/axis grid) with overlap^2 -> 0.9989, in 4 iterations, and stops at the grid
limit. The Helmholtz integral iteration is well-behaved (as expected -- G_kappa
is a bounded smoothing operator; no differential ill-conditioning); an earlier
apparent divergence was a SIGN error in the Kalos/BSH energy update
(eps += <V psi | psi~ - psi>/<psi~|psi~>), not the method. So the M-HK capstone
is demonstrated end to end on the grid; production robustness (finer grids,
KAIN/Anderson acceleration) is a driver/application concern, not libintti's.

**LIBRARY-IZATION started -- include/intti/fegrid.hpp** (2026-09-05): the first
FE-engine library header, promoting the validated prototype machinery to
templated, CI-tested code. Contents: `FEGrid1D<Real>` + `make_fegrid1d(ne,np,L)`
(the tensorial FE grid -- ne elements x np Gauss-Legendre nodes per axis,
barycentric-Lagrange local polynomials), `detail::fe_conv1d` (the t-adapted 1D
convolution -- the core operator), `fe_dage3d(grid, tgrid, rho, nv, vmax)` (the
general-density DAGE Coulomb/Yukawa potential, kernel selected by the TGrid so
`make_tgrid(coulomb())` gives 1/r and `make_tgrid(yukawa(kappa))` the Helmholtz
apply), and `fe_inner` (grid inner product). Matrix/grid-level only, no per-
quartet surface. CI tests (tests/test_fegrid.cpp, ~1.8s, suite 206/206): grid
quadrature exactness (1e-9), the t-adapted 1D convolution vs the analytic
Gaussian-Gaussian convolution across t in {0.5..128} (1e-6, confirming t-
uniformity), and a tiny 3D DAGE Coulomb self-energy smoke test (5e-2 at a small
grid). Tight accuracy + spectral convergence stay in the prototypes (too slow
host-serial for CI). This first version is host (serial); the tensor-at-a-time
API is shaped for a later Kokkos/GPU port (team-scratch over lines). Next:
promote the element-pair (pq|rs) / co-density K paths, wire the operators to the
ShellBasis matrix API, and Kokkos-ize.

**Step-3 (3D t-adapted DAGE) PROTOTYPED** (prototype/fe_dage3d.cpp, 2026-09-05):
the Coulomb potential of a GENERAL 3D density on the tensorial FE grid,
V(r1) = sum_t w_t int rho(r2) e^{-t^2|r1-r2|^2} dr2, done as three successive 1D
convolutions along z,y,x (kernel factorizes per axis). Each 1D convolution uses
the t-adapted substitution u2=u1-v/t and evaluates the FE function at the
off-grid points u1-v/t by barycentric-Lagrange interpolation within the source
element (far elements auto-screen via the clamped v-range) -- machine-exact per
axis for all t, no delta tail. Because separability is in the KERNEL not the
density, this handles NON-separable densities (the co-densities that broke the
step-1/2 pair-density factorization). Validated on a non-separable density (sum
of s-Gaussians) vs the analytic Coulomb, and SPECTRALLY CONVERGENT: rel error
2.4e-5 at 5 elements x degree 7 (N=40/axis) -> 5.1e-8 at 7 elements x degree 8
(N=63/axis), a ~460x drop for a modest refinement -- confirming the residual is
REPRESENTATION-limited (the t-adapted quadrature is machine-exact per steps
1-2). This is the engine that unlocks the single-3D-tensor co-density
exchange (form g_qi once, one DAGE solve per (q,i)) and the grid SCF solver.

**Multi-centre 3D DAGE validated** (prototype/fe_dage_multicenter.cpp,
2026-09-05): the SAME 3D DAGE on a GLOBAL tensorial box grid covering two nuclei,
applied to a non-separable TWO-CENTRE density, reproduces the analytic
two-centre Coulomb -- CROSS-CENTRE term included (Boys F0) -- to rel 9.5e-8 at
6 elements x degree 8 (N=54/axis), bond separation 1.4. The machinery is
unchanged (separability is in the kernel); the molecular case is just a bigger
box and a multi-centre density, and far element pairs auto-screen in the 1D
convolution. This is the user's chosen tensorial (global-grid) route -- it
accepts over-resolution in the far regions for the Cartesian-factorization
simplicity. So multi-centre Coulomb for general densities is de-risked; the
multipole far-field (design point 4) remains the EFFICIENCY layer for large
separations (replace the auto-screened far element pairs with a multipole
interaction) rather than a correctness need.
Remaining: refine to production accuracy + multipole-far efficiency + screening/
cost tuning; wire the 3D DAGE into a real co-density K build (molecular) and the
Helmholtz grid solver; turn the prototypes into library code.

**DESIGN NOTE -- grid-RI: the FE route IS resolution-of-the-identity, and the
grid is the auxiliary basis** (user insight, 2026-09-05). The DAGE/FEM route
already prototyped is exactly an RI: it resolves the density into a compact
intermediate (the FE grid) instead of an auxiliary GTO set, computes the Coulomb
potential V of that representation directly via the t-quadrature convolution, and
contracts J_mn = <chi_m chi_n | V>. Versus GTO-RI it DROPS the (P|Q)^{-1} Coulomb-
metric solve and the fit coefficients entirely -- DAGE gives V from rho exactly
(to grid resolution), so it is a fitting-free, metric-free RI; it is grid-linear
(no 3-index (mn|P) tensor); and the kernel is generic (Coulomb / erf / erfc /
Yukawa-Helmholtz / position-dependent omega(r) for local hybrids) where the GTO
metric is not. K follows via co-densities (g_pi = chi_p phi_i), already
prototyped.

The real design question is GRID CONSTRUCTION, and RI answers it: the grid must
span the IMPORTANT part of the {pq} product space, which RI shows is only O(nao)
wide (aux ~ 3-4 x nao), NOT the full O(nao^2) product set -- so an adapted grid
sized to that span, not to every product, is where the efficiency is. Two
composable, automatable constructions:
  1. Cholesky-pivot-driven: reuse two_step_cholesky (cholesky.hpp) -- already a
     pivoted Cholesky over the pair space that auto-selects the O(nao) important
     pair-density directions (RI's aux auto-construction) -- as the TARGET SPAN,
     and locally h-refine the FE grid only until it represents those pivot
     densities to tolerance. This literally ties the two pillars: the RI aux
     selection defines the grid.
  2. Error-indicator adaptive h-refinement (standard adaptive FEM): subdivide
     elements where the local interpolation residual of the represented density
     / potential exceeds a threshold -- automatic, atom-centred (cusps drive
     refinement), element-local so it stays cheap.
Subtlety: the final <pq|V> contraction still needs pq resolved WHERE the AOs
live, but chi_m chi_n has compact support and V is smoother than rho, so the same
atom-centred adapted grid serves both the (global, smooth) potential and the
(local) product quadrature. This grid-RI construction is the concrete plan for
the near-nucleus refinement / bubbles + multipole-far items above, and the
efficient bridge between the RI pillar (ncenter/ri.hpp) and the FE pillar
(fegrid.hpp).

**GRID CONSTRUCTION -- tensorial per-axis hp-adaptive mesh** (user, 2026-09-05).
Adaptive FEM suffices and converges rapidly BECAUSE the product set {chi_p chi_q}
is massively linearly dependent (rank ~O(nao) vs O(nao^2) products), so resolving
the PRIMITIVES resolves the whole product span -- convergence is set by the
primitives, not the products. The cost worry is answered by the tensorial +
separable structure: a Cartesian-product grid, and a Gaussian primitive
factorizes per axis (exp(-a((x-X)^2+(y-Y)^2+(z-Z)^2)) = product of 1D Gaussians),
so building the grid REDUCES TO THREE INDEPENDENT 1D PROBLEMS -- on each axis,
place nodes so every 1D Gaussian {exp(-a (x-X)^2)} that appears (all (a, X_axis))
is resolved to tolerance. O(nprim) 1D work, trivially fast, and it is EXACTLY the
per-axis structure DAGE needs (kernel factorizes -> 1D convolutions), so the
tensorial choice serves construction AND operator. hp-adaptive in the Gaussian
case: a 1D Gaussian is entire, so barycentric-Lagrange interpolation on an
element converges EXPONENTIALLY in the order p (spectral) -- the optimal 1D mesh
is few elements at high p, chosen (h = element sizes around each centre/width,
p = order) so each primitive is resolved with minimal DOFs; a classic, cheap 1D
hp-mesh design solvable greedily per axis. Honest cost caveat: the tensor product
of three fine 1D meshes creates fine cells across the whole (x-planes)x(y-planes)
x(z-planes) lattice, including "phantom" fine cells far from any atom -- the known
tensorial-vs-atom-centred overhead -- but DAGE's per-line 1D convolutions and the
element-pair screening (clamped v-range) make even a large tensor grid affordable.
SLATER densities (STOs): non-separable (r couples axes) and cusped, but the
library's STO route already writes exp(-zeta r) as the Gaussian integral
transform + delta-tail correction; each e^{-t^2 r^2} component IS separable, so
the tensorial 1D construction applies per t-node (the Slater becomes a t-
superposition of separable Gaussians), and the cusp is captured by the analytic
delta-tail correction rather than the grid -- the same trick already used for STO
integrals. So both GTO and STO grid construction stay 1D Gaussian-resolution
problems. Concrete next prototype: a per-axis 1D hp-mesh generator over a set of
(exponent, centre) 1D Gaussians to a target accuracy, feeding make_fegrid1d /
fe_dage3d.
PROTOTYPED (prototype/fe_hp_mesh.cpp, 2026-09-05): the per-axis 1D hp-mesh
generator -- recursive element refinement that tries increasing order p (Gauss-
Legendre + barycentric) up to pmax and bisects (h) only when no p reaches eps,
seeded at the atom centres. On a deliberately stiff set (24 Gaussians, exponents
1e-2..1e4 over 3 centres) to eps=1e-8 it builds a 48-element / 610-DOF mesh at
worst error 9.9e-9, versus 45856 DOFs for the uniform-order/uniform-element mesh
of equal accuracy -- 75x fewer DOFs, confirming the hp/tensorial construction is
both cheap (a 1D problem) and dramatically more compact than a uniform grid.
VARIABLE-ORDER DAGE PROTOTYPED (prototype/fe_hp_dage.cpp, 2026-09-05): a VGrid1D
carrying a distinct GL order per element, with the t-adapted 1D convolution and
3D DAGE generalized to it (the uniform FEGrid1D cannot express variable order),
driven by the hp mesh. Validated: the Coulomb self-energy of a non-separable
two-Gaussian density on the hp grid matches the analytic double-sum oracle to
1.95e-5 -- the hp mesh drives a correct DAGE end to end. (The DOF win is
quantified at construction level above; a DAGE-level uniform comparison is
impractical host-serial precisely because the uniform reference needs so many
points -- which IS the hp advantage.)
LIBRARY-IZED (fegrid.hpp, 2026-09-05): FEGrid1D generalized to per-element order
(nps/noff + per-element ref nodes/bary weights); make_fegrid1d(ne,np,L) is now
the uniform special case (byte-identical, existing tests pass) built via
detail::make_fegrid1d_elements; and make_fegrid1d_hp(gaussians, eps, pmin, pmax)
is the per-axis grid constructor (domain to amplitude eps, seed at centres, hp-
refine). fe_conv1d/fe_dage3d/fe_inner consume variable order unchanged. CI:
HpGridResolvesGaussian + HpDageCoulombSmoke (suite 217/217).
GRID-RI J ON A MOLECULE PROTOTYPED (prototype/fe_gridri_j.cpp, 2026-09-05): the
grid-RI payoff on a real GTO basis. Build the total density rho = sum_uv D_uv
chi_u chi_v on the hp grid, DAGE -> V, contract J_uv = <chi_u chi_v | V>, and
compare to the library's EXACT GTO coulomb_build on the same unnormalized-
primitive basis + density. Result (4 s-AOs, two centres, hp grid 56 nodes/axis):
the full J matrix matches element-by-element, worst rel error 2.4e-4 -- with NO
auxiliary fit and NO (P|Q)^{-1} metric inversion, just density-on-grid + DAGE.
So the FE route reproduces the molecular Coulomb matrix, confirming grid-RI end
to end.
GRID-RI K (co-density exchange) PROTOTYPED (prototype/fe_gridri_k.cpp,
2026-09-05): K_uv = sum_i (ui|vi) via co-densities g_vi = chi_v phi_i -- for each
occupied orbital i (columns of M, D = M M^T), form g_vi on the grid, DAGE ->
V_vi, contract K_uv = sum_i <g_ui | V_vi>. Validated vs the exact GTO
exchange_build (4 s-AOs, rank-1 D, hp grid 32 nodes/axis): the K matrix matches
element-by-element, worst rel error 6.4e-3 (coarse grid; improves with
refinement). So BOTH grid-RI J and K reproduce the exact GTO Fock matrices via
density-on-grid + DAGE.
P (AND HIGHER-l) SHELLS PROTOTYPED (prototype/fe_gridri_jp.cpp, 2026-09-05): the
grid AO evaluator gains the x^lx y^ly z^lz polynomial factor (via cart_comp) --
the whole cost of higher angular momentum on the grid is a pointwise AO
evaluation (as predicted for spherical/contracted AOs). grid-RI J with an s+p+s
basis (5 Cartesian AOs) matches the exact GTO coulomb_build to worst rel error
1.65e-5. So the AO evaluator is general in l; the same holds for real solid
harmonics (2l+1 components) and contracted AOs.
LIBRARY-IZED (include/intti/gridri.hpp, 2026-09-05): the grid-RI Fock builders.
grid_for_basis(basis, eps) auto-builds the hp grid from the AO-product envelopes;
ao_values_on_grid evaluates every Cartesian AO on the grid (general-l via
cart_comp + the x^l factor, per-axis factorised); grid_coulomb_build (density ->
one DAGE -> contract J = <pq|V>) and grid_exchange_build (co-densities g_ui =
chi_u phi_i from occupied Cocc -> DAGE -> contract K). Matrix-level API (D /
orbitals in, J / K out; never per-quartet). Host reference (precomputes AO values
nao*N^3; Kokkos/streaming port is future). CI tests (tests/test_gridri.cpp):
grid-RI J on an s+p basis and grid-RI K (rank-1 occ) both match the exact GTO
coulomb_build / exchange_build (coarse grid, ~5s; suite 219/219). Next: real
solid harmonics + contracted AOs in the evaluator (both free, per the notes
below); a benchmark vs GTO-RI; and a Kokkos/streaming port.

ORTHONORMAL / MO-BASIS grid-RI is viable (user, 2026-09-05). For J it is a non-
issue by INVARIANCE: rho = sum D_uv chi_u chi_v is basis-independent, so V =
DAGE(rho) is computed once regardless of the D representation and grid-RI J is
already stable (never touches the primitive-integral cancellation). The
orthonormal-basis question is really about K, whose co-densities g_ui = chi_u
phi_i use the delocalized occupied orbitals. Concerns and why they are bounded:
(i) "fit all functions at once" = the molecular extent, which the AO-PRODUCT grid
already spans -- delocalized phi_i store dense values on the SAME grid, no larger
grid needed; (ii) delocalization is absorbed by the ELEMENT-PAIR processing
(near -> t-quadrature, far -> multipole): cost ~ element-pairs x t-nodes, not
per-function support, so a delocalized phi_i only loses some function-locality
screening, bounded by the far-field multipole -- a modest increase, not a blow-
up; (iii) MO integrals are dense, but MOOT under our matrix-level API -- we only
ever produce Fock-like quantities <chi|V>, never the 4-index MO ERIs. So grid-RI
in the orthonormal / natural-orbital basis buys the stability win (above) at
bounded extra cost.

**NUMERICAL STABILITY -- grid-RI evaluates in the contracted/orthonormal AO
basis, avoiding the contraction cancellation** (user, 2026-09-05). The analytic
route's transform (ij|kl) = sum C_ai C_bj C_ck C_dl (ab|cd) sums HUGE primitive
integrals (tight-Gaussian normalizations) against SIGN-ALTERNATING contraction
coefficients (near-orthogonal contracted functions / small overlap eigenvalues),
so a small result is formed by catastrophic cancellation of large terms -- the
ANO / near-linear-dependence pathology, worse the higher the linear dependence.
The grid route mitigates this structurally: it NEVER forms the huge primitive
integrals. It evaluates the contracted AO VALUE pointwise, chi_i(r) = sum_p C_pi
g_p(r) -- a sum of modest FUNCTION VALUES, not of huge 4-index integrals -- and
integrates the O(1) products on the grid. The cancellation moves from "huge
integrals" to "function values," far better conditioned. Since any transform is
free pointwise (the c2s/contraction-free property above), the natural extension
is to evaluate DIRECTLY in the ORTHONORMAL AO basis (Lowdin / natural orbitals)
on the grid, where the density's small-eigenvalue directions are handled stably
-- grid-RI is the natural home for stable orthonormal-basis integral evaluation.
(Honest caveat: pointwise chi_i still sums C_pi g_p(r) with primitive
normalizations, so near-nucleus cancellation is reduced, not zero -- but it is
the cancellation of a well-defined function value, and libintti's arbitrary
precision can be applied cheaply to the local AO evaluation if needed.)

**SPHERICAL AOs ON THE GRID -- c2s is free** (user, 2026-09-05). Spherical AOs
are handled in the ANALYTIC route via c2s.hpp (c2s_matrix), normalization.hpp
(sph_rescale) and the cint facade (build_sph_transform, int2e_sph) -- Cartesian
integrals contracted with the c2s matrix. The GRID route needs none of that: a
spherical AO is R_lm(x,y,z) e^{-alpha r^2} with R_lm a real solid harmonic
(homogeneous degree-l polynomial), so it is evaluated DIRECTLY at the grid
points -- the c2s "transform" is absorbed into pointwise AO evaluation, no matrix
contraction. Consequences: (i) grid-RI works in the compact 2l+1 space (vs ncart:
5 vs 6 for d, 7 vs 10 for f, 9 vs 15 for g) -> fewer co-densities/contractions;
(ii) evaluating the PURE solid harmonic never introduces the Cartesian
contaminant (the s in d-cart, p in f-...) that the analytic route forms and then
projects out; (iii) grid-RI is AO-representation-agnostic -- Cartesian,
spherical, or any contracted combination plugs into the SAME DAGE machinery,
needing only a pointwise evaluator, whereas the analytic route needs a bespoke
transform per representation. (For the analytic GTO-RI / coulomb_build route, c2s
stays a matrix contraction -- the grid advantage is specific to grid-RI.) So the
grid-RI AO evaluator should take real solid harmonics directly for l>=1.
The SAME mechanism makes CONTRACTION free (user, 2026-09-05): a contracted AO
chi_u = sum_p d_p g_p is evaluated pointwise as a single value, so on the grid
the density / co-densities are built from the CONTRACTED AOs directly -- the grid
cost scales with the number of final AOs (nao), NOT the primitive count (nprim)
or the Cartesian count. Pointwise evaluation absorbs BOTH the c2s transform AND
the contraction; the grid route is agnostic to HOW the AO is built (Cartesian or
spherical, primitive or generally contracted) and only needs its value. This is
a large win exactly where the analytic engine works hardest -- generally-
contracted / ANO bases with nprim >> nao -- where the analytic route (even the
shared-intermediate M-SHARK #5 path) still touches every primitive, while the
grid route touches only the nao contracted AO values per point. Grid-RI is thus
the natural home for heavily-contracted and spherical bases.
GPU note (user Q, 2026-09-05): hp variation is NOT a load-balancing problem for
the tensorial DAGE and is arguably good. The DAGE parallelism is over lines
(N^2/axis) + t-nodes, and in a TENSOR-PRODUCT grid every line along an axis uses
the IDENTICAL 1D mesh, so all parallel work-items do structurally identical work
-- no cross-warp/team imbalance; the order variation is a fixed shared pattern
inside each line's element loop (uniform occupancy). Compactness (75x fewer DOFs)
directly aids occupancy and pushes back capacity-tiling. The nuance is the
element-PAIR kernels (co-density exchange, near/far pair loops) where work ~
p_i*p_j varies: handle by BUCKETING pairs by order (p in {4..16}, few buckets) --
the standard mixed-order-FEM-on-GPU technique -- layered on the dynamic
scheduling the screening (clamped-v near/far split) already requires. Varying
element SIZE alone is a non-issue (work scales with p and pair count, not size).

## M-PERF -- high-rank loop / BLAS audit (2026-09-04)

A project-wide audit for loops whose cost scales with system size and could be
collapsed, factorized step-wise, or dispatched to BLAS. `detail::matmul_nn`
(square) and `gemm_nn` (rectangular) in cholesky.hpp are precision-generic (BLAS
for float/double, triple-loop fallback for long double / __float128 / MPFR), so
GEMM dispatch stays portable and extended-precision-safe. Ranked by leverage:

- **threeel.hpp:511-583 -- O(nao^6) three-electron sextet loop** (three_electron_
  energy / _fock). Builds and contracts every 3e integral individually. Because
  op23 is absent (A[1][2]=0), integrating out electrons 2 and 3 gives a FIELD
  form E = int rho(r1) W2(r1) W3(r1) dr1, and per (t,s) node the density folds
  into two independent pair-pair "field" tensors -- each a geminal-J coupling
  V2_{ad,t} = sum_be D_be <ad|e^{-t^2 r12^2}|be>, V3_{ad,s} = sum_cf D_cf
  <ad|e^{-s^2 r13^2}|cf> (O(n^4) per node, cf. coulomb_build) -- which are then
  contracted with electron 1's density through a 3-way Hermite overlap:
  O(n^6 nt^2) -> O(n^4 nt^2). NOTE the reduction is NOT a trivial factorisation:
  though A[1][2]=0 (no op23), the INVERSE is dense -- Ai[1][2] = t^2 s^2 / det
  != 0 (verified: the (1,2) cofactor of A is +t^2 s^2), so the per-direction
  exponent carries a cross term proportional to (P_ad-P_be).(P_ad-P_cf) that
  couples the electron-2 and electron-3 centres THROUGH electron 1. The be-sum
  and cf-sum therefore do not separate by reordering; the O(n^4) form must expand
  that cross-coupling in electron 1's Hermite basis (the v2_{ad,nu} =
  sum_be D_be <Lambda_nu^ad|e^{-t^2 r^2}|be> field build, then a 3-Hermite
  overlap contraction), and must reproduce te_core including the moment=1,2
  (r12^2, r12.r13) cases. This is effectively a new 3-electron engine, oracle-
  gated against test_threeel to machine precision -- a research-grade dedicated
  pass (start with a standalone field-fold prototype vs three_electron_energy
  before touching te_core), NOT a mechanical refactor. Highest-value item. A
  cheaper safe partial: cache te_core's exponent-triple+node quantities
  (te_inv3, central moments, prefac) shared across sextets (keeps O(n^6)).
- **rigrad.hpp RI-K Hessian -- O(N^5) loops that are textbook GEMMs**
  (798-807 response H=R^T S; 787-795 S=M^-1 R; plus the H/G/c3/c2 intermediate
  families in RI-K gradient 535-573 and Hessian 650-710, and RI-J Hessian
  446-463). Rectangular GEMMs over O(N^4) tensors. Needs the rectangular gemm_nn
  wrapper; keep the triple-loop fallback for extended precision.
- **ri.hpp RI J/K + fit as hand triple-loops** (ri_jk K-build O(naux nao^3)
  100-119; ri_fit B=T Mhalf O(nao^2 naux^2) 66-72; Mhalf reconstruction 53-58;
  ri_k_occ 137-153). All GEMMs; gather the strided B^P slice contiguous first.
- **Naive O(ns^4) shell-quartet builds with no screening/symmetry**: DONE
  (erigrad.hpp two_electron_gradient, erihess.hpp two_electron_hessian,
  giao2e.hpp giao_jk_dB). Exact 8-fold permutational symmetry: loop canonical
  quartets only (a>=b, c>=d, pair(ab)>=pair(cd)), evaluate the promoted/demoted
  integral blocks ONCE, and replay the *unchanged* per-quartet contraction on
  every distinct orbit member by permuting the block axes (detail::permute_block
  + eri_perms). Correctness of the sign/centre bookkeeping rides entirely on the
  original body -- it recomputes everything from each member's own shell data;
  only the permutation-invariant integral values are shared, so the GIAO phase
  sign flips under a<->b/c<->d/(ab)<->(cd) are handled automatically. erihess's
  per-quartet std::map became a canonical block cache shared across the orbit.
  Optional Schwarz+density screening via a tau param (default 0 = exact; the
  Schwarz bound carries a +1 (grad) / +2 (hess) promotion margin so it bounds
  the derivative integrals). Validated bit-exact-to-rounding vs the pre-symmetry
  reference on an s/p/d/mixed system (grad 1e-12, hess 2e-14, giao 8e-16 rel);
  measured 5.1x on a 12-shell s/p/d set (approaching the 8x ceiling, gap =
  permute-copy overhead + smaller diagonal orbits). Screening-consistency tests
  added (tiny-tau == exact). Existing FD oracles + full suite green (194/194).
- **kbuild.hpp exchange full 8-fold** DONE (opt-in behind a shared symmetry
  flag). The deterministic per-output-block build (K_ab=K_ba 2x, no atomics,
  byte-for-byte MPI-reproducible) stays the DEFAULT. New Symmetry::Full
  (symmetry.hpp -- a library-wide flag; see the cross-builder audit below for
  why it stays K-only in practice)
  evaluates each unique quartet (bra pair a<=c, ket pair b<=d, bra idx >= ket
  idx) ONCE: it accumulates the four symmetry-distinct density contractions
  (Acc1 K_ab<-D_cd, Acc2 K_cb<-D_ad, Acc3 K_ad<-D_cb, Acc4 K_cd<-D_ab) over the
  t-nodes, then atomic-scatters the deduplicated 8-element orbit into K. The
  dedup (drop ops whose shell 4-tuple already appeared) makes the coincidence
  bookkeeping (a==c, b==d, braPair==ketPair) automatic and provably complete.
  Measured 4.08x on a 24-shell/66-AO s/p/d set; matches the deterministic build
  to 2.5e-14 (rounding, since atomic order is nondeterministic). Cost: a Full
  build is not bit-reproducible and the MPI Full build is numerically- not
  byte-reproducible -- hence opt-in, default preserves all guarantees. Validated
  vs exact dense K (1e-11) and vs the deterministic build (1e-11), screened and
  unscreened. GPU caveat: the four KNC*KNC accumulators + g tables are a large
  per-work-item local footprint (fine on CPU/OpenMP; may hurt GPU occupancy).
- **Symmetry across all builders -- audit (2026-09-05).** Question: should the
  atomic-scatter Symmetry::Full apply to every builder? Finding: NO -- it is a
  K-specific win. It pays only where an expensive, UN-folded integral evaluation
  is duplicated across output blocks (K). Builder by builder:
  * K (exchange): the one true beneficiary; atomic 8-fold above (~4x). DONE.
  * J (coulomb_build) and tc (tc_gradu_grad_build): density is PRE-FOLDED into
    Hermite space, so the pair-pair symmetry (p|q)=(q|p) would save only the
    cheap hermite_b setup, not the dominant density-folded contraction (each
    direction folds a different density d^q vs d^p) -- ~1.1-1.3x for a loss of
    determinism. NOT worth it; leave deterministic.
  * coulomb_2c (P|Q) and coulomb_3c (mu nu|P): had an unexploited 2x, but it is
    a plain output-block mirror (transpose / bra-swap) -- free and DETERMINISTIC,
    no atomics needed. DONE unconditionally (a>=b / m>=n triangle + mirror),
    bit-exact, validated vs the PySCF int2c2e/int3c2e oracles. No flag: the
    Symmetry flag is reserved for the determinism-trading atomic case (K).
  * oneel/nuclear (1e), threeel_ri (P<->Q 2x), erigrad/erihess/giao2e (8-fold):
    already exploit their full symmetry, deterministically.
  Conclusion: the flag stays K-only; every other builder either already has its
  symmetry or gains it deterministically (2c/3c) -- "Symmetry::Full everywhere"
  would be no-ops or determinism-losing regressions elsewhere.
- **GPU-memory tiling -- graceful degradation when the problem exceeds device
  memory (2026-09-05).** GPUs are the main target, so builds must tile when the
  working set does not fit. Fock builds are LINEAR, so an output tile is exact:
  K = sum of output tiles, no approximation. This is the capacity axis; it is
  ORTHOGONAL to the occupancy caveat (per-work-item local footprint of the
  Sym8/deterministic kernels), which needs a team-scratch refactor and real GPU
  hardware to tune -- tiling does not fix occupancy and vice versa.
  * **exchange_build_tiled** DONE: computes K one output shell-row tile at a time
    (tile_shells output shells/tile), bounding the device K allocation to
    (tile rows) x nao instead of nao^2. Reuses the deterministic kernel body via
    an output-range + tiled-output parameter (untiled path byte-for-byte
    unchanged); the tiled path drops the K_ab=K_ba mirror for bounded,
    tile-invariant memory. Correctness = tile-invariance: the result is
    BIT-IDENTICAL across tile sizes (each element computed independently), tested
    for tile_shells in {1,2,3,5}, and equals the untiled build to rounding. This
    is the reusable pattern + the CPU-validatable proof methodology.
  * **coulomb_build (J)** DONE (coulomb_build_tiled): loops bra shell-pair tiles
    (pair_tile pairs/tile) for phases 2/3 at tile-local Hermite/product offsets,
    dropping the per-pair j^p/J^p arrays to the tile footprint; phase-1 d^q (full
    ket Hermite density) is rebuilt per tile (cheap) and D stays resident. Because
    each output pair's j^p is the full ket sum regardless of tiling, the result is
    BIT-EXACT vs coulomb_build and across pair_tile {1,2,3,7,1000}
    (Fock.CoulombTilingIsBitExact). No far-field in the tiled path.
  * **tc_gradu_grad_build** -- N/A in current form: it is a HOST build
    (DefaultHostExecutionSpace) whose per-bra-pair field j^{pr} is already
    LOCAL scratch (allocated inside the bra lambda), not a global O(npair) array.
    Its only full arrays are the ket Hermite densities `kets` (input) and F
    (output) -- neither reduced by bra-loop tiling. So there is nothing to tile
    for GPU memory until tc is ported to a device kernel, at which point the
    tiling axis is KET-STREAMING (accumulate F over ket-density tiles, since F is
    linear in the ket sum), not the bra tiling J/K use. Deferred to the tc GPU
    port; forcing a bra-tile now would save no memory.
  * **RI-J via the 3-index tensor (mu nu|P), nao^2 x naux** DONE (ri_j_tiled):
    the largest object and the real "too big for memory" case. RI-J is a sum over
    the auxiliary index, so it factors into two aux passes -- gamma_Q = sum_munu
    (munu|Q) D, then J = sum_P (munu|P) (M^{-1} gamma)_P -- each needing only ONE
    3-center block live (new coulomb_3c_auxblock builds (munu|P) for an aux shell
    range). Peak 3-center memory nao^2 x (tile aux AOs) instead of nao^2 x naux;
    never materialises the full tensor or the fit vectors B. Matches the untiled
    ri_jk J and is tile-size-independent to rounding (the per-tile GEMM regroups
    the P-sum, so not bit-exact -- unlike the exchange row-tiling where each
    element is fully computed in one tile). Validated (RI.TiledJMatchesUntiled).
  * **RI-K tiling** -- the harder follow-on: RI-K collapses to a single P-sum
    only through the fitted B = T M^{-1/2}, whose M^{-1/2} mixes ALL auxiliaries,
    so a per-P-block build still needs the full 3-center tensor. Tiling it needs
    either full B resident, a two-pass/disk scheme, or the O(naux^2) double-aux
    form K = sum_QR M^{-1}_QR (T^Q D T^R) (block-pair recompute). Deferred.
  * Only the output/K axis is tiled so far; D (density) and the pair E-tables are
    still resident. Density streaming + E-table tiling are the further axes for
    truly out-of-core problems.
- **cholesky.hpp:256-260 rank update as scalar AXPYs** (O(naux^2 nprod)) -- really
  a GEMV/GEMM against the accumulated L block; float/double dispatch, scalar
  fallback for extended precision.
- **cdjk.hpp CD/RI exchange density low rank** DONE (new overload, zero risk to
  the generic path): cholesky_jk_occ(basis, cb, C, nocc, occ_scale, J, K) takes
  the occupied orbitals (D = occ_scale C C^T) and builds K = occ_scale sum_J
  (L^J C)(L^J C)^T via two rectangular gemms per aux vector -> O(naux nao^2
  nocc) instead of O(naux nao^3) (nao/nocc x on K at scale). J unchanged. The
  generic cholesky_jk is untouched; validated bit-close (1e-11) to it on
  D=occ_scale C C^T incl. null-output branches. The pre-existing dense entry
  point stays the fallback for non-low-rank (e.g. energy-weighted) densities.
- **sto.hpp:185-197/410-422 contract_to_sto / expand_density_to_prim** -- dense
  C*M*C^T as triple loops in the STO J/K hot path; dispatch to gemm_nn AND exploit
  that C has exactly ns nonzeros per row (O(na ns np) not O(na np^2)).
- **localhybrid.hpp:106-122 local_exchange** carries a needless occupied index:
  eps(r) = -1/2 u^T V u with u_mu = sum_i C_mi psi_i collapses the VC (nao^2 nocc)
  and Kij (nocc^2 nao) loops to O(nao^2); also pass a screening tau to the
  per-point V collocation build (currently 0).
- **geohess.hpp nuclear_attraction_hessian** DONE: was rebuilding the full
  quadrature 18x per charge (2 nuclear_geoderiv calls x 9 (e,f)). New
  detail::nuclear_geoderiv_multi builds the e_coeffs + per-t g-tables ONCE at
  the maximum elevation (bra+2, ket+1) per charge and extracts all 18 (na,nb)
  derivatives by sub-table + apply_shifts (e_coeffs/hermite_b values for a fixed
  (i,j,tau) are identical at any higher elevation, so the extraction is exact).
  Bit-exact (0.0 diff) vs the original per-(e,f) assembly incl. a d shell;
  measured 2.56x on an s/p/d/mixed 4-atom set (the e_coeffs+g-build is hoisted;
  the per-request accumulate is unchanged, so the win grows with l/size). FD
  oracle + full suite green (196/196).
- **threeel_ri.hpp** aux loops: full permutation symmetry of the ghost-partner 3e
  aux integral (~6x on the O(naux^3) energy/Fock loops); the O(naux nao^4)
  effective-exchange term is intrinsically expensive (screening only).
- **oneel.hpp ns^2 pair sweeps** DONE: overlap_matrix/kinetic_matrix/
  multipole_matrices take an optional tau; each pair is skipped when its
  Gaussian prefactor exp(-mu R_AB^2) (detail::pair_gauss_prefactor) <= tau. At
  tau=0 this drops only the exactly-underflowed (zero) blocks -- exact, and
  asymptotically linear for large systems where distant pairs underflow; tau>0
  is an approximate distance screen. Bra-ket (anti)symmetry was already done.
  Screening-consistency test added (distant tight pair dropped, within tau).
  (multipole.hpp far-field is a separate item under M-MP; per-axis O(L^4)
  factoring of its jbuild contraction remains a minor open item.)
- Minor: nuclear.hpp/localhybrid per-point E-coefficient rebuild (batch points
  into one attraction call); c2s.hpp c2s_matrix rebuilt per apply (cache per l);
  batch.hpp tail bcoef/invf recomputed per output component; jbuild far-field
  O(L^6) multipole contraction could factor per-axis to O(L^4).

DONE under this milestone (all CI-green):
- blas.hpp: precision-generic transpose-capable row-major GEMM (BLAS for
  float/double, triple-loop fallback for long double/__float128).
- item 2 (RI/GEMM) COMPLETE: ri.hpp (J/K/fit), rigrad.hpp (O(N^5) RI-K Hessian
  response + S=M^-1 R + all H/G/coeff intermediates + RI-J), sto.hpp
  (sparsity-aware contract, bit-identical, O(na ns np)), cholesky.hpp
  (rank-update GEMV).
- item 3 (mostly done): kbuild exchange 2x (upper-triangle + mirror); threeel_ri
  P<->Q 2x on both O(naux^3) aux loops; screening.hpp MBIE-1/QQR distance-
  including estimate (min(Schwarz, monopole/R), validated valid-upper-bound +
  tighter-than-Schwarz vs eri_quartet).
- item 4 (mostly done): localhybrid occupied-index collapse (eps=-1/2 u^T V u);
  oneel 2x symmetry across overlap/kinetic/multipole (symmetric) and angular
  momentum (antisymmetric).
- tc_gradu_grad_build rewritten from an O(N^4) quartet loop to a density-folded
  geminal J-build (host-parallel over bra pairs, Schwarz-style screening).
- CI fix: device.hpp View(std::string label,...) for Kokkos 4.5.01.

REMAINING (dedicated/careful -- the hard/subtle items): item 1 threeel
O(n^6)->O(n^4) nested-J-build (a genuine re-derivation of te_core into the field
form); item 3 erigrad/erihess/giao2e 8-fold quartet permutational symmetry +
Schwarz/density screening on the O(ns^4) derivative quartet loops (subtle sign/
index bookkeeping in hot routines; screening.hpp provides the estimate); item 4
geohess nuclear_attraction_hessian 18x-per-charge (needs a nuclear_geoderiv
elevated-l block variant returning all (e,f) components), minor caching
(c2s_matrix per l, batch tail bcoef -- device-lambda hoist).
