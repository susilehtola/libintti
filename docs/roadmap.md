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
  (test_helmholtz); (2) the
  M(kappa) t-tensor build reusing the ExpSum Yukawa grid; (3) a Helmholtz-SCF
  proof-of-concept on H / He / H2 confirming the fixed point converges to the
  in-basis Galerkin/diagonalization energy, then measuring whether kinetic-
  exactness buys accuracy per basis function. Oracle: closed-form Yukawa
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
     decontract -> recontract. Needed for efficiency on real def2/ANO basis sets
     (we have no contraction strategy yet).

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
  the central moments cm(k1,k2,k3) couple all three electrons (Ai is dense even
  though A[1][2]=0), so it is a genuine re-derivation into the nested-J-build
  form, not a mechanical refactor -- dedicated work, oracle-gated (test_threeel).
  The single highest-value change; also caches te_core's exponent-triple+node
  quantities (te_inv3, central moments, prefac) shared across sextets.
- **rigrad.hpp RI-K Hessian -- O(N^5) loops that are textbook GEMMs**
  (798-807 response H=R^T S; 787-795 S=M^-1 R; plus the H/G/c3/c2 intermediate
  families in RI-K gradient 535-573 and Hessian 650-710, and RI-J Hessian
  446-463). Rectangular GEMMs over O(N^4) tensors. Needs the rectangular gemm_nn
  wrapper; keep the triple-loop fallback for extended precision.
- **ri.hpp RI J/K + fit as hand triple-loops** (ri_jk K-build O(naux nao^3)
  100-119; ri_fit B=T Mhalf O(nao^2 naux^2) 66-72; Mhalf reconstruction 53-58;
  ri_k_occ 137-153). All GEMMs; gather the strided B^P slice contiguous first.
- **Naive O(ns^4) shell-quartet builds with no screening/symmetry**:
  erihess.hpp:42-178 (two_electron_hessian), erigrad.hpp:64-149
  (two_electron_gradient), giao2e.hpp:66-134 (giao_jk_dB). Add Schwarz + density
  screening and 8-fold permutational symmetry (~8x + screened reduction; GIAO
  needs sign care under the symmetry map; erihess also has a per-quartet std::map
  cache to replace with a stack array).
- **kbuild.hpp:104-229 exchange -- no permutational symmetry.** K_ab=K_ba gives a
  free 2x (compute a<=b, mirror); full (ac|bd) 8-fold is ~8x but needs atomics
  and a reworked MPI split. Hottest kernel, so even 2x matters.
- **cholesky.hpp:256-260 rank update as scalar AXPYs** (O(naux^2 nprod)) -- really
  a GEMV/GEMM against the accumulated L block; float/double dispatch, scalar
  fallback for extended precision.
- **cdjk.hpp:97-102 CD/RI exchange ignores density low rank.** Factor D=CC^T
  (rank nocc) -> K build O(naux nao^3) -> O(naux nao^2 nocc); needs a PSD/low-rank
  guard + fallback.
- **sto.hpp:185-197/410-422 contract_to_sto / expand_density_to_prim** -- dense
  C*M*C^T as triple loops in the STO J/K hot path; dispatch to gemm_nn AND exploit
  that C has exactly ns nonzeros per row (O(na ns np) not O(na np^2)).
- **localhybrid.hpp:106-122 local_exchange** carries a needless occupied index:
  eps(r) = -1/2 u^T V u with u_mu = sum_i C_mi psi_i collapses the VC (nao^2 nocc)
  and Kij (nocc^2 nao) loops to O(nao^2); also pass a screening tau to the
  per-point V collocation build (currently 0).
- **geohess.hpp:131-176 nuclear_attraction_hessian** rebuilds the full quadrature
  18x per charge (2 calls x 9 (e,f)); materialise one elevated-l geoderiv block
  per charge and extract all 9 components (~18x).
- **threeel_ri.hpp** aux loops: full permutation symmetry of the ghost-partner 3e
  aux integral (~6x on the O(naux^3) energy/Fock loops); the O(naux nao^4)
  effective-exchange term is intrinsically expensive (screening only).
- **oneel.hpp / multipole.hpp** ns^2 pair sweeps: exploit the Gaussian prefactor
  K=exp(-mu R_AB^2) underflow screen (-> asymptotically linear) and S/T/multipole
  bra-ket (anti)symmetry (~2x).
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
