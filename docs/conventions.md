# libintti ↔ libcint/PySCF conventions

Pinned numerically by `prototype/pyscf_validation.py` (PySCF 2.13,
uncontracted s/p/f + s/d two-center system, full `int2e_cart` tensor,
max relative deviation **2.3×10⁻¹⁴**). These are the facts the M7
libcint-compatible façade must implement.

## Component ordering

libintti's Cartesian component ordering (`intti::cart_comp`,
include/intti/gto.hpp: lx descending, then ly descending — xx, xy, xz, yy,
yz, zz) **is** libcint's ordering. AO order is shell-sequential. No
permutation needed.

## Cartesian normalization (`int2e_cart`, `cart=True`)

libintti primitives are unnormalized. PySCF/libcint cart AOs correspond to
(`intti::cart_norm_pyscf`, include/intti/normalization.hpp):

- l ≤ 1: unit-normalized per component,
  N = √[(2α/π)^{3/2} (4α)^l / ((2lx−1)!!(2ly−1)!!(2lz−1)!!)];
- l ≥ 2: a **common factor for all components** of the shell — the axial
  (x^l) normalization times √(4π/(2l+1)):
  N = √[(2α/π)^{3/2} (4α)^l / (2l−1)!!] · √(4π/(2l+1)).
  Verified at l = 2 (the deviation pattern was exactly 5/(4π) on the
  diagonal) and l = 3 (f shell in the harness).

Integrals therefore map as
(ab|cd)_pyscf = N_a N_b N_c N_d (ab|cd)_intti(unnormalized).

## ERI values

s/p blocks agreed to machine precision *before* any normalization work —
grid (default Möbius, N = 64), ordering, and the quadrature itself are
exact against libcint's analytic integrals at the 10⁻¹⁴ level.

## Spherical harmonics

libintti's `c2s_matrix` (include/intti/c2s.hpp) rows are sphere-orthonormal
real solid harmonics in m = −l..+l order. Pinned by the M7 façade harness
(`prototype/pyscf_validation.py --facade`, `intti::sph_rescale`,
include/intti/normalization.hpp) against PySCF 2.13's `int2e_sph`, to
1.1×10⁻¹⁵ relative deviation for a d/f two-center system:

- **l ≤ 1**: libcint's spherical AOs are *identical* to the
  `cart_norm_pyscf`-normalized Cartesian AOs — same functions, same order
  (x, y, z for p), bit-for-bit (checked to 0.0 absolute deviation against
  PySCF for an s+p shell). No `c2s_matrix` row corresponds to libcint's
  ordering here (`c2s_matrix(1)`'s m = −1, 0, +1 rows are y, z, x — a
  *permutation*, not a diagonal rescale, of libcint's x, y, z), so the
  façade bypasses `c2s_matrix` for these shells and copies the Cartesian
  block directly.
- **l ≥ 2**: the row order already matches m = −l..+l (PySCF's `dxy, dyz,
  dz^2, dxz, dx2-y2` for d and `f-3..f+3` for f are exactly `c2s_matrix`'s
  row order), and every row of every shell shares one l-independent
  constant, **1/√(4π) = 0.28209479177387814**. I.e.
  `sph_block = c2s_matrix(l) · cart_block(cart_norm_pyscf) / sqrt(4 pi)`.

`intti::sph_rescale<Real>(l)` (include/intti/normalization.hpp) returns
this constant (1 for l ≤ 1, unused there; 1/√(4π) for l ≥ 2).

## Contraction

libintti evaluates contracted AOs either through
`eri_quartets_accumulate` (include/intti/batch.hpp) with coefficient
products, or by tabulating contracted products on grids
(include/intti/product.hpp). PySCF contraction coefficients are normalized
per contracted function; the façade folds PySCF's `bas`/`env` coefficient
conventions directly into the accumulation weights.

Pinned empirically (M7, `prototype/pyscf_validation.py --facade`, cc-pVDZ
oxygen s-shell: 8 primitives, 2 contracted functions, and synthetic
multi-primitive d/f shells) against PySCF 2.13's `mol._env`: PySCF's
`env[ptr_coeff + c*nprim + p]` coefficients are already scaled by PySCF's
*own* internal primitive normalization, `pyscf.gto.mole.gto_norm(l,
alpha)`, which equals `intti::cart_norm_component(l, 0, 0, alpha) *
sqrt(4 pi / (2l+1))` for every l (verified l = 0..4) — i.e.
`intti::cart_norm_pyscf(l, alpha)` exactly for l ≥ 2, but carrying an
*extra* l-independent factor `sqrt(4 pi / (2l+1))` for l ≤ 1 that
`cart_norm_pyscf` does not include. The façade therefore uses

```
weight(prim p, contracted fn c) = env_coeff(p, c) * coeff_rescale(l)
coeff_rescale(l) = sqrt((2l+1) / (4 pi))   for l <= 1
                  = 1                       for l >= 2
```

as the coefficient in the primitive-quartet accumulation; no further
per-primitive `cart_norm_pyscf` multiplication is needed; it is already
implied. This was confirmed by reconstructing PySCF's `int1e_ovlp_cart`
diagonal (both axial and off-axial Cartesian components) from raw `env`
coefficients for contracted d and f shells, matching to double precision.

## Complex scalars: the ERI convention vs the pair-density Gram matrix

The two-electron integral is

```
(ab|cd) = ∫∫ ω_a*(1) ω_b(1)  (1/r₁₂)  ω_c*(2) ω_d(2)
```

— the **first index of each pair is the conjugated one** (`make_giao_pair`,
include/intti/giao.hpp). This is the standard complex-orbital chemist
convention and is what every Fock-like build uses (`coulomb_build`,
`exchange_build`, `giao_jk`). It is consistent across the API.

What is **not** the same object is the Coulomb-metric **Gram matrix** of the
pair densities ρ_P = ω_a* ω_b. Because ρ_P* = ρ_(ba),

```
⟨ρ_P|ρ_Q⟩ = (ba|cd)      -- the BRA PAIR SWAPPED
```

For **real** orbitals ρ_P* = ρ_P and the two coincide, which is why the
real code may freely use `(P|Q)` as a metric. For **complex** (London/GIAO)
pairs they differ, and the distinction is not cosmetic:

- `(mn|ls)` is complex **symmetric** (`M_PQ = M_QP`, by electron exchange)
  but **not Hermitian** — so it admits **no Cholesky decomposition at all**.
- `(nm|ls)` is Hermitian positive definite with a real positive diagonal —
  this is what a finite-B Cholesky/RI metric must decompose.

Both facts are pinned numerically at B ≠ 0 by
`GIAO.FiniteFieldPairGramIsHermitianPositive` (tests/test_giao.cpp).

Consequences enforced in code (all no-ops for the real path):

- `schwarz()` (fock.hpp) forms `Q_P = sqrt(max (P|P))`, identifying `(P|P)`
  with `⟨ρ_P|ρ_P⟩`; it `static_assert`s `!is_complex_v`.
- `two_step_cholesky()` / `pivoted_cholesky()` (cholesky.hpp) decompose
  `(P|Q)` as that Gram matrix; they `static_assert` `!is_complex_v`.

These guards matter because `Kokkos::complex` is now a `kokkos_scalar_v`
(the finite-field device path), so `PairTable<Kokkos::complex<double>>` is
legal and these routines would otherwise instantiate and be **silently
wrong** rather than failing to build.

Exchange has the same trap at the Fock level: at finite B the exchange must
be `K_mn = Σ_ls (ml|sn) D_ls`, **not** `(ml|ns)`. The two coincide for real
ERIs, but only `(ml|sn)` is Hermitian once the London phases make the
integrals complex (`giao_jk`, giao2e.hpp).

## Derivatives: analytic shift, never numerical, never from the interpolant

All derivative integrals use the exact McMurchie-Davidson centre shift

```
d/dA_x : F(i) -> i F(i-1) - 2a F(i+1)
```

on the Gaussian factor, so a derivative is an exact linear combination of
shifted-angular-momentum integrals from the same engine. Because the t grid is
perturbation-independent, differentiation commutes with the quadrature, and
arbitrary-order mixed derivatives are just repeated shifts (`geoderiv`). There
is **no finite differencing inside the library** -- FD appears only in tests, as
the independent oracle.

**Design rule for the FE/grid pillar.** `fegrid`/`gridri` currently take no
derivatives (they do the potential solve only). If derivatives are ever needed
in the FE representation -- gradients of grid-RI J/K, or a kinetic operator on
the grid -- evaluate the **analytic** derivative at the nodes (available via the
same MD shift) rather than differentiating the Lagrange interpolant.
Differentiating a LIP expansion loses roughly one order of accuracy per
derivative and is worst at element boundaries, which would silently cap the
precision of a pillar whose whole point is being convergent to near machine
precision.

### If derivatives are added to the grid route

The FE grid is geometry-dependent (it is built from the AO-product envelopes),
so a naive dE/dR picks up a grid-motion term

```
dE/dR = sum_g w_g df/dR  +  sum_g [ (dw_g/dR) f + w_g grad f . (dr_g/dR) ]
```

the second group being the Pulay analogue. Three rules:

1. **Freeze the grid while differentiating** (perturbation-independent grid).
   The grid-motion term then vanishes identically and dE/dR is the quadrature of
   the analytically differentiated integrand -- exactly why the t-quadrature
   derivatives work, since the t grid is perturbation-independent. It also makes
   the force the exact derivative of the energy actually computed, so forces are
   consistent with the surface being optimised on; a grid that moves with
   geometry gives energy and gradient from different discretisations (the "grid
   noise" pathology of Becke grids in DFT).

2. **Raise the resolved degree.** The hp construction resolves
   (x-c)^d exp(-a(x-c)^2) for d = 0..2*lmax; the MD shift raises the Cartesian
   degree by one per derivative, so use pdeg = 2*lmax + n_deriv. Grid size grows
   slowly with degree, so this is a refinement, not a new grid.

3. **Validate against the analytic derivative, not finite differences.** The
   exact MD-shift derivative integrals are available, so grid-derivative
   completeness is directly measurable. Converge on
   ||grad_grid - grad_analytic||, NOT on the energy: derivative precision lags
   energy precision on a given grid, so an energy-converged grid does not imply
   a derivative-converged one.
