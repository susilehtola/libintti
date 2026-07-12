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
real solid harmonics in m = −l..+l order. libcint's `int2e_sph` uses the
same m order with per-row scale factors; the mapping is a diagonal rescale
to be pinned when the façade lands (M7). Rotational behavior is identical
(both row sets are orthogonal under Wigner rotations).

## Contraction

libintti evaluates contracted AOs either through
`eri_quartets_accumulate` (include/intti/batch.hpp) with coefficient
products, or by tabulating contracted products on grids
(include/intti/product.hpp). PySCF contraction coefficients are normalized
per contracted function; the façade must fold PySCF's `bas`/`env`
coefficient conventions (already gto_norm-scaled) directly into the
accumulation weights.
