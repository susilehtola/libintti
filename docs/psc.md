# Prolate-spheroidal representation of orbital products

Validated by `prototype/psc_validation.py` (run 2026-07-12); this is the
mathematical basis for M3's cross-representation integrals and the
prolate-spheroidal Cholesky fit functions.

## Representation

A two-center orbital product φ_a(r−A)φ_b(r−B) is written in prolate
spheroidal coordinates (ξ, η, φ) with foci at A, B:
r_A = (R/2)(ξ+η), r_B = (R/2)(ξ−η), dV = (R/2)³(ξ²−η²) dξ dη dφ.
Its azimuthal content about the interfocal axis is **exactly** finite:
χ = Σ_m f_m(ξ,η) e^{imφ} with |m| ≤ l_a+l_b, extracted exactly by FFT over
2(l_a+l_b)+1 uniform φ points. f_m(ξ,η) lives on a Gauss–Legendre grid
(η ∈ [−1,1]; ξ ∈ [1, ξ_max] with ξ_max from the product's exponential decay).

Measured (s and p products, R = 1.4):

| grid | ss charge rel. err |
|------|--------------------|
| 24×24 | 1.3×10⁻⁶ |
| 32×32 | 5.1×10⁻¹⁰ |
| 48×48 | 4.6×10⁻¹⁴ |

p×p products behave identically (4.5×10⁻¹⁵ at 48×48); the m-truncation is
exact to machine zero (max |f_m|, m≠0, of a pz·s product: 0.0).

## Kernel structure at fixed t

For **coaxial** products (shared interfocal axis) the φ integrals of the
Gaussian kernel are diagonal in m with modified Bessel factors:

∫∫ dφ₁ dφ₂ e^{im₁φ₁+im₂φ₂} e^{2t²ρ₁ρ₂cos(φ₁−φ₂)} = (2π)² δ_{m₁,−m₂} I_{m₁}(2t²ρ₁ρ₂),

evaluated stably as `ive(m, 2t²ρ₁ρ₂)·exp(−t²(ρ₁−ρ₂)²)` (scaled Bessel). Each
(t, m) contribution is a 2D-grid × 2D-grid bilinear form — a GEMM with a
t-dependent kernel matrix shared by *all* products on the same grid pair.

**Grid-represented products cannot resolve the kernel at arbitrarily large
t** (the kernel width 1/t falls below the grid spacing), so they use the
truncated LinLog t grid with the delta-function tail correction
(tail = π/t_c² · ⟨χ₁|χ₂⟩); the untruncated Möbius map is reserved for
analytic (GTO) representations. This was confirmed empirically: with
t_c = 30 and no tail the coaxial error was non-monotonic in the grid size;
with t_c = 20 matched to the grid resolution plus the analytic tail it
converges monotonically:

| (ξ,η) grid | coaxial (ss\|ss) rel. err |
|------------|---------------------------|
| 16×16 | 1.2×10⁻³ |
| 24×24 | 2.6×10⁻⁶ |
| 32×32 | 8.1×10⁻⁷ |

(the 8×10⁻⁷ floor is the leading-order-only tail correction and the modest
t grid of the prototype, not the representation).

## General geometry

Non-coaxial frames couple the m channels. The prototype validates the
uniform-φ point-cloud formulation (each product expanded onto N_φ azimuthal
points in its own frame; the kernel is then a plain Gaussian between the two
clouds — still per-t GEMM-shaped, with the t-independent distance matrix
precomputed once per frame pair):

| N_φ | non-coaxial (ss\|ss) rel. err |
|-----|-------------------------------|
| 4 | 3.1×10⁻³ |
| 6 | 6.3×10⁻⁴ |
| 8 | 1.1×10⁻³ |
| 12 | 1.2×10⁻³ |

N_φ = 6 already saturates for s products — the residual ~10⁻³ is the small
16×16 prototype grids, not the φ quadrature (superexponentially convergent
for these periodic analytic integrands).

## Mixed representations

An analytic GTO bra against a PSC-gridded ket needs only the analytic
per-grid-point kernel integral
∫e^{−p(r−P)²}e^{−t²(r−r₂)²}dr = (π/(p+t²))^{3/2} e^{−pt²/(p+t²)|P−r₂|²};
validated at 5.8×10⁻⁴ with the same prototype grid sizes. This is the
template for every GTO ↔ diatomic ↔ 3D-grid cross integral: the bra side is
evaluated at whatever points (or Hermite expansions) the ket representation
uses, at each t node.

## Consequences for M3

- One code path (`intti::interaction` in include/intti/product.hpp) covers
  all representation pairings; each pairing reduces to per-t GEMMs.
- Coaxial pairs (the dominant case for diatomics and for pair products
  sharing an atom pair) get the cheap m-diagonal Bessel path.
- Grid representations always pair with LinLog + delta tail; t_c is set by
  the spatial resolution of the coarser grid.
- Production accuracy requires larger (ξ,η) grids than the prototype's
  (48×48 reaches 10⁻¹⁴ in the representation itself) and the next-order
  tail correction if t_c must stay small.
