# libintti

Efficient quadrature code for tensor expressions in quantum chemistry.

(*intti* is Finnish slang for the army; the integral is what the author did
instead of non-military service, *sivari* being the name of [Dage Sundholm's
group's](https://www.helsinki.fi/en/researchgroups/sundholm-group) library.)

## Method

libintti evaluates two-electron interactions through the Gaussian resolution
of the Coulomb operator

```
1/r₁₂ = (2/√π) ∫₀^∞ exp(−t² r₁₂²) dt
```

on a numerical quadrature grid in t. At fixed t the kernel factorizes over
Cartesian directions, so a primitive Cartesian GTO quartet reduces to
products of analytic 1D integrals:

```
(ab|cd) = Σᵢ wᵢ · Ix(tᵢ)·Iy(tᵢ)·Iz(tᵢ)  [ +  (π/t_c²)·S_abcd ]
```

Two grid mappings are provided:

- **Möbius (default):** t = s·u/(1−u) with Gauss–Legendre nodes u ∈ (0,1).
  The ERI integrand's effective exponent saturates at large t and the
  integrand decays algebraically (∝ t⁻³), which the rational map integrates
  natively over the full range [0,∞) — no truncation, no tail correction.
  Converges to the working-precision floor of the scalar type by N ≈ 48–64
  nodes for exponents spanning [10⁻², 10⁶].
- **Linear+logarithmic panels** (Jusélius & Sundholm 2007), truncated at
  t = t_c with the delta-function tail correction of Losilla et al.:
  exp(−t²r²) → (π^{3/2}/t³) δ³(r), so the tail contributes π/t_c² times the
  four-orbital overlap S_abcd, turning the O(1/t_c²) truncation error into
  O(1/t_c⁴). Use this when t_c is imposed externally, e.g. by the spatial
  resolution of a real-space grid (the future FEM path).

Supported kernels: Coulomb 1/r, range-separated erf(ωr)/r and erfc(ωr)/r,
and Yukawa exp(−κr)/r — all as reweightings/limits of the same t grid.

Since GTOs have exact analytic 1D integrals at each t, quadrature error comes
only from the t grid; the same machinery extends to any orbital representable
on a tailored tensor grid (numerical atomic orbitals, diatomic orbitals,
finite elements), which is the longer-term goal of the library, together
with J/K matrix builds, local exchange energy densities for local hybrid
functionals, and a thread-safe libcint-compatible interface.

### Key references

- J. Jusélius and D. Sundholm, *Parallel implementation of a direct method
  for calculating electrostatic potentials*, J. Chem. Phys. **126**, 094101
  (2007). [doi:10.1063/1.2436880](https://doi.org/10.1063/1.2436880)
- S. A. Losilla, M. M. Mehine, and D. Sundholm, *Construction of the
  two-electron contribution to the Fock matrix by numerical integration*,
  Mol. Phys. **110**, 2569 (2012).
  [doi:10.1080/00268976.2012.720725](https://doi.org/10.1080/00268976.2012.720725)
- S. A. Losilla, M. A. Watson, A. Aspuru-Guzik, and D. Sundholm,
  *Construction of the Fock matrix on a grid-based molecular orbital basis
  using GPGPUs*, J. Chem. Theory Comput. **11**, 2053 (2015).
  [doi:10.1021/ct501128u](https://doi.org/10.1021/ct501128u)

## Arbitrary precision

The core is header-only and templated on the scalar type. Builtin
floating-point types (`float`, `double`, `long double`) execute through
Kokkos parallel patterns; class-type scalars with the standard math
functions available via ADL (e.g. MPFR C++ wrappers, Boost.Multiprecision)
automatically use a serial host path. Quadrature nodes, weights, and all
intermediates are generated in the requested precision.

## Building

Requires CMake ≥ 3.23, a C++20 compiler, and [Kokkos](https://kokkos.org)
(found via `find_package`, fetched automatically otherwise).

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

Options: `INTTI_BUILD_TESTS` (default `ON`), `INTTI_ENABLE_MPI` (default
`OFF`; placeholder for later milestones).

## Status

- **M1:** primitive Cartesian ERI quartets (`intti::eri_quartet`) for all
  four kernels, templated on the scalar type, validated against analytic
  McMurchie–Davidson integrals (in double and long double precision).
- **M2:** batched quartet driver (`intti::eri_quartets` over a `PairTable`
  with precomputed per-pair Hermite expansion tables and a reusable,
  allocation-free workspace) and a two-step pivoted Cholesky decomposition
  of the ERI matrix (`intti::two_step_cholesky`: pivot selection followed by
  RI-style vector construction; requires LAPACK). The prolate-spheroidal
  representation of orbital products — the bridge to integrals between
  atomic, diatomic, and 3D basis sets — is validated in
  `prototype/psc_validation.py` and documented in `docs/psc.md`; the C++
  interfaces are fixed in `include/intti/product.hpp` for M3.

## License

MPL-2.0.
