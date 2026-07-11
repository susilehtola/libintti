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

on a numerical quadrature grid in t (composite linear + logarithmic panels).
At fixed t the kernel factorizes over Cartesian directions, so a primitive
Cartesian GTO quartet reduces to products of analytic 1D integrals:

```
(ab|cd) = Σᵢ wᵢ · Ix(tᵢ)·Iy(tᵢ)·Iz(tᵢ)  +  (π/t_c²)·S_abcd
```

The grid is truncated at t = t_c and the tail is rectified with the
delta-function trick of Losilla et al.: exp(−t²r²) → (π^{3/2}/t³) δ³(r), so
the tail contributes π/t_c² times the four-orbital overlap S_abcd. This turns
the O(1/t_c²) truncation error into O(1/t_c⁴).

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

Milestone 1: primitive Cartesian ERI quartets (`intti::eri_quartet`) for all
four kernels, validated against analytic McMurchie–Davidson integrals.

## License

MPL-2.0.
