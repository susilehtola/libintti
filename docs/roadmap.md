# libintti roadmap — M7 and beyond

Status: M1–M9 delivered; CI green on GitHub. ERIs validated against
PySCF/libcint to 2.3×10⁻¹⁴ (and an RHF driven entirely by libintti matches
PySCF to 1.4×10⁻¹³ Ha). This document specifies the remaining milestones
precisely enough for delegated implementation. **Ground rules for any implementer:**

1. Every deliverable has a stated oracle (an existing reference the result
   must match). Failing the oracle means the implementation is wrong — do
   not loosen tolerances; report the measured error and stop.
2. New math must be validated against an independent reference (analytic
   result, PySCF, or an existing libintti path) before optimization.
3. Follow the codebase idiom: header-only templates on Real, Kokkos
   patterns for builtin FP with serial class-type fallbacks, MPL-2.0 + SPDX
   headers, tests in tests/ with calibrated (not guessed) tolerances.

## M7 — libcint-compatible C façade  [DONE — commit 041faa2]

Delivered and validated beyond the stated oracle (unseen bases cc-pVTZ /
6-31G* / def2-SVP at 1e-14, and an RHF driven entirely by libintti ERIs
matching PySCF to 1.4e-13 Ha). Original spec retained below for reference.

### Original spec

Goal: `int2e_cart`/`int2e_sph`-compatible entry points so libintti is a
drop-in ERI backend for PySCF-style codes.

- `include/intti/cint.h` + `src` C ABI: decode libcint's `atm` (6 slots),
  `bas` (8 slots: atom, l, nprim, nctr, kappa, ptr_exp, ptr_coeff, unused),
  `env` arrays; coordinates at `PTR_COORD`, exponents/coefficients at the
  bas pointers. Coefficients arrive already gto_norm-scaled (see
  docs/conventions.md); Cartesian normalization is
  `intti::cart_norm_pyscf` (include/intti/normalization.hpp).
- Contracted quartet blocks via `intti::eri_quartets_accumulate`
  (include/intti/batch.hpp) with coefficient products as weights, one
  segment per contracted block.
- Spherical: pin the per-row rescale between `intti::c2s_matrix`
  (include/intti/c2s.hpp; sphere-orthonormal, m = −l..+l) and libcint's
  rows by extending prototype/pyscf_validation.py with `int2e_sph`; record
  the factors in docs/conventions.md and implement `sph_rescale(l)` in
  normalization.hpp.
- Oracle: extended pyscf_validation.py compares the façade (ctypes) against
  `mol.intor` for (a) the existing uncontracted system, (b) a contracted
  real basis (H at STO-3G, O at cc-pVDZ), cart and sph, to 1e-12.
- Acceptance stretch: a PySCF SCF on H2O/cc-pVDZ with ERIs monkeypatched to
  the façade reproduces the reference SCF energy to 1e-10 Ha.

## M8 — MPI distribution  [DONE — commit d49acaf]

Delivered; MPI vs serial deviation is exactly 0 at 1, 2, 4 ranks (disjoint
ownership). Environment: `source /etc/profile.d/modules.sh && module load
mpi/openmpi-x86_64` (plain `module load mpi` does not resolve); configure
with `-DCMAKE_CXX_COMPILER=mpicxx -DINTTI_ENABLE_MPI=ON`. Original spec
retained below.

### Original spec

- `coulomb_build`/`exchange_build` MPI variants: bra pairs (J) and (a, b)
  shell blocks (K) strided by rank, `MPI_Allreduce` on the result; the
  owned index sets are disjoint so the reduction is exact.
- Guarded by the existing `INTTI_ENABLE_MPI` option; Kokkos initialized
  after MPI_Init; a separate test binary run under `mpirun -np {1,2,4}`.
- Oracle: distributed result equals the serial build bitwise (same
  summation order per owned block) or to 1e-15 if order differs.

## M9 (delivered) — scalar generalization: GIAOs + quadruple precision

Commits `f8d4fb3` (scalar/real split), `c3b304f` (GIAO), and the quadmath
work. The scalar type is now separate from its real type; `kokkos_scalar_v`
replaced `std::is_floating_point_v` as the execution dispatch (the latter is
true for `__float128`, which Kokkos cannot handle, and false for
`std::complex`). GIAO ERIs are `eri_quartet` on `std::complex`; quadruple
precision is `__float128` behind `INTTI_ENABLE_QUADMATH`.

Two traps recorded for whoever touches this next:
- `Q(0.8)` rounds through *double* and silently caps quad precision at 1e-16.
  Parse decimals straight into quad with `strtoflt128`.
- `M_PIq` / `FLT128_EPSILON` are `Q`-suffixed literal macros that only
  compile with GNU extensions; use `acosq(-1)` instead.

## M10 — batched interactions / per-(t,m) GEMM  [delegable: LOW — strong model]

Restructure `intti::interaction` (include/intti/product.hpp) for lists:
all PSC products sharing a grid pair reuse the per-(t, m) kernel matrices,
so Cholesky S and M matrices in mixed bases become GEMM sequences (the
structure demonstrated in docs/psc.md). Oracle: matches the scalar
`interaction` to 1e-13; target ≥10× on a naux² PSC Gram matrix. Risk:
memory/accuracy tradeoffs in kernel-matrix caching need judgment.

## M11 — GPU enablement  [BLOCKED on hardware — do not attempt runtime claims here]

Work: replace the per-thread stack arrays in the hot kernels (batch.hpp
phase G, jbuild.hpp phase 2, kbuild.hpp — they currently size fixed arrays
off LMAX/JLMAX/KLMAX) with Kokkos team scratch, so a device backend can run
them without blowing per-thread register/stack limits. This refactor is
worth doing on its own and is verifiable on the host backend (results must
be unchanged).

**Hardware reality on the development machine (checked 2026-07-12):** it is
an Intel laptop — Alder Lake Iris Xe integrated graphics only. There is *no*
AMD GPU (`/dev/kfd` does not exist), so the installed `rocm/*` modules are
unusable; there is no CUDA device; no oneAPI/SYCL toolchain is installed;
and the system Kokkos is built with OPENMP and SERIAL backends only.

Therefore: a GPU backend **cannot be validated here**. Do not claim GPU
correctness or performance from this machine. Either (a) do only the
team-scratch refactor, validated on the host backend, or (b) run the GPU
work on a machine with a real discrete GPU. An Iris Xe via a SYCL backend
would at best exercise the device code paths; it is not a meaningful
performance target for J/K builds.

## M12 — optimized quadrature tooling  [delegable: LOW — strong model]

Minimax/Beylkin–Monzón weights for the t kernels and STO Gaussian
expansions, generated with the arbitrary-precision core (long double /
MPFR); the log-GL baseline and its measured convergence (README, M1 study;
STO study in the M2 session notes) are the curves to beat. Oracle:
pointwise and ERI worst-case error curves vs node count, reproducing the
published Beylkin–Monzón rates.

## Standing items

- **No performance data exists.** The library is exhaustively verified for
  correctness and has never been benchmarked against libcint. Its cost model
  (64 t nodes per integral) may well make it *slower* than analytic codes for
  plain GTO ERIs — in which case its value lies in what analytic codes cannot
  do (mixed atomic/diatomic/3D representations, GIAOs without complex Boys
  functions, local exchange energy densities). Benchmark before optimizing.
- **The stated scientific goal is not served yet:** local exchange energy
  densities for local hybrids, which motivated the whole design.
- CI does not exercise the PySCF cross-validation or the MPI tests (neither
  is installed on the runners); both are verified locally only.
- Even-tempered radial-grid shift optimization and the Graf addition theorem
  for parallel-offset PSC axes: fold into M10's design review.
- K from Cholesky vectors covers production exchange; kbuild.hpp remains the
  exact reference path.
- Complex (GIAO) integrals currently run the serial host path only; batched
  and J/K builds for complex densities are not implemented.
