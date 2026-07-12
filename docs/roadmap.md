# libintti roadmap — M7 and beyond

Status: `main` at `b70508f`, 55/55 tests, ERIs validated against PySCF to
2.3×10⁻¹⁴. This document specifies the remaining milestones precisely enough
for delegated implementation. **Ground rules for any implementer:**

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

## M9 — batched interactions / per-(t,m) GEMM  [delegable: LOW — strong model]

Restructure `intti::interaction` (include/intti/product.hpp) for lists:
all PSC products sharing a grid pair reuse the per-(t, m) kernel matrices,
so Cholesky S and M matrices in mixed bases become GEMM sequences (the
structure demonstrated in docs/psc.md). Oracle: matches the scalar
`interaction` to 1e-13; target ≥10× on a naux² PSC Gram matrix. Risk:
memory/accuracy tradeoffs in kernel-matrix caching need judgment.

## M10 — GPU enablement  [delegable: MEDIUM for the mechanics]

Replace the per-thread stack arrays in the hot kernels (batch.hpp phase G,
jbuild.hpp phase 2, kbuild.hpp) with team scratch, and enable a HIP/CUDA
Kokkos backend. NOTE: this machine has ROCm installed (`module avail` shows
rocm/rocm-7.1 and gfx targets), so runtime validation on real hardware is
possible — do not settle for a compile-only check. Oracle: every existing
test passes with the GPU backend; results match the host backend to 1e-13.

## M11 — optimized quadrature tooling  [delegable: LOW — strong model]

Minimax/Beylkin–Monzón weights for the t kernels and STO Gaussian
expansions, generated with the arbitrary-precision core (long double /
MPFR); the log-GL baseline and its measured convergence (README, M1 study;
STO study in the M2 session notes) are the curves to beat. Oracle:
pointwise and ERI worst-case error curves vs node count, reproducing the
published Beylkin–Monzón rates.

## Standing items

- Push to a remote; the GitHub CI (gcc/clang, OpenMP, ASan) has never run.
- Even-tempered radial-grid shift optimization and the Graf addition
  theorem for parallel-offset PSC axes: fold into M9's design review.
- K from Cholesky vectors covers production exchange; kbuild.hpp remains
  the exact reference path.
