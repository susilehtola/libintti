/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (C) 2026 Susi Lehtola */
#ifndef INTTI_CINT_H
#define INTTI_CINT_H

/* libcint-compatible C ABI: entry points that decode libcint's atm/bas/env
 * data model (see include/intti/cint.h layout notes below and
 * docs/conventions.md) and evaluate two-electron repulsion integral shell
 * quartets through the libintti quadrature core.
 *
 * atm[natm][6]: slot 1 (PTR_COORD) is the offset into env of the atom's
 *   3 Cartesian coordinates; the other slots are unused by this facade.
 * bas[nbas][8]: 0 = atom index, 1 = angular momentum l, 2 = nprim,
 *   3 = nctr, 4 = kappa (unused), 5 = ptr_exp (offset into env of nprim
 *   exponents), 6 = ptr_coeff (offset into env of the nprim x nctr
 *   contraction coefficients, COLUMN-major: env[ptr_coeff + c*nprim + p]
 *   is the coefficient of primitive p in contracted function c), 7 unused.
 * env[]: flat scalar pool holding coordinates, exponents and contraction
 *   coefficients at the offsets above. Coefficients arrive already
 *   gto_norm-scaled by the caller (PySCF/libcint convention); see
 *   docs/conventions.md for the exact rescale libintti applies to recover
 *   its own (unnormalized-primitive) convention.
 *
 * Both entry points evaluate the contracted shell quartet named by shls[4]
 * (shell indices bra0, bra1, ket0, ket1) and write it to out in libcint's
 * Fortran-style (column-major) layout with dimensions (di, dj, dk, dl),
 * where di is the number of AOs of shell shls[0] (ncart(l)*nctr for
 * int2e_cart, (2l+1)*nctr for int2e_sph), and so on for dj, dk, dl.
 *
 * opt and cache follow libcint's signature for drop-in compatibility but
 * are ignored (NULL is accepted).
 *
 * Return value: 1 if any integral in the block is numerically nonzero,
 * 0 if the whole block is exactly zero (libcint's screening convention).
 */

#ifdef __cplusplus
extern "C" {
#endif

int intti_int2e_cart(double *out, const int *shls, const int *atm, int natm,
                      const int *bas, int nbas, const double *env, void *opt,
                      double *cache);

/* Matrix-level Coulomb/exchange: the surface an SCF driver should use, mirroring
 * pyscf.scf.hf.SCF.get_jk(mol, dm, hermi, with_j, with_k, omega). dms/vj/vk are
 * ndm consecutive nao x nao row-major matrices. hermi is one int per density:
 * 1 = symmetric, 2 = anti-symmetric, 0 = general. omega = 0 selects plain
 * Coulomb, otherwise the erf range-separated kernel. tau = 0 disables screening.
 * Requires an UNCONTRACTED Cartesian basis; returns 0 on success, -1 if any
 * shell is contracted, -2 on an internal size mismatch.
 *
 * Prefer this over intti_int2e_* for anything that builds a Fock matrix: the
 * per-quartet entries exist for libcint drop-in compatibility, and driving an
 * SCF through them defeats the batching the library is built around. */
int intti_get_jk(double *vj, double *vk, const double *dms, int ndm, const int *hermi,
                 int with_j, int with_k, const int *atm, int natm, const int *bas,
                 int nbas, const double *env, double omega, double tau);

int intti_int2e_sph(double *out, const int *shls, const int *atm, int natm,
                     const int *bas, int nbas, const double *env, void *opt,
                     double *cache);

#ifdef __cplusplus
}
#endif

#endif /* INTTI_CINT_H */
