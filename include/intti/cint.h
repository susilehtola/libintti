/* SPDX-License-Identifier: MPL-2.0 */
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

int intti_int2e_sph(double *out, const int *shls, const int *atm, int natm,
                     const int *bas, int nbas, const double *env, void *opt,
                     double *cache);

#ifdef __cplusplus
}
#endif

#endif /* INTTI_CINT_H */
