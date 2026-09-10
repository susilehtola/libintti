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

/* Derivative J/K in the bra-gradient convention, mirroring
 * pyscf.grad.rhf.get_jk(mol, dm): vj/vk are 3 x nao x nao with the derivative on
 * the first AO index, NOT folded onto atoms. Same restrictions as
 * intti_get_jk. */
int intti_get_jk_ip1(double *vj, double *vk, const double *dm, const int *atm, int natm,
                     const int *bas, int nbas, const double *env, double tau);

/* Skeleton (fixed-density) electronic Hessian -- what
 * pyscf.hessian.rhf.partial_hess_elec computes: sum D d2h + d2E_2e - sum W d2S,
 * as (3 nbas) x (3 nbas) row-major indexed by SHELL centre (caller folds onto
 * atoms). Excludes the CPHF response terms and the nuclear repulsion Hessian,
 * both of which are the caller's. Same restrictions as intti_get_jk. */
int intti_hess_skeleton(double *hess, const double *dm, const double *W, const int *atm,
                        int natm, const int *bas, int nbas, const double *env, double tau);

/* The four int2e_ip1 contractions pyscf.hessian.rhf.make_h1 needs, for one
 * atom's shell slice [shl0, shl1). Outputs are 3 x nao x nao each; pass NULL to
 * skip one. Same restrictions as intti_get_jk. */
int intti_ip1_h1_jk(double *vj1, double *vj2, double *vk1, double *vk2, const double *dm,
                    int shl0, int shl1, const int *atm, int natm, const int *bas,
                    int nbas, const double *env, double tau);

/* One-electron derivative matrices in PySCF's gradient conventions, 3 x nao x
 * nao. which: 0 = int1e_ipovlp, 1 = int1e_ipkin, 2 = int1e_ipnuc (all nuclei,
 * weighted -Z), 3 = int1e_iprinv for nucleus `iatm` UNWEIGHTED. Same
 * restrictions as intti_get_jk. */
int intti_int1e_ip(double *out, int which, int iatm, const int *atm, int natm,
                   const int *bas, int nbas, const double *env);

/* Density-fitted (RI) two-electron Hessians, the pyscf.df.hessian.rhf seam.
 * hj = d^2[+1/2 Tr(D J)] (PySCF's ej); hk = d^2[-1/4 Tr(D K)] (PySCF's -ek),
 * with D = cocc cocc^T -- pass cocc = sqrt(2) C_occ for a closed shell. Each is
 * (3 ncen) x (3 ncen) row-major, ncen = nbas + anbas, ORBITAL shell centres
 * then AUXILIARY shell centres; the caller folds shells onto atoms, which is
 * also where the auxiliary-basis response comes from. Pass NULL to skip either.
 * Both bases must be uncontracted Cartesian: the RI derivative layer has no
 * contraction-aware builders yet. */
int intti_ri_hess_jk(double *hj, double *hk, const double *dm, const double *cocc,
                     int nvec, const int *atm, int natm, const int *bas, int nbas,
                     const double *env, const int *aatm, int anatm, const int *abas,
                     int anbas, const double *aenv, double tau_lin);

/* Derivative RI J/K matrices, the pyscf.df.hessian.rhf._gen_jk seam: vj/vk are
 * (3 ngrp) x nao x nao holding dJ/dR and dK/dR at fixed density. `group` maps
 * the nbas + anbas shell centres (orbital first, then auxiliary) onto groups --
 * pass each shell's ATOM for the per-atom form that make_h1 wants, which is
 * also PySCF's auxbasis_response = 2. D = cocc cocc^T, so pass
 * cocc = sqrt(2) C_occ for a closed shell. Pass NULL to skip either output.
 * Both bases must be uncontracted Cartesian. */
int intti_ri_deriv_jk(double *vj, double *vk, const double *dm, const double *cocc,
                      int nvec, const int *group, int ngrp, const int *atm, int natm,
                      const int *bas, int nbas, const double *env, const int *aatm,
                      int anatm, const int *abas, int anbas, const double *aenv,
                      double tau_lin);

/* RI (density-fitted) J/K as an explicitly CACHED handle. intti_ri_open builds
 * the fit ONCE (B is nao^2 x naux); each intti_ri_get_jk is then only GEMMs.
 * There is deliberately no one-shot form: that signature would rebuild the fit
 * on every call, which measured 34 ms -> 570 s across an SCF. Densities need not
 * be symmetric -- the general and antisymmetric ones response theory produces
 * are served exactly. vj/vk are ndm x nao x nao. Cartesian only; either basis
 * may be generally contracted, independently of the other.
 * Returns NULL on a violated restriction; free with intti_ri_close. */
void *intti_ri_open(const int *atm, int natm, const int *bas, int nbas,
                    const double *env, const int *aatm, int anatm, const int *abas,
                    int anbas, const double *aenv, double tau_lin);
void intti_ri_close(void *handle);
int intti_ri_get_jk(void *handle, double *vj, double *vk, const double *dms, int ndm,
                    int with_j, int with_k);

/* Two- and three-centre Coulomb tensors, for direct comparison with libcint's
 * int2c2e (naux x naux) and int3c2e (nao x nao x naux). Whole tensors, not a
 * per-quartet surface. Cartesian only; either basis may be generally
 * contracted. */
int intti_coulomb_2c(double *out, const int *aatm, int anatm, const int *abas,
                     int anbas, const double *aenv);
int intti_coulomb_3c(double *out, const int *atm, int natm, const int *bas, int nbas,
                     const double *env, const int *aatm, int anatm, const int *abas,
                     int anbas, const double *aenv);

/* Real -> complex spherical harmonics (Condon-Shortley) on a whole nao x nao
 * AO matrix, real and imaginary parts returned separately. Input is in the REAL
 * spherical basis, one l block per bas entry (nctr == 1). Set libcint_order for
 * input in libcint/PySCF's convention, which keeps p as (x,y,z); 0 for the
 * standard m = -l..+l ordering c2s_matrix produces. A boundary transform, not a
 * second integral path. */
int intti_real_to_complex(double *re, double *im, const double *M, const int *bas,
                          int nbas, int libcint_order);

/* Rewrite an nao x nao AO matrix from intti's convention into another code's.
 * to: 0 = intti, 1 = libcint/PySCF. spherical: nonzero for spherical shells.
 * Ordering and PHASE only -- normalisation can depend on the primitive exponent,
 * so it is not a per-l property and is handled at basis-set conversion. */
int intti_convert_convention(double *out, const double *M, int to, int spherical,
                             const int *bas, int nbas);

int intti_int2e_sph(double *out, const int *shls, const int *atm, int natm,
                     const int *bas, int nbas, const double *env, void *opt,
                     double *cache);

#ifdef __cplusplus
}
#endif

#endif /* INTTI_CINT_H */
