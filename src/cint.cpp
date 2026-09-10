// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

// Implementation of the libcint-compatible C facade (include/intti/cint.h).
//
// Contracted evaluation uses a single batched intti::eri_quartets() call per
// shell quartet, over the full list of primitive (bra pair, ket pair)
// combinations, followed by a plain host-side weighted accumulation loop
// over the primitive x contraction combinations (an eri_quartets_accumulate
// segment per contracted block would recompute the same primitive integral
// once per contraction-index combination, since the accumulate driver keys
// one (coeff, segment) pair per batch entry; the host loop reuses each
// primitive quartet's raw value across every contraction combination
// instead). This is the "plain accumulation loop" option named in the
// roadmap; correctness first, and the arithmetic it performs (a handful of
// multiply-adds per output component per primitive/contraction combination)
// is cheap next to the quadrature itself.
//
// Weight derivation (empirically verified against PySCF 2.13, see the
// report and docs/conventions.md): PySCF's env contraction coefficients are
// already scaled by its own internal gto_norm(l, alpha), which equals
// intti::cart_norm_pyscf(l, alpha) for l >= 2 but carries an extra
// l-independent factor sqrt(4 pi / (2l+1)) for l <= 1. The correction
// applied here, coeff_rescale(l), is exactly the inverse of that factor, so
// weight = env_coeff * coeff_rescale(l) reconstructs the coefficient that
// multiplies intti's unnormalized primitive to give PySCF's cart_norm_pyscf
// convention directly -- no additional per-primitive normalization call is
// needed.

#include "intti/cint.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "intti/batch.hpp"
#include "intti/c2s.hpp"
#include "intti/gto.hpp"
#include "intti/jk.hpp"
#include "intti/erihess.hpp"
#include "intti/geohess.hpp"
#include "intti/harmonics.hpp"
#include "intti/kernel.hpp"
#include "intti/contracted.hpp"
#include "intti/deriv.hpp"
#include "intti/nuclear.hpp"
#include "intti/ri.hpp"
#include "intti/rigrad.hpp"
#include "intti/erihess.hpp"
#include "intti/geohess.hpp"
#include "intti/kernel.hpp"
#include "intti/contracted.hpp"
#include "intti/deriv.hpp"
#include "intti/nuclear.hpp"
#include "intti/math.hpp"
#include "intti/normalization.hpp"
#include "intti/tgrid.hpp"

namespace {

// libcint bas[] column indices (bas[ish*8 + slot]).
enum BasSlot { ATOM_OF = 0, ANG_OF = 1, NPRIM_OF = 2, NCTR_OF = 3, PTR_EXP = 5, PTR_COEFF = 6 };
// libcint atm[] column index of the coordinate offset into env.
enum AtmSlot { CHARGE_OF = 0, PTR_COORD = 1 };

struct ShellInfo {
  int l{0}, nprim{0}, nctr{0};
  const double *alpha{nullptr};   // nprim entries
  const double *coeff{nullptr};   // nprim x nctr, column-major: coeff[c*nprim+p]
  double center[3]{0, 0, 0};
};

ShellInfo decode_shell(int ish, const int *atm, const int *bas, const double *env) {
  ShellInfo s;
  const int *b = bas + static_cast<std::size_t>(ish) * 8;
  s.l = b[ANG_OF];
  s.nprim = b[NPRIM_OF];
  s.nctr = b[NCTR_OF];
  s.alpha = env + b[PTR_EXP];
  s.coeff = env + b[PTR_COEFF];
  const int *a = atm + static_cast<std::size_t>(b[ATOM_OF]) * 6;
  const double *coord = env + a[PTR_COORD];
  s.center[0] = coord[0];
  s.center[1] = coord[1];
  s.center[2] = coord[2];
  return s;
}

// Correction from PySCF's internal gto_norm(l, alpha) (already baked into
// env's contraction coefficients) to intti::cart_norm_pyscf(l, alpha); see
// the file header and docs/conventions.md.
double coeff_rescale(int l) {
  if (l <= 1) return std::sqrt((2 * l + 1) / (4 * intti::pi_v<double>()));
  return 1.0;
}

void ensure_kokkos() {
  static const bool initialized = [] {
    if (!Kokkos::is_initialized() && !Kokkos::is_finalized()) {
      Kokkos::initialize();
      std::atexit([] {
        if (Kokkos::is_initialized()) Kokkos::finalize();
      });
    }
    return true;
  }();
  (void)initialized;
}

const intti::TGrid<double> &default_grid() {
  static const intti::TGrid<double> grid = [] {
    ensure_kokkos();
    return intti::make_tgrid(intti::coulomb<double>());
  }();
  return grid;
}

// Apply an (n_out x n_in) row-major transform T along one axis of a
// Fortran-layout (column-major) tensor: axis has 'pre' faster-varying
// entries below it and 'post' slower-varying repetitions above it.
void transform_axis(const std::vector<double> &in, int pre, int n_in, int post,
                     const std::vector<double> &T, int n_out, std::vector<double> &out) {
  out.assign(static_cast<std::size_t>(pre) * n_out * post, 0.0);
  for (int p = 0; p < post; ++p)
    for (int i = 0; i < n_in; ++i)
      for (int r = 0; r < pre; ++r) {
        const double val =
            in[static_cast<std::size_t>(r) + static_cast<std::size_t>(pre) * (i + static_cast<std::size_t>(n_in) * p)];
        if (val == 0.0) continue;
        for (int m = 0; m < n_out; ++m)
          out[static_cast<std::size_t>(r) +
              static_cast<std::size_t>(pre) * (m + static_cast<std::size_t>(n_out) * p)] +=
              T[static_cast<std::size_t>(m) * n_in + i] * val;
      }
}

// Block-diagonal (over contraction index) cart -> spherical transform for
// one shell: identity for l <= 1 (libcint's spherical AOs coincide exactly
// with the cart ones for these shells, same order), c2s_matrix(l) *
// sph_rescale(l) for l >= 2. Row-major (n_out x n_in).
std::vector<double> build_sph_transform(int l, int nctr) {
  const int nc = intti::ncart(l);
  const int nm = 2 * l + 1;
  const int n_in = nc * nctr, n_out = nm * nctr;
  std::vector<double> T(static_cast<std::size_t>(n_out) * n_in, 0.0);
  if (l <= 1) {
    for (int c = 0; c < nctr; ++c)
      for (int k = 0; k < nc; ++k)
        T[static_cast<std::size_t>(c * nm + k) * n_in + (c * nc + k)] = 1.0;
  } else {
    const auto C = intti::c2s_matrix<double>(l); // (nm x nc) row-major
    const double s = intti::sph_rescale<double>(l);
    for (int c = 0; c < nctr; ++c)
      for (int m = 0; m < nm; ++m)
        for (int k = 0; k < nc; ++k)
          T[static_cast<std::size_t>(c * nm + m) * n_in + (c * nc + k)] = C[static_cast<std::size_t>(m) * nc + k] * s;
  }
  return T;
}

int eval_int2e(double *out, const int *shls, const int *atm, int /*natm*/, const int *bas,
               int /*nbas*/, const double *env, bool spherical) {
  ShellInfo sh[4];
  for (int i = 0; i < 4; ++i) sh[i] = decode_shell(shls[i], atm, bas, env);

  const int la = sh[0].l, lb = sh[1].l, lc = sh[2].l, ld = sh[3].l;
  if (la > intti::LMAX || lb > intti::LMAX || lc > intti::LMAX || ld > intti::LMAX) {
    std::fprintf(stderr, "intti_cint: angular momentum exceeds LMAX (%d)\n", intti::LMAX);
    return 0;
  }
  const int nca = intti::ncart(la), ncb = intti::ncart(lb), ncc = intti::ncart(lc),
            ncd = intti::ncart(ld);
  const int npi = sh[0].nprim, npj = sh[1].nprim, npk = sh[2].nprim, npl = sh[3].nprim;
  const int ndi = nca * sh[0].nctr, ndj = ncb * sh[1].nctr, ndk = ncc * sh[2].nctr,
            ndl = ncd * sh[3].nctr;

  ensure_kokkos();

  // primitive shells for each of the 4 basis functions
  auto prim_shells = [](const ShellInfo &s) {
    std::vector<intti::PrimitiveShell<double>> v(s.nprim);
    for (int p = 0; p < s.nprim; ++p)
      v[p] = intti::PrimitiveShell<double>{s.alpha[p], {s.center[0], s.center[1], s.center[2]}, s.l};
    return v;
  };
  const auto pi_shells = prim_shells(sh[0]);
  const auto pj_shells = prim_shells(sh[1]);
  const auto pk_shells = prim_shells(sh[2]);
  const auto pl_shells = prim_shells(sh[3]);

  // bra pairs (pi, pj) then ket pairs (pk, pl) in one pair list
  std::vector<intti::ShellPair<double>> pairs;
  pairs.reserve(static_cast<std::size_t>(npi) * npj + static_cast<std::size_t>(npk) * npl);
  std::vector<int> bra_idx(static_cast<std::size_t>(npi) * npj);
  std::vector<int> ket_idx(static_cast<std::size_t>(npk) * npl);
  for (int pi = 0; pi < npi; ++pi)
    for (int pj = 0; pj < npj; ++pj) {
      pairs.push_back(intti::make_pair(pi_shells[pi], pj_shells[pj]));
      bra_idx[static_cast<std::size_t>(pi) * npj + pj] = static_cast<int>(pairs.size() - 1);
    }
  for (int pk = 0; pk < npk; ++pk)
    for (int pl = 0; pl < npl; ++pl) {
      pairs.push_back(intti::make_pair(pk_shells[pk], pl_shells[pl]));
      ket_idx[static_cast<std::size_t>(pk) * npl + pl] = static_cast<int>(pairs.size() - 1);
    }

  auto tab = intti::make_pair_table(pairs);
  std::vector<std::pair<int, int>> quartets;
  quartets.reserve(static_cast<std::size_t>(npi) * npj * npk * npl);
  for (int pi = 0; pi < npi; ++pi)
    for (int pj = 0; pj < npj; ++pj)
      for (int pk = 0; pk < npk; ++pk)
        for (int pl = 0; pl < npl; ++pl)
          quartets.push_back({bra_idx[static_cast<std::size_t>(pi) * npj + pj],
                               ket_idx[static_cast<std::size_t>(pk) * npl + pl]});

  auto batch = intti::make_batch(tab, quartets);
  Kokkos::View<double *> outv("intti::cint::out", batch.nout_total);
  intti::QuartetWorkspace<double> ws;
  intti::eri_quartets(tab, batch, default_grid(), outv, ws);
  auto oh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, outv);

  const double ri = coeff_rescale(la), rj = coeff_rescale(lb), rk = coeff_rescale(lc),
               rl = coeff_rescale(ld);
  auto weight = [](const ShellInfo &s, int p, int c, double rescale) {
    return s.coeff[static_cast<std::size_t>(c) * s.nprim + p] * rescale;
  };

  // contracted, Cartesian, Fortran-layout block: index (ci*nca+ka) fastest, ...
  std::vector<double> cart_block(static_cast<std::size_t>(ndi) * ndj * ndk * ndl, 0.0);
  int iq = 0;
  for (int pi = 0; pi < npi; ++pi)
    for (int pj = 0; pj < npj; ++pj)
      for (int pk = 0; pk < npk; ++pk)
        for (int pl = 0; pl < npl; ++pl, ++iq) {
          const int off = batch.h_offset[iq];
          for (int ci = 0; ci < sh[0].nctr; ++ci) {
            const double wi = weight(sh[0], pi, ci, ri);
            for (int cj = 0; cj < sh[1].nctr; ++cj) {
              const double wj = weight(sh[1], pj, cj, rj);
              for (int ck = 0; ck < sh[2].nctr; ++ck) {
                const double wk = weight(sh[2], pk, ck, rk);
                for (int cl = 0; cl < sh[3].nctr; ++cl) {
                  const double wl = weight(sh[3], pl, cl, rl);
                  const double w = wi * wj * wk * wl;
                  if (w == 0.0) continue;
                  for (int ka = 0; ka < nca; ++ka) {
                    const int ii = ci * nca + ka;
                    for (int kb = 0; kb < ncb; ++kb) {
                      const int jj = cj * ncb + kb;
                      for (int kc = 0; kc < ncc; ++kc) {
                        const int kk = ck * ncc + kc;
                        const std::size_t raw_base =
                            static_cast<std::size_t>(off) + ((static_cast<std::size_t>(ka) * ncb + kb) * ncc + kc) * ncd;
                        const std::size_t out_base =
                            static_cast<std::size_t>(ii) +
                            static_cast<std::size_t>(ndi) *
                                (jj + static_cast<std::size_t>(ndj) * (kk + static_cast<std::size_t>(ndk) * (cl * ncd)));
                        for (int kd = 0; kd < ncd; ++kd)
                          cart_block[out_base + static_cast<std::size_t>(ndi) * ndj * ndk * kd] +=
                              w * oh(raw_base + kd);
                      }
                    }
                  }
                }
              }
            }
          }
        }

  bool nonzero = false;
  const std::size_t nout = cart_block.size();
  if (!spherical) {
    for (std::size_t k = 0; k < nout; ++k) {
      out[k] = cart_block[k];
      if (cart_block[k] != 0.0) nonzero = true;
    }
    return nonzero ? 1 : 0;
  }

  const auto Ti = build_sph_transform(la, sh[0].nctr);
  const auto Tj = build_sph_transform(lb, sh[1].nctr);
  const auto Tk = build_sph_transform(lc, sh[2].nctr);
  const auto Tl = build_sph_transform(ld, sh[3].nctr);
  const int nmi = (2 * la + 1) * sh[0].nctr, nmj = (2 * lb + 1) * sh[1].nctr,
            nmk = (2 * lc + 1) * sh[2].nctr, nml = (2 * ld + 1) * sh[3].nctr;

  std::vector<double> t0, t1, t2, t3;
  transform_axis(cart_block, 1, ndi, ndj * ndk * ndl, Ti, nmi, t0);
  transform_axis(t0, nmi, ndj, ndk * ndl, Tj, nmj, t1);
  transform_axis(t1, nmi * nmj, ndk, ndl, Tk, nmk, t2);
  transform_axis(t2, nmi * nmj * nmk, ndl, 1, Tl, nml, t3);

  for (std::size_t k = 0; k < t3.size(); ++k) {
    out[k] = t3[k];
    if (t3[k] != 0.0) nonzero = true;
  }
  return nonzero ? 1 : 0;
}

} // namespace

// defined below, with the contracted J/K facade
intti::ContractedBasis<double> contracted_basis_from(const int *atm, const int *bas,
                                                     int nbas, const double *env);


extern "C" int intti_int2e_cart(double *out, const int *shls, const int *atm, int natm,
                                 const int *bas, int nbas, const double *env, void * /*opt*/,
                                 double * /*cache*/) {
  return eval_int2e(out, shls, atm, natm, bas, nbas, env, false);
}


// ---------------------------------------------------------------------------
// Density-fitted (RI) two-electron Hessians.
//
// The seam is pyscf.df.hessian.rhf._partial_hess_ejk, which returns (e1, ej, ek)
// with partial_hess_elec = e1 + ej - ek. The conventions line up because both
// sides differentiate the same energy expressions:
//
//   ri_j_hessian(D)                 = d^2 [ +1/2 Tr(D J) ]  ==  PySCF ej
//   ri_k_hessian_occ(CL, CR)        = d^2 [ -1/4 Tr(D K) ]  == -PySCF ek
//
// with D = CL CR^T, so a closed-shell dm0 = 2 C_occ C_occ^T is passed as
// CL = CR = sqrt(2) C_occ. Both are returned separately rather than summed, so
// a caller can check them independently -- summing first would let an error in
// one hide inside the other.
//
// Output is (3 ncen) x (3 ncen) row-major with ncen = nbas + anbas: ORBITAL
// shell centres first, then AUXILIARY shell centres. The caller folds shells
// onto atoms with bas[:,ATOM_OF] / abas[:,ATOM_OF], which is also what makes
// the auxiliary-basis response terms (PySCF's auxbasis_response) come out --
// they are simply the auxiliary block of the same matrix.
//
// RESTRICTIONS: both bases must be uncontracted Cartesian. The RI derivative
// layer has no contraction-aware builders yet -- unlike the direct J/K path,
// which does. Reported rather than silently mis-answered.
extern "C" int intti_ri_hess_jk(double *hj, double *hk, const double *dm,
                                const double *cocc, int nvec, const int *atm, int natm,
                                const int *bas, int nbas, const double *env,
                                const int *aatm, int anatm, const int *abas, int anbas,
                                const double *aenv, double tau_lin) {
  ensure_kokkos();
  (void)natm;
  (void)anatm;
  auto build = [](const int *a, const int *b, int nb, const double *e,
                  std::vector<double> &scale, intti::ShellBasis<double> &out) {
    std::vector<intti::PrimitiveShell<double>> shells;
    for (int ish = 0; ish < nb; ++ish) {
      const ShellInfo s = decode_shell(ish, a, b, e);
      if (s.nctr != 1 || s.nprim != 1) return -1;
      shells.push_back(intti::PrimitiveShell<double>{
          s.alpha[0], {s.center[0], s.center[1], s.center[2]}, s.l});
      const double c = s.coeff[0] * coeff_rescale(s.l);
      for (int k = 0; k < intti::ncart(s.l); ++k) scale.push_back(c);
    }
    out = intti::make_basis(shells);
    return 0;
  };
  auto has_contraction = [](const int *b, int nb) {
    for (int ish = 0; ish < nb; ++ish) {
      const int *r = b + static_cast<std::size_t>(ish) * 8;
      if (r[NPRIM_OF] != 1 || r[NCTR_OF] != 1) return true;
    }
    return false;
  };
  const std::size_t dimc = static_cast<std::size_t>(3) * (nbas + anbas);
  if (has_contraction(bas, nbas) || has_contraction(abas, anbas)) {
    // The contracted RI-J Hessian: fit in the contracted auxiliary space, then
    // the primitive derivative kernel on the pushed-down density and gamma.
    // Result stays indexed by CONTRACTED shell, so the caller's shell -> atom
    // fold is unchanged. RI-K has no contracted path yet and is refused rather
    // than answered with the wrong basis.
    const auto co = contracted_basis_from(atm, bas, nbas, env);
    const auto ca = contracted_basis_from(aatm, abas, anbas, aenv);
    if (hj) {
      if (!dm) return -7;
      const auto H = intti::ri_j_hessian(co, ca, dm, default_grid(), tau_lin);
      for (std::size_t i = 0; i < dimc * dimc; ++i) hj[i] = H[i];
    }
    if (hk) {
      if (!cocc || nvec < 1) return -7;
      const auto H = intti::ri_k_hessian_occ(co, ca, cocc, cocc, nvec, default_grid(),
                                             tau_lin);
      for (std::size_t i = 0; i < dimc * dimc; ++i) hk[i] = H[i];
    }
    return 0;
  }
  std::vector<double> oscale, ascale;
  intti::ShellBasis<double> orb, aux;
  if (build(atm, bas, nbas, env, oscale, orb) != 0) return -1;
  if (build(aatm, abas, anbas, aenv, ascale, aux) != 0) return -1;
  const int nao = orb.nao;
  if (static_cast<int>(oscale.size()) != nao) return -2;
  if (static_cast<int>(ascale.size()) != aux.nao) return -2;
  const std::size_t dim = dimc;
  const auto &grid = default_grid();

  // The AUXILIARY normalization does not need rescaling on the way out: the
  // Hessian is a scalar contraction over auxiliary indices, and the fit
  // coefficients absorb whatever convention the auxiliary AOs carry, as long as
  // the metric M and the three-centre integrals use the SAME one. Only the
  // orbital-side density has to be converted.
  if (hj) {
    if (!dm) return -7;
    std::vector<double> Ds(static_cast<std::size_t>(nao) * nao);
    for (int i = 0; i < nao; ++i)
      for (int j = 0; j < nao; ++j)
        Ds[static_cast<std::size_t>(i) * nao + j] =
            dm[static_cast<std::size_t>(i) * nao + j] * oscale[i] * oscale[j];
    const auto H = intti::ri_j_hessian(orb, aux, Ds.data(), grid, tau_lin);
    for (std::size_t i = 0; i < dim * dim; ++i) hj[i] = H[i];
  }
  if (hk) {
    if (!cocc || nvec < 1) return -7;
    std::vector<double> Cs(static_cast<std::size_t>(nao) * nvec);
    for (int i = 0; i < nao; ++i)
      for (int k = 0; k < nvec; ++k)
        Cs[static_cast<std::size_t>(i) * nvec + k] =
            cocc[static_cast<std::size_t>(i) * nvec + k] * oscale[i];
    const auto H = intti::ri_k_hessian_occ(orb, aux, Cs.data(), Cs.data(), nvec, grid,
                                           tau_lin);
    for (std::size_t i = 0; i < dim * dim; ++i) hk[i] = H[i];
  }
  return 0;
}


// Derivative RI J/K MATRICES, the pyscf.df.hessian.rhf._gen_jk seam (which
// make_h1 wraps as h1ao[ia] = dh/dR_ia + vj1 - vk1/2, the CPHF right-hand side).
//
// vj[3*g + e] = dJ/dR_{g,e} and vk[3*g + e] = dK/dR_{g,e} at FIXED density, for
// perturbation group g. `group` maps each of the nbas + anbas shell centres --
// orbital shells first, then auxiliary -- onto a group; pass each shell's ATOM
// and the output is per atom, which is both what the caller wants and what makes
// this affordable. The library folds at the accumulation, so the shell-resolved
// object (43 GB at nao = 1000 against a 2.4 GB atom-folded output) is never
// built. Passing the identity map reproduces the shell-resolved form.
//
// The auxiliary-basis response is not a separate term here: it is the auxiliary
// shells' contribution to whichever group carries them, so mapping aux shells to
// their atoms is exactly PySCF's auxbasis_response = 2.
//
// D = cocc cocc^T, so a closed shell dm0 = 2 C_occ C_occ^T goes in as
// cocc = sqrt(2) C_occ. Both bases must be uncontracted Cartesian.
extern "C" int intti_ri_deriv_jk(double *vj, double *vk, const double *dm,
                                 const double *cocc, int nvec, const int *group, int ngrp,
                                 const int *atm, int natm, const int *bas, int nbas,
                                 const double *env, const int *aatm, int anatm,
                                 const int *abas, int anbas, const double *aenv,
                                 double tau_lin) {
  ensure_kokkos();
  (void)natm;
  (void)anatm;
  auto build = [](const int *a, const int *b, int nb, const double *e,
                  std::vector<double> &scale, intti::ShellBasis<double> &out) {
    std::vector<intti::PrimitiveShell<double>> shells;
    for (int ish = 0; ish < nb; ++ish) {
      const ShellInfo s = decode_shell(ish, a, b, e);
      if (s.nctr != 1 || s.nprim != 1) return -1;
      shells.push_back(intti::PrimitiveShell<double>{
          s.alpha[0], {s.center[0], s.center[1], s.center[2]}, s.l});
      const double c = s.coeff[0] * coeff_rescale(s.l);
      for (int k = 0; k < intti::ncart(s.l); ++k) scale.push_back(c);
    }
    out = intti::make_basis(shells);
    return 0;
  };
  std::vector<double> oscale, ascale;
  intti::ShellBasis<double> orb, aux;
  if (build(atm, bas, nbas, env, oscale, orb) != 0) return -1;
  if (build(aatm, abas, anbas, aenv, ascale, aux) != 0) return -1;
  const int nao = orb.nao;
  if (static_cast<int>(oscale.size()) != nao) return -2;
  const std::size_t N = static_cast<std::size_t>(nao) * nao;
  if (!group || ngrp < 1) return -7;
  std::vector<int> grp(group, group + (nbas + anbas));
  for (int g : grp)
    if (g < 0 || g >= ngrp) return -8;
  const auto &grid = default_grid();

  if (vj) {
    if (!dm) return -7;
    std::vector<double> Ds(N);
    for (int i = 0; i < nao; ++i)
      for (int j = 0; j < nao; ++j)
        Ds[static_cast<std::size_t>(i) * nao + j] =
            dm[static_cast<std::size_t>(i) * nao + j] * oscale[i] * oscale[j];
    std::vector<intti::JKRequest<double>> reqs(1);
    reqs[0].D = Ds.data();
    reqs[0].sym = intti::DensitySymmetry::Symmetric;
    reqs[0].terms = intti::FockTerms::Coulomb;
    const auto r = intti::ri_j_deriv_build(orb, aux, reqs, grid, tau_lin, 0, grp);
    for (int x = 0; x < 3 * ngrp; ++x)
      for (int i = 0; i < nao; ++i)
        for (int j = 0; j < nao; ++j) {
          const std::size_t o = static_cast<std::size_t>(x) * N + i * nao + j;
          vj[o] = r.J[0][o] * oscale[i] * oscale[j];
        }
  }
  if (vk) {
    if (!cocc || nvec < 1) return -7;
    std::vector<double> Cs(static_cast<std::size_t>(nao) * nvec);
    for (int i = 0; i < nao; ++i)
      for (int k = 0; k < nvec; ++k)
        Cs[static_cast<std::size_t>(i) * nvec + k] =
            cocc[static_cast<std::size_t>(i) * nvec + k] * oscale[i];
    const auto r = intti::ri_k_deriv_occ(orb, aux, Cs.data(), Cs.data(), nvec, grid,
                                         tau_lin, 0, grp);
    for (int x = 0; x < 3 * ngrp; ++x)
      for (int i = 0; i < nao; ++i)
        for (int j = 0; j < nao; ++j) {
          const std::size_t o = static_cast<std::size_t>(x) * N + i * nao + j;
          vk[o] = r.K[0][o] * oscale[i] * oscale[j];
        }
  }
  return 0;
}


// ---------------------------------------------------------------------------
// RI (density-fitted) J/K, as an explicitly CACHED handle.
//
// ri_fit builds B (nao^2 x naux) once and every subsequent ri_jk is only GEMMs.
// That is the whole point: J/K is asked for once per SCF ITERATION, so caching
// wins outright -- replacing ri_fit + ri_jk with the tiled builders in the
// capstone RI-RHF measured 34 ms -> 570 s, because each iteration then pays a
// full integral pass instead of a GEMM (ri.hpp records the criterion).
//
// So the facade does NOT offer a one-shot intti_ri_get_jk(mol, dm): that
// signature would look convenient and rebuild the fit on every call, hiding a
// four-order-of-magnitude cost in an innocuous-looking function. The handle
// makes the caching part of the interface, and its lifetime part of the
// caller's job. B is nao^2 x naux -- 32 GB at nao = 1000, naux = 4000 -- so
// when it does not fit, the tiled builders are the answer, not this.
//
// Densities need not be symmetric: ri_jk contracts K = sum_P B^P D B^P as two
// GEMMs and assumes nothing about D, so the general and antisymmetric densities
// that response theory produces are served exactly.
namespace {
struct RIHandle {
  intti::RIFit<double> fit;
  std::vector<double> scale; // per-AO PySCF <-> intti conversion
  int nao{0};
};
} // namespace

extern "C" void *intti_ri_open(const int *atm, int natm, const int *bas, int nbas,
                               const double *env, const int *aatm, int anatm,
                               const int *abas, int anbas, const double *aenv,
                               double tau_lin) {
  ensure_kokkos();
  (void)natm;
  (void)anatm;
  auto build = [](const int *a, const int *b, int nb, const double *e,
                  std::vector<double> &scale, intti::ShellBasis<double> &out) {
    std::vector<intti::PrimitiveShell<double>> shells;
    for (int ish = 0; ish < nb; ++ish) {
      const ShellInfo s = decode_shell(ish, a, b, e);
      if (s.nctr != 1 || s.nprim != 1) return -1;
      shells.push_back(intti::PrimitiveShell<double>{
          s.alpha[0], {s.center[0], s.center[1], s.center[2]}, s.l});
      const double c = s.coeff[0] * coeff_rescale(s.l);
      for (int k = 0; k < intti::ncart(s.l); ++k) scale.push_back(c);
    }
    out = intti::make_basis(shells);
    return 0;
  };
  auto is_contracted = [](const int *b, int nb) {
    for (int ish = 0; ish < nb; ++ish) {
      const int *r = b + static_cast<std::size_t>(ish) * 8;
      if (r[NPRIM_OF] != 1 || r[NCTR_OF] != 1) return true;
    }
    return false;
  };
  // Either basis may be contracted independently of the other -- and usually is:
  // a Coulomb-fitting auxiliary set is far less contracted than the orbital set.
  // The contracted three-centre builder wants a contracted PAIR, so an
  // uncontracted partner is wrapped as trivially contracted (one primitive per
  // shell) rather than duplicating the whole path for the mixed case.
  const bool cc = is_contracted(bas, nbas) || is_contracted(abas, anbas);
  auto *h = new RIHandle;
  if (cc) {
    const auto co = contracted_basis_from(atm, bas, nbas, env);
    const auto cav = contracted_basis_from(aatm, abas, anbas, aenv);
    h->nao = co.nao;
    h->scale.assign(co.nao, 1.0); // normalization is already in the coefficients
    h->fit = intti::ri_fit(co, cav, default_grid(), tau_lin);
    return h;
  }
  std::vector<double> oscale, ascale;
  intti::ShellBasis<double> orb, aux;
  if (build(atm, bas, nbas, env, oscale, orb) != 0) {
    delete h;
    return nullptr;
  }
  if (build(aatm, abas, anbas, aenv, ascale, aux) != 0) {
    delete h;
    return nullptr;
  }
  h->nao = orb.nao;
  h->scale = std::move(oscale);
  h->fit = intti::ri_fit(orb, aux, default_grid(), tau_lin);
  return h;
}

extern "C" void intti_ri_close(void *handle) { delete static_cast<RIHandle *>(handle); }

extern "C" int intti_ri_get_jk(void *handle, double *vj, double *vk, const double *dms,
                               int ndm, int with_j, int with_k) {
  auto *h = static_cast<RIHandle *>(handle);
  if (!h) return -1;
  const int nao = h->nao;
  const std::size_t N = static_cast<std::size_t>(nao) * nao;
  std::vector<double> Ds(N), J, K;
  if (with_j) J.assign(N, 0.0);
  if (with_k) K.assign(N, 0.0);
  for (int d = 0; d < ndm; ++d) {
    const double *D = dms + static_cast<std::size_t>(d) * N;
    for (int i = 0; i < nao; ++i)
      for (int j = 0; j < nao; ++j)
        Ds[static_cast<std::size_t>(i) * nao + j] =
            D[static_cast<std::size_t>(i) * nao + j] * h->scale[i] * h->scale[j];
    if (with_j) std::fill(J.begin(), J.end(), 0.0);
    if (with_k) std::fill(K.begin(), K.end(), 0.0);
    intti::ri_jk(h->fit, Ds.data(), with_j ? J.data() : nullptr,
                 with_k ? K.data() : nullptr);
    for (int i = 0; i < nao; ++i)
      for (int j = 0; j < nao; ++j) {
        const std::size_t o = static_cast<std::size_t>(i) * nao + j;
        const double sc = h->scale[i] * h->scale[j];
        if (with_j && vj) vj[d * N + o] = J[o] * sc;
        if (with_k && vk) vk[d * N + o] = K[o] * sc;
      }
  }
  return 0;
}


// Two- and three-centre Coulomb TENSORS, for validation against libcint's
// int2c2e / int3c2e.
//
// These are whole-tensor quantities -- (naux x naux) and (nao x nao x naux) --
// not a per-quartet surface, so they sit inside the matrix-level API. They earn
// their place: the contracted two-centre metric carried a silent double-count
// for l >= 2 that the RI energy could only reveal indirectly, because the fit
// T M^-1 T^T is invariant to any diagonal rescaling of the auxiliary AOs and
// the unit tests compared contracted against contracted. A direct per-builder
// oracle localises such an error instead of requiring a bisection.
//
// Cartesian only; either basis may be generally contracted.
extern "C" int intti_coulomb_2c(double *out, const int *aatm, int anatm, const int *abas,
                                int anbas, const double *aenv) {
  ensure_kokkos();
  (void)anatm;
  bool contracted = false;
  for (int ish = 0; ish < anbas; ++ish) {
    const int *b = abas + static_cast<std::size_t>(ish) * 8;
    if (b[NPRIM_OF] != 1 || b[NCTR_OF] != 1) contracted = true;
  }
  const auto &grid = default_grid();
  if (contracted) {
    const auto ca = contracted_basis_from(aatm, abas, anbas, aenv);
    const auto M = intti::coulomb_2c(ca, grid);
    for (std::size_t i = 0; i < M.size(); ++i) out[i] = M[i];
    return 0;
  }
  std::vector<intti::PrimitiveShell<double>> shells;
  std::vector<double> scale;
  for (int ish = 0; ish < anbas; ++ish) {
    const ShellInfo s = decode_shell(ish, aatm, abas, aenv);
    shells.push_back(intti::PrimitiveShell<double>{
        s.alpha[0], {s.center[0], s.center[1], s.center[2]}, s.l});
    const double c = s.coeff[0] * coeff_rescale(s.l);
    for (int k = 0; k < intti::ncart(s.l); ++k) scale.push_back(c);
  }
  auto aux = intti::make_basis(shells);
  const int n = aux.nao;
  const auto M = intti::coulomb_2c(aux, grid);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      out[static_cast<std::size_t>(i) * n + j] =
          M[static_cast<std::size_t>(i) * n + j] * scale[i] * scale[j];
  return 0;
}

extern "C" int intti_coulomb_3c(double *out, const int *atm, int natm, const int *bas,
                                int nbas, const double *env, const int *aatm, int anatm,
                                const int *abas, int anbas, const double *aenv) {
  ensure_kokkos();
  (void)natm;
  (void)anatm;
  auto has_contraction = [](const int *b, int nb) {
    for (int ish = 0; ish < nb; ++ish) {
      const int *r = b + static_cast<std::size_t>(ish) * 8;
      if (r[NPRIM_OF] != 1 || r[NCTR_OF] != 1) return true;
    }
    return false;
  };
  const auto &grid = default_grid();
  if (has_contraction(bas, nbas) || has_contraction(abas, anbas)) {
    const auto co = contracted_basis_from(atm, bas, nbas, env);
    const auto ca = contracted_basis_from(aatm, abas, anbas, aenv);
    const auto T = intti::coulomb_3c(co, ca, grid);
    for (std::size_t i = 0; i < T.size(); ++i) out[i] = T[i];
    return 0;
  }
  auto build = [](const int *a, const int *b, int nb, const double *e,
                  std::vector<double> &scale, intti::ShellBasis<double> &basis) {
    std::vector<intti::PrimitiveShell<double>> shells;
    for (int ish = 0; ish < nb; ++ish) {
      const ShellInfo s = decode_shell(ish, a, b, e);
      shells.push_back(intti::PrimitiveShell<double>{
          s.alpha[0], {s.center[0], s.center[1], s.center[2]}, s.l});
      const double c = s.coeff[0] * coeff_rescale(s.l);
      for (int k = 0; k < intti::ncart(s.l); ++k) scale.push_back(c);
    }
    basis = intti::make_basis(shells);
  };
  std::vector<double> oscale, ascale;
  intti::ShellBasis<double> orb, aux;
  build(atm, bas, nbas, env, oscale, orb);
  build(aatm, abas, anbas, aenv, ascale, aux);
  const int nao = orb.nao, naux = aux.nao;
  const auto T = intti::coulomb_3c(orb, aux, grid);
  for (int i = 0; i < nao; ++i)
    for (int j = 0; j < nao; ++j)
      for (int P = 0; P < naux; ++P) {
        const std::size_t o = (static_cast<std::size_t>(i) * nao + j) * naux + P;
        out[o] = T[o] * oscale[i] * oscale[j] * ascale[P];
      }
  return 0;
}


// Real -> complex spherical harmonics (Condon-Shortley), as a boundary
// transform on a whole AO matrix. `M` is nao x nao in the REAL spherical basis
// with shells ordered m = -l..+l; the result is written as separate real and
// imaginary parts, since the C ABI here is double-only.
//
// This is not a second integral path: within a shell the two conventions differ
// by a fixed unitary depending only on l, so the engine stays real Cartesian and
// only the basis the caller sees changes. Its use is where m has to be a good
// quantum number -- magnetic properties, spin-orbit -- because L_z is diagonal
// in the complex basis and is not in the real one.
extern "C" int intti_real_to_complex(double *re, double *im, const double *M,
                                     const int *bas, int nbas, int libcint_order) {
  if (!re || !im || !M || !bas) return -7;
  std::vector<int> ls(nbas), off(nbas);
  int nao = 0;
  for (int ish = 0; ish < nbas; ++ish) {
    const int l = bas[static_cast<std::size_t>(ish) * 8 + ANG_OF];
    const int nctr = bas[static_cast<std::size_t>(ish) * 8 + NCTR_OF];
    if (nctr != 1) return -1; // one l block per shell entry here
    ls[ish] = l;
    off[ish] = nao;
    nao += 2 * l + 1;
  }
  const auto C = intti::real_to_complex(
      ls, off, nao, M,
      libcint_order ? intti::RealOrder::Libcint : intti::RealOrder::Standard);
  for (std::size_t i = 0; i < C.size(); ++i) {
    re[i] = C[i].real();
    im[i] = C[i].imag();
  }
  return 0;
}

extern "C" int intti_int2e_sph(double *out, const int *shls, const int *atm, int natm,
                                const int *bas, int nbas, const double *env, void * /*opt*/,
                                double * /*cache*/) {
  return eval_int2e(out, shls, atm, natm, bas, nbas, env, true);
}


// Build a ContractedBasis from libcint's atm/bas/env, in PySCF's cart=True
// convention.
//
// env's contraction coefficients already carry PySCF's internal gto_norm; the
// contracted builders instead apply cart_norm_pyscf(l, alpha) to each primitive
// (detail::effective_coeff). Absorbing the difference into the shell
// coefficients here,
//     coeff[c][p] = env_coeff[c][p] * coeff_rescale(l) / cart_norm_pyscf(l, a_p)
// makes intti's contracted AO EQUAL PySCF's, so no output rescaling is needed at
// all -- unlike the primitive path, which carries a per-AO diagonal through the
// density and back. Same conversion factor, applied once and in the right place.
intti::ContractedBasis<double> contracted_basis_from(const int *atm, const int *bas,
                                                     int nbas, const double *env) {
  std::vector<intti::ContractedShell<double>> shells;
  shells.reserve(nbas);
  for (int ish = 0; ish < nbas; ++ish) {
    const ShellInfo si = decode_shell(ish, atm, bas, env);
    intti::ContractedShell<double> sh;
    sh.l = si.l;
    for (int d = 0; d < 3; ++d) sh.center[d] = si.center[d];
    sh.alpha.assign(si.alpha, si.alpha + si.nprim);
    sh.coeff.resize(static_cast<std::size_t>(si.nctr) * si.nprim);
    const double rs = coeff_rescale(si.l);
    for (int c = 0; c < si.nctr; ++c)
      for (int p = 0; p < si.nprim; ++p)
        sh.coeff[static_cast<std::size_t>(c) * si.nprim + p] =
            si.coeff[static_cast<std::size_t>(c) * si.nprim + p] * rs /
            intti::cart_norm_pyscf(si.l, si.alpha[p]);
    shells.push_back(std::move(sh));
  }
  return intti::make_contracted_basis(std::move(shells));
}

// ---------------------------------------------------------------------------
// Matrix-level entry point: PySCF's get_jk, served by intti's J/K builders.
//
// This is the surface an SCF driver should plug into, NOT intti_int2e_*. The
// per-quartet entries above exist for libcint drop-in compatibility and for
// validation; driving an SCF through them would hand the device one quartet at
// a time, which is exactly what the matrix-level API exists to prevent.
//
// The signature mirrors pyscf.scf.hf.SCF.get_jk(mol, dm, hermi, with_j, with_k,
// omega), and the mapping is close to one-to-one because both interfaces were
// designed for the same job:
//     dm list        -> a vector of JKRequest
//     hermi 0/1      -> DensitySymmetry::General / Symmetric
//     with_j/with_k  -> FockTerms
//     omega          -> the range-separated kernel on the t grid
// hermi = 2 (anti-hermitian, as PySCF uses for some response densities) maps to
// DensitySymmetry::Antisymmetric.
//
// General contraction (nctr > 1, nprim > 1) is served natively by the
// contraction-aware builders, at every density symmetry: the contracted
// exchange runs the full ordered primitive (a,b) loop with atomic accumulation
// and no mirror, so it needs no hermiticity assumption, and the contracted
// Coulomb folds D + D^T, which is exact because J sees only the symmetric part
// of D. That matters because it is exactly the case magnetic response needs --
// pyscf/prop/nmr/rhf.py drives its response with an antisymmetric
// dm1 = d1 - d1^H, on a contracted basis.
//
// RESTRICTIONS, checked and reported rather than silently mis-answered:
//   * Cartesian only (mol.cart = True). The spherical transform is available in
//     the library but is not applied here.
// Returns 0 on success, negative on a violated restriction.
//
// Convention. If PySCF's AO is chi^p_mu = s_mu chi^ours_mu, then
// (mn|ls)^p = s_m s_n s_l s_s (mn|ls)^ours, so scaling D on the way in and J/K
// on the way out by the same diagonal is exact -- the factors are the env
// contraction coefficient times the gto_norm -> cart_norm_pyscf correction
// already used by the quartet path.
extern "C" int intti_get_jk(double *vj, double *vk, const double *dms, int ndm,
                            const int *hermi, int with_j, int with_k, const int *atm,
                            int natm, const int *bas, int nbas, const double *env,
                            double omega, double tau) {
  ensure_kokkos();
  // Contracted basis: route to the contraction-aware builders. They share each
  // primitive intermediate across every contracted function and Cartesian
  // component, so this is not a decontract-recontract -- the contraction enters
  // only in the density fold and the output gather.
  bool contracted = false;
  for (int ish = 0; ish < nbas; ++ish) {
    const int *b = bas + static_cast<std::size_t>(ish) * 8;
    if (b[NPRIM_OF] != 1 || b[NCTR_OF] != 1) contracted = true;
  }
  if (contracted) {
    const intti::TGrid<double> cgrid =
        (omega == 0.0) ? default_grid() : intti::make_tgrid(intti::erf_rs<double>(omega));
    const auto cbasis = contracted_basis_from(atm, bas, nbas, env);
    const int cnao = cbasis.nao;
    const std::size_t cn2 = static_cast<std::size_t>(cnao) * cnao;
    // No symmetry restriction here, unlike the primitive fused engines. The
    // contracted exchange runs the FULL primitive (a,b) loop with atomic
    // accumulation and no mirror (kbuild.hpp), so it is already correct for
    // general and antisymmetric densities -- verified against the primitive
    // general path. hermi is therefore only a hint and is not needed.
    for (int d = 0; d < ndm; ++d) {
      if (with_j && vj) {
        std::vector<double> J(cn2, 0.0);
        intti::coulomb_build(cbasis, dms + static_cast<std::size_t>(d) * cn2, cgrid,
                             J.data(), tau);
        for (std::size_t i = 0; i < cn2; ++i) vj[d * cn2 + i] = J[i];
      }
      if (with_k && vk) {
        std::vector<double> K(cn2, 0.0);
        intti::exchange_build(cbasis, dms + static_cast<std::size_t>(d) * cn2, cgrid,
                              K.data(), tau);
        for (std::size_t i = 0; i < cn2; ++i) vk[d * cn2 + i] = K[i];
      }
    }
    return 0;
  }
  std::vector<intti::PrimitiveShell<double>> shells;
  std::vector<double> scale; // per-AO PySCF/intti conversion
  for (int ish = 0; ish < nbas; ++ish) {
    const ShellInfo s = decode_shell(ish, atm, bas, env);
    if (s.nctr != 1 || s.nprim != 1) return -1; // contracted: not supported here
    shells.push_back(intti::PrimitiveShell<double>{
        s.alpha[0], {s.center[0], s.center[1], s.center[2]}, s.l});
    const double c = s.coeff[0] * coeff_rescale(s.l);
    for (int k = 0; k < intti::ncart(s.l); ++k) scale.push_back(c);
  }
  auto basis = intti::make_basis(shells);
  const int nao = basis.nao;
  const std::size_t N = static_cast<std::size_t>(nao) * nao;
  if (static_cast<int>(scale.size()) != nao) return -2;

  const intti::TGrid<double> grid =
      (omega == 0.0) ? default_grid()
                     : intti::make_tgrid(intti::erf_rs<double>(omega));

  // scale the incoming densities into intti's AO convention
  std::vector<std::vector<double>> Dscaled(ndm);
  std::vector<intti::JKRequest<double>> reqs(ndm);
  for (int d = 0; d < ndm; ++d) {
    Dscaled[d].resize(N);
    for (int i = 0; i < nao; ++i)
      for (int j = 0; j < nao; ++j)
        Dscaled[d][static_cast<std::size_t>(i) * nao + j] =
            dms[static_cast<std::size_t>(d) * N + i * nao + j] * scale[i] * scale[j];
    const int h = hermi ? hermi[d] : 1;
    reqs[d].D = Dscaled[d].data();
    reqs[d].sym = (h == 1)   ? intti::DensitySymmetry::Symmetric
                  : (h == 2) ? intti::DensitySymmetry::Antisymmetric
                             : intti::DensitySymmetry::General;
    reqs[d].terms = (with_j && with_k) ? intti::FockTerms::CoulombExchange
                    : with_j           ? intti::FockTerms::Coulomb
                                       : intti::FockTerms::Exchange;
  }
  const auto res = intti::jk_build(basis, reqs, grid, tau);
  for (int d = 0; d < ndm; ++d) {
    if (with_j && vj)
      for (int i = 0; i < nao; ++i)
        for (int j = 0; j < nao; ++j)
          vj[static_cast<std::size_t>(d) * N + i * nao + j] =
              res.J[d][static_cast<std::size_t>(i) * nao + j] * scale[i] * scale[j];
    if (with_k && vk)
      for (int i = 0; i < nao; ++i)
        for (int j = 0; j < nao; ++j)
          vk[static_cast<std::size_t>(d) * N + i * nao + j] =
              res.K[d][static_cast<std::size_t>(i) * nao + j] * scale[i] * scale[j];
  }
  return 0;
}

// Derivative J/K in the bra-gradient convention, mirroring
// pyscf.grad.rhf.get_jk(mol, dm): vj/vk are 3 x nao x nao, the derivative acting
// on the FIRST AO index only, unfolded onto atoms (the caller does that with its
// own aoslices). Same restrictions and conventions as intti_get_jk.
extern "C" int intti_get_jk_ip1(double *vj, double *vk, const double *dm, const int *atm,
                                int natm, const int *bas, int nbas, const double *env,
                                double tau) {
  ensure_kokkos();
  // Contracted basis: the fan-out route (jk.hpp). The primitive quartet
  // derivatives are evaluated once and scattered coefficient-weighted into
  // every contracted index combination, so a generally-contracted gradient
  // costs no more integrals than a segmented one. The density is taken as
  // General -- PySCF's grad.rhf.get_jk convention transposes the ket indices,
  // which is only immaterial for a symmetric D, and CPHF supplies densities
  // that are not.
  bool contracted = false;
  for (int ish = 0; ish < nbas; ++ish) {
    const int *b = bas + static_cast<std::size_t>(ish) * 8;
    if (b[NPRIM_OF] != 1 || b[NCTR_OF] != 1) contracted = true;
  }
  if (contracted) {
    const auto cbasis = contracted_basis_from(atm, bas, nbas, env);
    const std::size_t cN = static_cast<std::size_t>(cbasis.nao) * cbasis.nao;
    std::vector<intti::JKRequest<double>> creq(1);
    creq[0].D = dm;
    creq[0].sym = intti::DensitySymmetry::General;
    creq[0].terms = intti::FockTerms::CoulombExchange;
    const auto cres = intti::jk_deriv_ao_build(cbasis, creq, default_grid(), tau);
    for (std::size_t o = 0; o < 3 * cN; ++o) {
      if (vj) vj[o] = cres.J[0][o];
      if (vk) vk[o] = cres.K[0][o];
    }
    return 0;
  }
  std::vector<intti::PrimitiveShell<double>> shells;
  std::vector<double> scale;
  for (int ish = 0; ish < nbas; ++ish) {
    const ShellInfo s = decode_shell(ish, atm, bas, env);
    if (s.nctr != 1 || s.nprim != 1) return -1;
    shells.push_back(intti::PrimitiveShell<double>{
        s.alpha[0], {s.center[0], s.center[1], s.center[2]}, s.l});
    const double c = s.coeff[0] * coeff_rescale(s.l);
    for (int k = 0; k < intti::ncart(s.l); ++k) scale.push_back(c);
  }
  auto basis = intti::make_basis(shells);
  const int nao = basis.nao;
  const std::size_t N = static_cast<std::size_t>(nao) * nao;
  if (static_cast<int>(scale.size()) != nao) return -2;
  std::vector<double> Ds(N);
  for (int i = 0; i < nao; ++i)
    for (int j = 0; j < nao; ++j)
      Ds[static_cast<std::size_t>(i) * nao + j] =
          dm[static_cast<std::size_t>(i) * nao + j] * scale[i] * scale[j];
  std::vector<intti::JKRequest<double>> reqs(1);
  reqs[0].D = Ds.data();
  reqs[0].sym = intti::DensitySymmetry::General;
  reqs[0].terms = intti::FockTerms::CoulombExchange;
  const auto res = intti::jk_deriv_ao_build(basis, reqs, default_grid(), tau);
  for (int x = 0; x < 3; ++x)
    for (int i = 0; i < nao; ++i)
      for (int j = 0; j < nao; ++j) {
        const std::size_t o = static_cast<std::size_t>(x) * N + i * nao + j;
        if (vj) vj[o] = res.J[0][o] * scale[i] * scale[j];
        if (vk) vk[o] = res.K[0][o] * scale[i] * scale[j];
      }
  return 0;
}

// Skeleton (fixed-density) electronic Hessian: the quantity
// pyscf.hessian.rhf.partial_hess_elec computes, namely
//   sum_mn D_mn d2h_mn + d2E_2e - sum_mn W_mn d2S_mn
// contracted with the density D and the energy-weighted density W. Returned as
// (3 nbas) x (3 nbas), row-major, indexed by SHELL centre; the caller folds
// shells onto atoms with bas[:,ATOM_OF], as it already does for the gradient.
//
// This is a contracted INTEGRAL quantity, not SCF machinery: the response terms
// that need CPHF are PySCF's business and are added by hess_elec on top. The
// nuclear repulsion Hessian is likewise PySCF's (hess_nuc) and is NOT included.
//
// Cartesian basis only; generally-contracted shells are served natively.
extern "C" int intti_hess_skeleton(double *hess, const double *dm, const double *W,
                                   const int *atm, int natm, const int *bas, int nbas,
                                   const double *env, double tau) {
  ensure_kokkos();
  // The output is indexed by SHELL centre, and for a contracted basis that must
  // stay the CONTRACTED shell -- moving a contracted shell's centre moves all
  // its primitives together, and the caller folds shells onto atoms with
  // bas[:,ATOM_OF]. The library's contracted builders keep that indexing.
  bool contracted = false;
  for (int ish = 0; ish < nbas; ++ish) {
    const int *b = bas + static_cast<std::size_t>(ish) * 8;
    if (b[NPRIM_OF] != 1 || b[NCTR_OF] != 1) contracted = true;
  }
  if (contracted) {
    const auto cbasis = contracted_basis_from(atm, bas, nbas, env);
    std::vector<intti::PointCharge<double>> cch;
    std::vector<int> cch_shell;
    for (int a = 0; a < natm; ++a) {
      const int *ai = atm + static_cast<std::size_t>(a) * 6;
      const double *c = env + ai[PTR_COORD];
      const double Z = static_cast<double>(ai[CHARGE_OF]);
      if (Z == 0.0) continue;
      int home = -1;
      for (int ish = 0; ish < nbas && home < 0; ++ish)
        if (bas[static_cast<std::size_t>(ish) * 8 + ATOM_OF] == a) home = ish;
      if (home < 0) return -3; // a charge with no basis shell on it
      cch.push_back(intti::PointCharge<double>{-Z, {c[0], c[1], c[2]}});
      cch_shell.push_back(home);
    }
    const auto &cgrid = default_grid();
    const auto cHs = intti::overlap_hessian(cbasis, W);
    const auto cHk = intti::kinetic_hessian(cbasis, dm);
    const auto cHv =
        intti::nuclear_attraction_hessian(cbasis, cch, cgrid, cch_shell, dm);
    const auto cH2 = intti::two_electron_hessian(cbasis, dm, cgrid, tau);
    const std::size_t cd = static_cast<std::size_t>(3) * nbas;
    for (std::size_t i = 0; i < cd * cd; ++i)
      hess[i] = cHk[i] + cHv[i] - cHs[i] + cH2[i];
    return 0;
  }
  std::vector<intti::PrimitiveShell<double>> shells;
  std::vector<double> scale;
  std::vector<int> shell_atom;
  for (int ish = 0; ish < nbas; ++ish) {
    const ShellInfo s = decode_shell(ish, atm, bas, env);
    if (s.nctr != 1 || s.nprim != 1) return -1;
    shells.push_back(intti::PrimitiveShell<double>{
        s.alpha[0], {s.center[0], s.center[1], s.center[2]}, s.l});
    shell_atom.push_back(bas[static_cast<std::size_t>(ish) * 8 + ATOM_OF]);
    const double c = s.coeff[0] * coeff_rescale(s.l);
    for (int k = 0; k < intti::ncart(s.l); ++k) scale.push_back(c);
  }
  auto basis = intti::make_basis(shells);
  const int nao = basis.nao, ns = nbas, dim = 3 * ns;
  const std::size_t N = static_cast<std::size_t>(nao) * nao;
  if (static_cast<int>(scale.size()) != nao) return -2;

  // scale the incoming matrices into intti's AO convention
  std::vector<double> Ds(N), Ws(N);
  for (int i = 0; i < nao; ++i)
    for (int j = 0; j < nao; ++j) {
      const std::size_t o = static_cast<std::size_t>(i) * nao + j;
      Ds[o] = dm[o] * scale[i] * scale[j];
      Ws[o] = W[o] * scale[i] * scale[j];
    }
  // nuclei, and for each the shell whose centre carries it (differentiating
  // that shell moves the nucleus, which is what nuclear_attraction_hessian
  // needs to attribute the Hellmann-Feynman second derivative correctly)
  std::vector<intti::PointCharge<double>> charges;
  std::vector<int> charge_shell;
  for (int a = 0; a < natm; ++a) {
    const int *ai = atm + static_cast<std::size_t>(a) * 6;
    const double *c = env + ai[PTR_COORD];
    const double Z = static_cast<double>(ai[CHARGE_OF]);
    if (Z == 0.0) continue;
    int home = -1;
    for (int ish = 0; ish < nbas && home < 0; ++ish)
      if (shell_atom[ish] == a) home = ish;
    if (home < 0) return -3; // a charge with no basis shell on it
    charges.push_back(intti::PointCharge<double>{-Z, {c[0], c[1], c[2]}});
    charge_shell.push_back(home);
  }
  const auto &grid = default_grid();
  const auto Hs = intti::overlap_hessian(basis, Ws.data());
  const auto Hk = intti::kinetic_hessian(basis, Ds.data());
  const auto Hv = intti::nuclear_attraction_hessian(basis, charges, grid, charge_shell,
                                                    Ds.data());
  const auto H2 = intti::two_electron_hessian(basis, Ds.data(), grid, tau);
  for (std::size_t i = 0; i < static_cast<std::size_t>(dim) * dim; ++i)
    hess[i] = Hk[i] + Hv[i] - Hs[i] + H2[i];
  return 0;
}

// The four int2e_ip1 contractions pyscf.hessian.rhf.make_h1 needs, for one
// atom's shell slice [shl0, shl1). Each output is 3 x nao x nao. See
// intti::ip1_h1_contractions for the scripts and the sign convention (PySCF
// contracts against MINUS the density with nabla; we use d/dA with the positive
// density, and the two conventions cancel).
extern "C" int intti_ip1_h1_jk(double *vj1, double *vj2, double *vk1, double *vk2,
                               const double *dm, int shl0, int shl1, const int *atm,
                               int natm, const int *bas, int nbas, const double *env,
                               double tau) {
  ensure_kokkos();
  // Contracted basis: the same digest fan-out as intti_get_jk_ip1. [shl0, shl1)
  // stays a CONTRACTED shell range (PySCF's aoslice_by_atom), translated inside.
  bool contracted = false;
  for (int ish = 0; ish < nbas; ++ish) {
    const int *b = bas + static_cast<std::size_t>(ish) * 8;
    if (b[NPRIM_OF] != 1 || b[NCTR_OF] != 1) contracted = true;
  }
  if (contracted) {
    const auto cbasis = contracted_basis_from(atm, bas, nbas, env);
    const std::size_t cN = static_cast<std::size_t>(cbasis.nao) * cbasis.nao;
    const auto cr =
        intti::ip1_h1_contractions(cbasis, dm, shl0, shl1, default_grid(), tau);
    double *couts[4] = {vj1, vj2, vk1, vk2};
    const std::vector<double> *csrcs[4] = {&cr.vj1, &cr.vj2, &cr.vk1, &cr.vk2};
    for (int w = 0; w < 4; ++w)
      if (couts[w])
        for (std::size_t o = 0; o < 3 * cN; ++o) couts[w][o] = (*csrcs[w])[o];
    return 0;
  }
  std::vector<intti::PrimitiveShell<double>> shells;
  std::vector<double> scale;
  for (int ish = 0; ish < nbas; ++ish) {
    const ShellInfo s = decode_shell(ish, atm, bas, env);
    if (s.nctr != 1 || s.nprim != 1) return -1;
    shells.push_back(intti::PrimitiveShell<double>{
        s.alpha[0], {s.center[0], s.center[1], s.center[2]}, s.l});
    const double c = s.coeff[0] * coeff_rescale(s.l);
    for (int k = 0; k < intti::ncart(s.l); ++k) scale.push_back(c);
  }
  auto basis = intti::make_basis(shells);
  const int nao = basis.nao;
  const std::size_t N = static_cast<std::size_t>(nao) * nao;
  if (static_cast<int>(scale.size()) != nao) return -2;
  std::vector<double> Ds(N);
  for (int i = 0; i < nao; ++i)
    for (int j = 0; j < nao; ++j)
      Ds[static_cast<std::size_t>(i) * nao + j] =
          dm[static_cast<std::size_t>(i) * nao + j] * scale[i] * scale[j];
  const auto r = intti::ip1_h1_contractions(basis, Ds.data(), shl0, shl1, default_grid(), tau);
  double *outs[4] = {vj1, vj2, vk1, vk2};
  const std::vector<double> *srcs[4] = {&r.vj1, &r.vj2, &r.vk1, &r.vk2};
  for (int w = 0; w < 4; ++w) {
    if (!outs[w]) continue;
    for (int x = 0; x < 3; ++x)
      for (int i = 0; i < nao; ++i)
        for (int j = 0; j < nao; ++j) {
          const std::size_t o = static_cast<std::size_t>(x) * N + i * nao + j;
          outs[w][o] = (*srcs[w])[o] * scale[i] * scale[j];
        }
  }
  return 0;
}

// One-electron derivative matrices in PySCF's gradient conventions, all
// 3 x nao x nao. `which` selects:
//   0 = int1e_ipovlp   <nabla mu|nu>
//   1 = int1e_ipkin    <nabla mu|T|nu>
//   2 = int1e_ipnuc    <nabla mu| sum_A -Z_A/r_A |nu>   (all nuclei)
//   3 = int1e_iprinv   <nabla mu| 1/r_C |nu> for the single nucleus `iatm`,
//                      UNWEIGHTED (the caller applies -Z, as PySCF does)
// Same restrictions as intti_get_jk.
extern "C" int intti_int1e_ip(double *out, int which, int iatm, const int *atm, int natm,
                              const int *bas, int nbas, const double *env) {
  ensure_kokkos();
  // Charges for the nuclear-attraction variants, shared by both basis routes.
  std::vector<intti::PointCharge<double>> charges;
  if (which == 2) {
    for (int a = 0; a < natm; ++a) {
      const int *ai = atm + static_cast<std::size_t>(a) * 6;
      const double *c = env + ai[PTR_COORD];
      charges.push_back({-static_cast<double>(ai[CHARGE_OF]), {c[0], c[1], c[2]}});
    }
  } else if (which == 3) {
    if (iatm < 0 || iatm >= natm) return -4;
    const int *ai = atm + static_cast<std::size_t>(iatm) * 6;
    const double *c = env + ai[PTR_COORD];
    charges.push_back({1.0, {c[0], c[1], c[2]}}); // unweighted: caller applies -Z
  } else if (which != 0 && which != 1) {
    return -5;
  }
  // Contracted basis: the contraction-aware derivative builders (contracted.hpp)
  // share each primitive-pair shift block across every contracted-function pair.
  // Without this branch mol.intor interception worked only on an uncontracted
  // basis, which is to say on no standard basis set -- PySCF's own gradient and
  // Hessian code could not run on intti integrals for a real calculation.
  // contracted_basis_from folds the PySCF normalization into the coefficients,
  // so no per-AO rescale is applied on the way out.
  bool contracted = false;
  for (int ish = 0; ish < nbas; ++ish) {
    const int *b = bas + static_cast<std::size_t>(ish) * 8;
    if (b[NPRIM_OF] != 1 || b[NCTR_OF] != 1) contracted = true;
  }
  if (contracted) {
    const auto cbasis = contracted_basis_from(atm, bas, nbas, env);
    const std::size_t cN = static_cast<std::size_t>(cbasis.nao) * cbasis.nao;
    std::array<std::vector<double>, 3> Gc;
    if (which == 0)
      Gc = intti::overlap_deriv(cbasis);
    else if (which == 1)
      Gc = intti::kinetic_deriv(cbasis);
    else
      Gc = intti::nuclear_deriv(cbasis, charges, default_grid());
    for (int x = 0; x < 3; ++x)
      for (std::size_t o = 0; o < cN; ++o)
        out[static_cast<std::size_t>(x) * cN + o] = Gc[x][o];
    return 0;
  }
  std::vector<intti::PrimitiveShell<double>> shells;
  std::vector<double> scale;
  for (int ish = 0; ish < nbas; ++ish) {
    const ShellInfo s = decode_shell(ish, atm, bas, env);
    if (s.nctr != 1 || s.nprim != 1) return -1;
    shells.push_back(intti::PrimitiveShell<double>{
        s.alpha[0], {s.center[0], s.center[1], s.center[2]}, s.l});
    const double c = s.coeff[0] * coeff_rescale(s.l);
    for (int k = 0; k < intti::ncart(s.l); ++k) scale.push_back(c);
  }
  auto basis = intti::make_basis(shells);
  const int nao = basis.nao;
  const std::size_t N = static_cast<std::size_t>(nao) * nao;
  if (static_cast<int>(scale.size()) != nao) return -2;
  std::array<std::vector<double>, 3> G;
  if (which == 0)
    G = intti::overlap_deriv(basis);
  else if (which == 1)
    G = intti::kinetic_deriv(basis);
  else
    G = intti::nuclear_deriv(basis, charges, default_grid());
  for (int x = 0; x < 3; ++x)
    for (int i = 0; i < nao; ++i)
      for (int j = 0; j < nao; ++j) {
        const std::size_t o = static_cast<std::size_t>(i) * nao + j;
        out[static_cast<std::size_t>(x) * N + o] = G[x][o] * scale[i] * scale[j];
      }
  return 0;
}
