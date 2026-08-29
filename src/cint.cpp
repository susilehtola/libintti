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

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "intti/batch.hpp"
#include "intti/c2s.hpp"
#include "intti/gto.hpp"
#include "intti/kernel.hpp"
#include "intti/math.hpp"
#include "intti/normalization.hpp"
#include "intti/tgrid.hpp"

namespace {

// libcint bas[] column indices (bas[ish*8 + slot]).
enum BasSlot { ATOM_OF = 0, ANG_OF = 1, NPRIM_OF = 2, NCTR_OF = 3, PTR_EXP = 5, PTR_COEFF = 6 };
// libcint atm[] column index of the coordinate offset into env.
enum AtmSlot { PTR_COORD = 1 };

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

extern "C" int intti_int2e_cart(double *out, const int *shls, const int *atm, int natm,
                                 const int *bas, int nbas, const double *env, void * /*opt*/,
                                 double * /*cache*/) {
  return eval_int2e(out, shls, atm, natm, bas, nbas, env, false);
}

extern "C" int intti_int2e_sph(double *out, const int *shls, const int *atm, int natm,
                                const int *bas, int nbas, const double *env, void * /*opt*/,
                                double * /*cache*/) {
  return eval_int2e(out, shls, atm, natm, bas, nbas, env, true);
}
