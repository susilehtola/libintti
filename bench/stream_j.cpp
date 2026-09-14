// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

// Streaming grid-RI Coulomb: never allocates the whole-molecule grid.
//
// gridri.hpp's reference implementation materialises the AO values as an
// nao x N^3 View, which on a real basis set is hundreds of gigabytes. That is a
// property of the reference implementation, not of the method: the work is
// element-local, so one element block of points is enough at a time. This
// benchmark does it that way to find out what the method actually costs.
//
// Per element block: evaluate the AOs that reach it, get V there analytically
// (the Gaussian density's own Coulomb potential, with the exponent-free
// multipole far branch), and accumulate
//     J_ab += sum_{g in block} w_g chi_a(g) chi_b(g) V(g)
// so the peak footprint is nao_local x p^3 rather than nao x N^3.
//
// Usage: stream_j <basis-dump> [max_blocks]
// max_blocks > 0 processes only that many element blocks and reports a rate,
// which is how a grid that would take hours is measured in seconds.

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "basis_io.hpp"
#include "intti/contracted.hpp"
#include "intti/gridri.hpp"
#include "intti/kernel.hpp"
#include "intti/nuclear.hpp"
#include "intti/tgrid.hpp"

namespace {

struct AoEntry {
  double cx, cy, cz;
  int lx, ly, lz;
  int poff, np;
  double amin, cmax; // for the block screen
};

std::vector<AoEntry> flatten(const intti::ContractedBasis<double> &b,
                             std::vector<double> &apool, std::vector<double> &ecpool) {
  std::vector<AoEntry> ao(b.nao);
  for (int A = 0; A < static_cast<int>(b.shells.size()); ++A) {
    const auto &sh = b.shells[A];
    const int nc = intti::ncart(sh.l), np = sh.nprim();
    for (int cA = 0; cA < sh.nctr(); ++cA)
      for (int k = 0; k < nc; ++k) {
        int a3[3];
        intti::cart_comp(sh.l, k, a3[0], a3[1], a3[2]);
        AoEntry e;
        e.cx = sh.center[0];
        e.cy = sh.center[1];
        e.cz = sh.center[2];
        e.lx = a3[0];
        e.ly = a3[1];
        e.lz = a3[2];
        e.poff = static_cast<int>(apool.size());
        e.np = np;
        e.amin = sh.alpha[0];
        e.cmax = 0;
        for (int p = 0; p < np; ++p) {
          apool.push_back(sh.alpha[p]);
          const double ec = intti::detail::effective_coeff(sh, cA, p);
          ecpool.push_back(ec);
          e.amin = std::min(e.amin, sh.alpha[p]);
          e.cmax = std::max(e.cmax, std::abs(ec));
        }
        ao[b.ao_off[A] + cA * nc + k] = e;
      }
  }
  return ao;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <basis-dump> [max_blocks]\n", argv[0]);
    return 1;
  }
  const std::string path = argv[1];
  const long max_blocks = argc > 2 ? std::atol(argv[2]) : 0;

  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    const auto basis = intti_bench::load_basis(path);
    const int nao = basis.nao;
    auto grid = intti::grid_for_basis(basis, 1e-2);
    auto tg = intti::make_tgrid(intti::coulomb());
    const int N = grid.N, ne = grid.ne;
    const double npts = double(N) * N * N;
    const long nblock_tot = long(ne) * ne * ne;
    std::printf("%s: nao=%d nshell=%d  grid N=%d ne=%d  points=%.3e  blocks=%ld\n",
                path.c_str(), nao, (int)basis.shells.size(), N, ne, npts, nblock_tot);

    std::vector<double> D((std::size_t)nao * nao);
    for (int i = 0; i < nao; ++i)
      for (int j = 0; j < nao; ++j)
        D[(std::size_t)i * nao + j] = 0.05 * std::cos(0.7 * i + 1.3 * j);
    for (int i = 0; i < nao; ++i)
      for (int j = i + 1; j < nao; ++j) {
        const double a = 0.5 * (D[(std::size_t)i * nao + j] + D[(std::size_t)j * nao + i]);
        D[(std::size_t)i * nao + j] = D[(std::size_t)j * nao + i] = a;
      }

    std::vector<double> apool, ecpool;
    const auto ao = flatten(basis, apool, ecpool);
    std::vector<double> J((std::size_t)nao * nao, 0.0);

    const double ln_cut = -std::log(1e-12); // AO block screen
    long done = 0, ptsdone = 0;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::array<double, 3>> pts;
    std::vector<double> wts;
    std::vector<int> live;
    std::vector<double> chi; // nlive x npt

    for (int ex = 0; ex < ne && (max_blocks <= 0 || done < max_blocks); ++ex)
      for (int ey = 0; ey < ne && (max_blocks <= 0 || done < max_blocks); ++ey)
        for (int ez = 0; ez < ne && (max_blocks <= 0 || done < max_blocks); ++ez) {
          pts.clear();
          wts.clear();
          for (int i = 0; i < grid.nps[ex]; ++i) {
            const int ix = grid.noff[ex] + i;
            for (int j = 0; j < grid.nps[ey]; ++j) {
              const int iy = grid.noff[ey] + j;
              for (int k = 0; k < grid.nps[ez]; ++k) {
                const int iz = grid.noff[ez] + k;
                pts.push_back({grid.xnode[ix], grid.xnode[iy], grid.xnode[iz]});
                wts.push_back(grid.xw[ix] * grid.xw[iy] * grid.xw[iz]);
              }
            }
          }
          const int npt = static_cast<int>(pts.size());
          // which AOs reach this block at all
          const double bx = 0.5 * (grid.be[ex] + grid.be[ex + 1]);
          const double by = 0.5 * (grid.be[ey] + grid.be[ey + 1]);
          const double bz = 0.5 * (grid.be[ez] + grid.be[ez + 1]);
          const double hx = 0.5 * (grid.be[ex + 1] - grid.be[ex]);
          const double hy = 0.5 * (grid.be[ey + 1] - grid.be[ey]);
          const double hz = 0.5 * (grid.be[ez + 1] - grid.be[ez]);
          live.clear();
          for (int a = 0; a < nao; ++a) {
            const double dx = std::max(0.0, std::abs(ao[a].cx - bx) - hx);
            const double dy = std::max(0.0, std::abs(ao[a].cy - by) - hy);
            const double dz = std::max(0.0, std::abs(ao[a].cz - bz) - hz);
            if (ao[a].amin * (dx * dx + dy * dy + dz * dz) < ln_cut) live.push_back(a);
          }
          ++done;
          ptsdone += npt;
          if (live.empty()) continue;

          const int nl = static_cast<int>(live.size());
          chi.assign((std::size_t)nl * npt, 0.0);
          for (int q = 0; q < nl; ++q) {
            const auto &e = ao[live[q]];
            for (int g = 0; g < npt; ++g) {
              const double dx = pts[g][0] - e.cx, dy = pts[g][1] - e.cy,
                           dz = pts[g][2] - e.cz;
              const double r2 = dx * dx + dy * dy + dz * dz;
              double rad = 0;
              for (int p = 0; p < e.np; ++p)
                rad += ecpool[e.poff + p] * std::exp(-apool[e.poff + p] * r2);
              for (int i = 0; i < e.lx; ++i) rad *= dx;
              for (int i = 0; i < e.ly; ++i) rad *= dy;
              for (int i = 0; i < e.lz; ++i) rad *= dz;
              chi[(std::size_t)q * npt + g] = rad;
            }
          }
          // the Gaussian density's own potential on this block
          // (expanded to primitives: potential_on_points takes a ShellBasis)
          static std::vector<double> Dp;
          static intti::ShellBasis<double> pb;
          static bool built = false;
          if (!built) {
            const auto f = intti::detail::expand_contracted(basis, pb);
            // push the contracted density down to the primitive basis
            Dp.assign((std::size_t)pb.nao * pb.nao, 0.0);
            const auto fan = intti::detail::fanout_matrix(f, pb);
            for (int i = 0; i < pb.nao; ++i)
              for (int j = 0; j < pb.nao; ++j) {
                double s = 0;
                for (int u = 0; u < nao; ++u) {
                  const double cu = fan[(std::size_t)u * pb.nao + i];
                  if (cu == 0.0) continue;
                  for (int v = 0; v < nao; ++v)
                    s += cu * D[(std::size_t)u * nao + v] *
                         fan[(std::size_t)v * pb.nao + j];
                }
                Dp[(std::size_t)i * pb.nao + j] = s;
              }
            built = true;
          }
          const auto V = intti::potential_on_points(pb, Dp.data(), pts, tg, 0.0, 1e-10);
          for (int q = 0; q < nl; ++q)
            for (int r = q; r < nl; ++r) {
              double s = 0;
              for (int g = 0; g < npt; ++g)
                s += wts[g] * chi[(std::size_t)q * npt + g] * chi[(std::size_t)r * npt + g] *
                     V[g];
              J[(std::size_t)live[q] * nao + live[r]] += s;
              if (r != q) J[(std::size_t)live[r] * nao + live[q]] += s;
            }
        }

    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    const double per_pt = ms * 1e-3 / std::max(1L, ptsdone);
    std::printf("  processed %ld of %ld blocks (%ld points) in %.1f s\n", done, nblock_tot,
                ptsdone, ms * 1e-3);
    std::printf("  %.3e s/point  ->  whole grid would take %.3e s (%.2f h)\n", per_pt,
                per_pt * npts, per_pt * npts / 3600);
    std::printf("  peak AO block footprint: %.1f kB (nao_local x p^3)\n",
                double(nao) * 8 * 512 / 1024.0);
  }
  Kokkos::finalize();
  return rc;
}
