// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola
#pragma once

#include "gto.hpp"
#include "tgrid.hpp"

namespace intti {

/// Primitive Cartesian ERI quartet (ab|cd) by t quadrature:
///   (ab|cd) = sum_i w_i Ix(t_i) Iy(t_i) Iz(t_i) + tail_coeff * S_abcd,
/// where I_d are analytic 1D two-pair integrals and S_abcd is the
/// four-orbital overlap (the delta-function tail correction).
///
/// out is a host array of ncart(la)*ncart(lb)*ncart(lc)*ncart(ld) values in
/// row-major (a, b, c, d) component order. Primitives are unnormalized.
void eri_quartet(const ShellPair &bra, const ShellPair &ket, const TGrid &grid,
                 double *out);

} // namespace intti
