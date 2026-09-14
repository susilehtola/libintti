#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026 Susi Lehtola
"""Dump a molecule + basis as libcint-convention shell data for the C++ benchmarks.

The C++ side converts to intti's convention with the SAME formula the validated
libcint facade uses (src/cint.cpp::contracted_basis_from), so nothing about
normalization is re-derived here -- this script only transports numbers.

Format (whitespace separated):
    nshell
    then per shell:  cx cy cz  l  nprim  nctr
                     alpha[nprim]
                     coeff[nctr*nprim]      (libcint env coefficients)

Usage: dump_basis.py <basis> <nwater> <outfile>
"""
import sys

import numpy as np
from pyscf import gto


def water_cluster(n):
    """n water molecules on a loose cubic lattice, ~5.5 bohr apart.

    Geometry is a rigid monomer translated; it is a cost benchmark, not a
    physical cluster, but the spacing is close enough to a hydrogen-bonded
    network that the screening behaviour is representative.
    """
    mono = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.81)), ("H", (1.75, 0.0, -0.45))]
    side = int(np.ceil(n ** (1.0 / 3.0)))
    atoms, k = [], 0
    for i in range(side):
        for j in range(side):
            for m in range(side):
                if k >= n:
                    break
                off = np.array([5.5 * i, 5.5 * j, 5.5 * m])
                for sym, pos in mono:
                    p = np.array(pos) + off
                    atoms.append((sym, tuple(p)))
                k += 1
    return atoms


def main():
    if len(sys.argv) != 4:
        sys.stderr.write(__doc__)
        return 1
    basis, nwat, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    atoms = water_cluster(nwat)
    mol = gto.M(atom=atoms, basis=basis, unit="Bohr", cart=True, verbose=0)
    with open(out, "w") as f:
        f.write("%d\n" % mol.nbas)
        for i in range(mol.nbas):
            c = mol.bas_coord(i)
            npr, nct = mol.bas_nprim(i), mol.bas_nctr(i)
            f.write("%.17g %.17g %.17g %d %d %d\n"
                    % (c[0], c[1], c[2], mol.bas_angular(i), npr, nct))
            f.write(" ".join("%.17g" % a for a in mol.bas_exp(i)) + "\n")
            # libcint stores coefficients as (nprim, nctr); the C++ reader wants
            # them contracted-function major, matching ContractedShell::coeff
            cc = np.asarray(mol.bas_ctr_coeff(i)).reshape(npr, nct)
            f.write(" ".join("%.17g" % cc[p, cn] for cn in range(nct)
                             for p in range(npr)) + "\n")
    sys.stderr.write("%s: %d water, %d shells, %d cart AOs -> %s\n"
                     % (basis, nwat, mol.nbas, mol.nao_cart(), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
