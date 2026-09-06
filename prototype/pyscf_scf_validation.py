# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026 Susi Lehtola
"""Reference RI-RHF (density-fitted) energy for tests/test_scf.cpp.

Milestone-13 oracle / capstone stage A: libintti drives a complete RI-RHF from
its matrix builders (S, T, V, ri_jk) and the total energy is compared with
PySCF's df.RHF on the SAME molecule, orbital basis and auxiliary basis.

The comparison is convention-free: an RHF energy is invariant under any
nonsingular transformation of the AO basis, and the RI J/K is a projection onto
the auxiliary span in the Coulomb metric, so it is invariant under nonsingular
transformations of the auxiliary basis too. Only the SPANS have to agree --
normalisation and shell ordering cannot move the number. The orbital basis is
uncontracted s/p (identical span in Cartesian and spherical form); the auxiliary
basis carries d functions, hence cart=True so that both codes use the 6-function
Cartesian d shell.

usage: pyscf_scf_validation.py
prints the total energy to paste into tests/test_scf.cpp.
"""
import numpy as np
from pyscf import df, gto, scf

# geometry in bohr -- must match kAtoms in tests/test_scf.cpp
ATOM = [
    ["O", (0.0, 0.0, -0.1230376)],
    ["H", (0.0, 1.4300472, 0.9762012)],
    ["H", (0.0, -1.4300472, 0.9762012)],
]

# uncontracted orbital basis -- must match orbital_shells()
O_S = [130.70932, 23.808861, 6.4436083, 1.1695961, 0.3803890]
O_P = [5.0331513, 1.1695961, 0.3803890]
H_S = [3.4252509, 0.6239137, 0.1688554]

# auxiliary basis -- must match auxiliary_shells()
A_S = [52.0, 13.0, 3.2, 0.8, 0.25]
A_P = [6.4, 1.6, 0.4]
A_D = [2.4, 0.6]
B_S = [6.8, 1.7, 0.42]
B_P = [1.4]


def uncontracted(*shells):
    """[(l, [exponents]), ...] -> PySCF basis entry, one primitive per shell."""
    out = []
    for l, exps in shells:
        for a in exps:
            out.append([l, [a, 1.0]])
    return out


def main():
    basis = {
        "O": uncontracted((0, O_S), (1, O_P)),
        "H": uncontracted((0, H_S)),
    }
    auxbasis = {
        "O": uncontracted((0, A_S), (1, A_P), (2, A_D)),
        "H": uncontracted((0, B_S), (1, B_P)),
    }
    mol = gto.M(atom=ATOM, basis=basis, unit="Bohr", cart=True, verbose=0)
    mf = df.density_fit(scf.RHF(mol), auxbasis=auxbasis)
    mf.conv_tol = 1e-12
    e = mf.kernel()
    assert mf.converged, "PySCF df.RHF did not converge"
    # both bases must be Cartesian for the spans to match libintti's
    assert mol.cart, "orbital basis is not Cartesian"
    assert mf.with_df.auxmol.cart, "auxiliary basis is not Cartesian"
    print(f"nao        = {mol.nao_nr()}")
    print(f"naux       = {mf.with_df.auxmol.nao_nr()}")
    print(f"E_nuc      = {mol.energy_nuc():.15f}")
    mf0 = scf.RHF(mol)
    mf0.conv_tol = 1e-12
    e0 = mf0.kernel()
    assert mf0.converged, "PySCF RHF did not converge"
    g0 = mf0.nuc_grad_method().kernel()
    print(f"E(df.RHF)  = {e:.15f}")
    print(f"E(RHF)     = {e0:.15f}")
    print(f"RI error   = {e - e0:.3e}")
    print("grad(RHF) = dE/dR in Ha/bohr, one row per atom:")
    for row in np.asarray(g0):
        print("    {" + ", ".join(f"{x:.15f}" for x in row) + "},")
    return e, e0, g0


if __name__ == "__main__":
    main()
