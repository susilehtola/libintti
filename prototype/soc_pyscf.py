# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026 Susi Lehtola
"""Pin the libintti two-electron spin-orbit operator against PySCF/libcint.

libintti's spin_orbit_2e_coulomb (include/intti/soc.hpp) builds the spin-same-
orbit two-electron operator (r12 x grad_1)/r12^3 contracted with a density. The
C++ test (tests/test_soc.cpp) validates the *builder* against a reduction-free
mixed centre finite difference, i.e. it proves the builder computes

    (mu lambda | SO_k | nu sigma) = eps_kij ( d_i mu  d_j lambda | nu sigma ),

the by-parts reduction we derived. What that self-check cannot show is that this
operator is the *standard* two-electron spin-orbit integral. This script closes
that gap: PySCF/libcint's int2e_p1vxp1 is the canonical SSO integral, and libcint
also provides int2e_ipvip1 = ( d_i mu  d_j lambda | nu sigma ), the analytic
double bra-derivative. We verify, to machine precision and independently of any
libintti code, that

    int2e_p1vxp1[k] == eps_kij * int2e_ipvip1[i,j],

so our reduction reproduces libcint's operator identically -- pinning sign,
component order and normalization. Combined with the C++ finite-difference test
this doubly validates the 2e SO integrals.

usage: python3 prototype/soc_pyscf.py
"""

import sys

import numpy as np
from pyscf import gto

EPS = np.zeros((3, 3, 3))
for _k in range(3):
    _i, _j = (_k + 1) % 3, (_k + 2) % 3
    EPS[_k, _i, _j] = 1.0
    EPS[_k, _j, _i] = -1.0


def check(mol, label, tol=1e-11):
    nao = mol.nao
    # (k, mu,nu,lambda,sigma) chemist -- SSO integral, electron 1 = (mu,lambda)
    T = mol.intor("int2e_p1vxp1")
    # (i,j, mu,lambda,nu,sigma) = ( d_i mu  d_j lambda | nu sigma )
    A = mol.intor("int2e_ipvip1").reshape(3, 3, nao, nao, nao, nao)
    G = np.einsum("kij,ij...->k...", EPS, A)
    err = np.abs(T - G).max()
    mag = np.abs(T).max()
    ok = err <= tol * (mag + 1.0)
    print(f"{label:24s} nao={nao:3d}  |int2e_p1vxp1 - eps*ipvip1|={err:.2e}"
          f"  |T|max={mag:.3e}  {'OK' if ok else 'FAIL'}")
    return ok


def main():
    ok = True
    # (a) s-only, minimal
    ok &= check(
        gto.M(atom="H 0 0 0; H 0 0 1.4; He 0.5 0.3 0.7",
              basis={"H": [[0, [1.2, 1.0]]], "He": [[0, [0.8, 1.0]]]},
              unit="Bohr", cart=True, spin=None), "s-only")
    # (b) up to d -- exercises the L-shift derivative path for l>0
    ok &= check(
        gto.M(atom="C 0 0 0; O 0 0 1.2; H 0.5 0.4 0.7",
              basis={"C": [[0, [1.5, 1.0]], [1, [0.9, 1.0]]],
                     "O": [[1, [1.1, 1.0]], [2, [0.8, 1.0]]],
                     "H": [[0, [0.6, 1.0]]]},
              unit="Bohr", cart=True, spin=None), "s,p,d shells")
    # (c) a contracted real basis
    ok &= check(gto.M(atom="O 0 0 0; H 0 1.43 1.1; H 0 -1.43 1.1",
                      basis="cc-pvdz", unit="Bohr", cart=True), "H2O/cc-pVDZ")
    print("ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
