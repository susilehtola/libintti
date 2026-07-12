# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Susi Lehtola
"""Cross-validation of libintti Cartesian ERIs against PySCF/libcint.

Builds an uncontracted two-center s/p/d system in PySCF and with the
intti_dump_eri tool, and compares the full int2e_cart tensor under the
normalization hypotheses of include/intti/normalization.hpp. Pins the
component ordering and normalization conventions for the libcint facade
(docs/conventions.md).

usage: pyscf_validation.py <path-to-intti_dump_eri> [workdir]
"""

import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
from pyscf import gto

# uncontracted shells: (center, l, exponent); centers in Bohr
CENTERS = [(0.0, 0.0, 0.0), (0.0, 0.0, 1.4)]
SHELLS = [
    (0, 0, 1.2),
    (0, 1, 0.8),
    (0, 3, 0.7),  # f shell: validates the 4 pi/(2l+1) factor beyond l = 2
    (1, 0, 0.5),
    (1, 2, 0.9),
]


def pyscf_eri():
    basis = {}
    atoms = []
    for ic, c in enumerate(CENTERS):
        sym = f"GHOST{ic + 1}"
        atoms.append(f"{sym} {c[0]} {c[1]} {c[2]}")
        basis[sym] = [[l, [alpha, 1.0]] for (jc, l, alpha) in SHELLS if jc == ic]
    mol = gto.M(atom="; ".join(atoms), basis=basis, unit="Bohr", cart=True,
                spin=None)
    return mol.intor("int2e_cart", aosym="s1"), mol.nao


def intti_eri(dumper, workdir, mode):
    spec = os.path.join(workdir, "shells.txt")
    out = os.path.join(workdir, f"eri_{mode}.bin")
    with open(spec, "w") as f:
        for (ic, l, alpha) in SHELLS:
            c = CENTERS[ic]
            f.write(f"{c[0]} {c[1]} {c[2]} {l} {alpha}\n")
    res = subprocess.run([dumper, spec, out, mode], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"dumper failed: {res.stderr}")
    nao = int(res.stdout.split()[-1])
    eri = np.fromfile(out, dtype=np.float64).reshape(nao, nao, nao, nao)
    return eri, nao


def main():
    dumper = sys.argv[1]
    workdir = sys.argv[2] if len(sys.argv) > 2 else tempfile.mkdtemp()
    ref, nao_ref = pyscf_eri()
    scale = np.abs(ref).max()
    print(f"pyscf nao = {nao_ref}, max |eri| = {scale:.6e}")
    best = None
    for mode in ("comp", "cca", "pyscf"):
        eri, nao = intti_eri(dumper, workdir, mode)
        assert nao == nao_ref, f"nao mismatch: {nao} vs {nao_ref}"
        dev = np.abs(eri - ref).max() / scale
        print(f"normalization '{mode}': max rel deviation {dev:.3e}")
        if best is None or dev < best[1]:
            best = (mode, dev)
    print(f"best: '{best[0]}' at {best[1]:.3e}")
    if best[1] > 1e-10:
        print("FAIL: no normalization hypothesis matches PySCF")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
