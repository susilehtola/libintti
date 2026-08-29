# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Susi Lehtola
"""Cross-validate libintti 2-/3-center Coulomb integrals against PySCF.

usage: pyscf_ri_validation.py <path-to-intti_dump_ri> [workdir]
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
from pyscf import gto, df

# orbital shells (center, l, exponent) and auxiliary shells
ORB = [(0, 0, 1.2), (0, 1, 0.8), (1, 0, 0.5), (1, 2, 0.6)]
AUX = [(0, 0, 2.6), (0, 1, 1.1), (0, 2, 0.9), (1, 0, 1.7), (1, 1, 0.7)]
CENTERS = [(0.0, 0.0, 0.0), (0.0, 0.0, 1.4)]


def mol_of(shells):
    basis, atoms = {}, []
    for ic, c in enumerate(CENTERS):
        sym = f"X{ic}"
        atoms.append(f"{sym} {c[0]} {c[1]} {c[2]}")
        basis[sym] = [[l, [a, 1.0]] for (jc, l, a) in shells if jc == ic]
    m = gto.M(atom="; ".join(atoms), basis=basis, unit="Bohr", cart=True, spin=None)
    return m


def write_spec(path, shells):
    with open(path, "w") as f:
        for (ic, l, a) in shells:
            c = CENTERS[ic]
            f.write(f"{c[0]} {c[1]} {c[2]} {l} {a}\n")


def main():
    dumper = sys.argv[1]
    wd = sys.argv[2] if len(sys.argv) > 2 else tempfile.mkdtemp()
    mol, auxmol = mol_of(ORB), mol_of(AUX)
    ospec, aspec, out = (os.path.join(wd, x) for x in ("orb.txt", "aux.txt", "ri.bin"))
    write_spec(ospec, ORB)
    write_spec(aspec, AUX)
    r = subprocess.run([dumper, ospec, aspec, out], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    tok = r.stdout.split()
    nao, naux = int(tok[1]), int(tok[3])
    assert nao == mol.nao and naux == auxmol.nao, (nao, mol.nao, naux, auxmol.nao)

    data = np.fromfile(out, dtype=np.float64)
    M = data[:naux * naux].reshape(naux, naux)
    T = data[naux * naux:].reshape(nao, nao, naux)

    M_ref = auxmol.intor("int2c2e_cart")
    T_ref = df.incore.aux_e2(mol, auxmol, intor="int3c2e_cart")

    d2 = np.abs(M - M_ref).max() / max(np.abs(M_ref).max(), 1e-12)
    d3 = np.abs(T - T_ref).max() / max(np.abs(T_ref).max(), 1e-12)
    print(f"  (P|Q)  2-center max rel dev {d2:.3e}  {'PASS' if d2 < 1e-12 else '*** FAIL ***'}",
          flush=True)
    print(f"  (uv|P) 3-center max rel dev {d3:.3e}  {'PASS' if d3 < 1e-12 else '*** FAIL ***'}",
          flush=True)
    ok = d2 < 1e-12 and d3 < 1e-12
    print(f"-> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
