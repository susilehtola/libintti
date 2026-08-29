# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Susi Lehtola
"""Cross-validate libintti one-electron property matrices against PySCF.

Builds an uncontracted s/p/d/f two-center system, dumps libintti's overlap,
kinetic, dipole and quadrupole matrices with intti_dump_oneel, and compares
to PySCF int1e_ovlp_cart / int1e_kin_cart / int1e_r_cart / int1e_rr_cart.

usage: pyscf_oneel_validation.py <path-to-intti_dump_oneel> [workdir]
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
from pyscf import gto

CENTERS = [(0.0, 0.0, 0.0), (0.0, 0.0, 1.4)]
SHELLS = [(0, 0, 1.2), (0, 1, 0.8), (0, 2, 0.9), (0, 3, 0.7),
          (1, 0, 0.5), (1, 1, 0.6), (1, 2, 0.45)]


def build_mol():
    basis, atoms = {}, []
    for ic, c in enumerate(CENTERS):
        sym = f"GHOST{ic + 1}"
        atoms.append(f"{sym} {c[0]} {c[1]} {c[2]}")
        basis[sym] = [[l, [a, 1.0]] for (jc, l, a) in SHELLS if jc == ic]
    mol = gto.M(atom="; ".join(atoms), basis=basis, unit="Bohr", cart=True, spin=None)
    mol.set_common_orig([0.0, 0.0, 0.0])
    return mol


def libintti(dumper, workdir):
    spec = os.path.join(workdir, "shells1e.txt")
    out = os.path.join(workdir, "oneel.bin")
    with open(spec, "w") as f:
        for (ic, l, a) in SHELLS:
            c = CENTERS[ic]
            f.write(f"{c[0]} {c[1]} {c[2]} {l} {a}\n")
    r = subprocess.run([dumper, spec, out], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"dumper failed: {r.stderr}")
    nao = int(r.stdout.split()[-1])
    data = np.fromfile(out, dtype=np.float64)
    n2 = nao * nao
    blocks = [data[i * n2:(i + 1) * n2].reshape(nao, nao) for i in range(22)]
    # S, T, dipole(3), quad(6), rinv@p0, rinv@p1, ipovlp(3), ipkin(3),
    # iprinv@p0(3)
    return nao, blocks


RINV_POINTS = [(0.0, 0.0, 0.4), (0.2, -0.1, 0.9)]


def main():
    dumper = sys.argv[1]
    workdir = sys.argv[2] if len(sys.argv) > 2 else tempfile.mkdtemp()
    mol = build_mol()
    nao, B = libintti(dumper, workdir)
    assert nao == mol.nao, f"nao mismatch {nao} vs {mol.nao}"

    S = mol.intor("int1e_ovlp_cart")
    T = mol.intor("int1e_kin_cart")
    r = mol.intor("int1e_r_cart")        # (3, nao, nao)
    rr = mol.intor("int1e_rr_cart")      # (9, nao, nao): xx xy xz yx yy yz zx zy zz

    refs = {
        "overlap":   (S, B[0]),
        "kinetic":   (T, B[1]),
        "dipole_x":  (r[0], B[2]), "dipole_y": (r[1], B[3]), "dipole_z": (r[2], B[4]),
        "quad_xx":   (rr[0], B[5]), "quad_xy": (rr[1], B[6]), "quad_xz": (rr[2], B[7]),
        "quad_yy":   (rr[4], B[8]), "quad_yz": (rr[5], B[9]), "quad_zz": (rr[8], B[10]),
    }
    for i, pt in enumerate(RINV_POINTS):
        mol.set_rinv_origin(pt)
        refs[f"rinv@p{i}"] = (mol.intor("int1e_rinv_cart"), B[11 + i])
    ipovlp = mol.intor("int1e_ipovlp_cart")   # (3, nao, nao)
    ipkin = mol.intor("int1e_ipkin_cart")
    for d, ax in enumerate("xyz"):
        refs[f"ipovlp_{ax}"] = (ipovlp[d], B[13 + d])
        refs[f"ipkin_{ax}"] = (ipkin[d], B[16 + d])
    mol.set_rinv_origin(RINV_POINTS[0])
    iprinv = mol.intor("int1e_iprinv_cart")   # (3, nao, nao)
    for d, ax in enumerate("xyz"):
        refs[f"iprinv_{ax}"] = (iprinv[d], B[19 + d])
    worst = 0.0
    ok = True
    for name, (ref, got) in refs.items():
        scale = max(np.abs(ref).max(), 1e-12)
        dev = np.abs(got - ref).max() / scale
        worst = max(worst, dev)
        flag = "PASS" if dev < 1e-12 else "*** FAIL ***"
        if dev >= 1e-12:
            ok = False
        print(f"  {name:10s} max rel dev {dev:.3e}  {flag}", flush=True)
    print(f"worst: {worst:.3e} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
