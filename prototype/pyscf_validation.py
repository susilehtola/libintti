# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Susi Lehtola
"""Cross-validation of libintti ERIs against PySCF/libcint.

Two independent checks, either or both of which may be requested:

--dumper <path-to-intti_dump_eri> [--workdir DIR]
    Builds an uncontracted two-center s/p/d/f system in PySCF and with the
    intti_dump_eri tool, and compares the full int2e_cart tensor under the
    normalization hypotheses of include/intti/normalization.hpp. Pins the
    component ordering and Cartesian normalization conventions
    (docs/conventions.md).

--facade <path-to-libintti_cint.so>
    Loads the libcint-compatible C facade (include/intti/cint.h) via ctypes
    and compares intti_int2e_cart/intti_int2e_sph, called directly on
    PySCF's own atm/bas/env arrays, against mol.intor('int2e_cart') /
    mol.intor('int2e_sph') for (a) the same uncontracted system as the
    --dumper check and (b) a contracted real basis (H2O/cc-pVDZ). Pins the
    contraction-coefficient rescale and the spherical c2s_matrix rescale
    (docs/conventions.md).

usage: pyscf_validation.py [--dumper PATH] [--workdir DIR] [--facade PATH]
"""

import argparse
import ctypes
import os
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

FACADE_TOL = 1e-12


def uncontracted_mol(cart):
    basis = {}
    atoms = []
    for ic, c in enumerate(CENTERS):
        sym = f"GHOST{ic + 1}"
        atoms.append(f"{sym} {c[0]} {c[1]} {c[2]}")
        basis[sym] = [[l, [alpha, 1.0]] for (jc, l, alpha) in SHELLS if jc == ic]
    return gto.M(atom="; ".join(atoms), basis=basis, unit="Bohr", cart=cart, spin=None)


def h2o_ccpvdz_mol(cart):
    return gto.M(atom="O 0 0 0; H 0 1.43 1.1; H 0 -1.43 1.1", basis="cc-pvdz",
                 unit="Bohr", cart=cart, spin=None)


# --------------------------------------------------------------------------
# --dumper: uncontracted int2e_cart, three normalization hypotheses
# --------------------------------------------------------------------------

def pyscf_eri():
    mol = uncontracted_mol(cart=True)
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


def run_dumper_check(dumper, workdir):
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
        return False
    print("PASS (dumper)")
    return True


# --------------------------------------------------------------------------
# --facade: intti_int2e_{cart,sph} via ctypes against mol.intor
# --------------------------------------------------------------------------

def load_facade(lib_path):
    lib = ctypes.CDLL(lib_path)
    for name in ("intti_int2e_cart", "intti_int2e_sph"):
        fn = getattr(lib, name)
        fn.restype = ctypes.c_int
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.c_void_p, ctypes.c_void_p,
        ]
    return lib


def facade_eri(lib, mol, cart):
    fn = lib.intti_int2e_cart if cart else lib.intti_int2e_sph
    atm = np.ascontiguousarray(mol._atm, dtype=np.int32)
    bas = np.ascontiguousarray(mol._bas, dtype=np.int32)
    env = np.ascontiguousarray(mol._env, dtype=np.float64)
    natm, nbas = atm.shape[0], bas.shape[0]
    atm_p = atm.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    bas_p = bas.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    env_p = env.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    ao_loc = mol.ao_loc_nr(cart=cart)
    nao = int(ao_loc[-1])
    eri = np.zeros((nao, nao, nao, nao))
    for i in range(nbas):
        di = int(ao_loc[i + 1] - ao_loc[i])
        for j in range(nbas):
            dj = int(ao_loc[j + 1] - ao_loc[j])
            for k in range(nbas):
                dk = int(ao_loc[k + 1] - ao_loc[k])
                for l in range(nbas):
                    dl = int(ao_loc[l + 1] - ao_loc[l])
                    buf = np.zeros(di * dj * dk * dl, dtype=np.float64)
                    shls = (ctypes.c_int * 4)(i, j, k, l)
                    fn(buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), shls,
                       atm_p, natm, bas_p, nbas, env_p, None, None)
                    block = buf.reshape((di, dj, dk, dl), order="F")
                    eri[ao_loc[i]:ao_loc[i + 1], ao_loc[j]:ao_loc[j + 1],
                        ao_loc[k]:ao_loc[k + 1], ao_loc[l]:ao_loc[l + 1]] = block
    return eri


def check_facade_system(lib, mol, label):
    ok = True
    for cart, kind in ((True, "cart"), (False, "sph")):
        ref = mol.intor("int2e_cart" if cart else "int2e_sph")
        eri = facade_eri(lib, mol, cart)
        scale = np.abs(ref).max()
        dev = np.abs(eri - ref).max() / scale
        status = "PASS" if dev < FACADE_TOL else "FAIL"
        print(f"facade[{label}] int2e_{kind}: max rel deviation {dev:.3e} ({status})")
        if dev >= FACADE_TOL:
            ok = False
    return ok


def run_facade_check(facade_path):
    lib = load_facade(facade_path)
    ok = True
    ok &= check_facade_system(lib, uncontracted_mol(cart=True), "uncontracted")
    ok &= check_facade_system(lib, h2o_ccpvdz_mol(cart=True), "H2O/cc-pVDZ")
    print("PASS (facade)" if ok else "FAIL (facade)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumper", help="path to intti_dump_eri")
    ap.add_argument("--workdir", default=None, help="scratch dir for --dumper")
    ap.add_argument("--facade", help="path to the libintti_cint shared library")
    args = ap.parse_args()

    if not args.dumper and not args.facade:
        ap.error("at least one of --dumper or --facade is required")

    ok = True
    if args.dumper:
        workdir = args.workdir or tempfile.mkdtemp()
        ok &= run_dumper_check(args.dumper, workdir)
    if args.facade:
        ok &= run_facade_check(args.facade)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
