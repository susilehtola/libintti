# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026 Susi Lehtola
"""Run PySCF's OWN SCF driver on libintti's J/K, via intti_get_jk.

This is the integration test the hand-written C++ SCF driver in tests/test_scf.cpp
was standing in for. Overriding get_jk is PySCF's documented extension point, and
its signature maps almost one-to-one onto intti::jk_build:

    dm list        -> vector<JKRequest>
    hermi 0/1/2    -> DensitySymmetry::General / Symmetric / Antisymmetric
    with_j/with_k  -> FockTerms
    omega          -> range-separated kernel on the t grid

Validating this way exercises the builders inside a real driver -- real DIIS,
real convergence control -- instead of one written for the test, and it is the
route by which PySCF's gradient and Hessian machinery can be driven on intti
integrals without reimplementing CPHF.

NOTE the restriction: intti's matrix builders take PRIMITIVE shells, so the
molecule must use an uncontracted Cartesian basis. intti_get_jk reports a
violation rather than answering incorrectly.

usage: pyscf_scf_backend.py <path-to-libintti_cint.so>
"""
import ctypes
import sys

import numpy as np
from pyscf import gto, scf
from pyscf.grad import rhf as grad_rhf

# same molecule and uncontracted basis as tests/test_scf.cpp
ATOM = [
    ["O", (0.0, 0.0, -0.1230376)],
    ["H", (0.0, 1.4300472, 0.9762012)],
    ["H", (0.0, -1.4300472, 0.9762012)],
]
O_S = [130.70932, 23.808861, 6.4436083, 1.1695961, 0.3803890]
O_P = [5.0331513, 1.1695961, 0.3803890]
H_S = [3.4252509, 0.6239137, 0.1688554]


def uncontracted(*shells):
    return [[l, [a, 1.0]] for l, exps in shells for a in exps]


def load(lib_path):
    lib = ctypes.CDLL(lib_path)
    d, i = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)
    f = lib.intti_get_jk
    f.restype = ctypes.c_int
    f.argtypes = [d, d, d, ctypes.c_int, i, ctypes.c_int, ctypes.c_int,
                  i, ctypes.c_int, i, ctypes.c_int, d,
                  ctypes.c_double, ctypes.c_double]
    g = lib.intti_get_jk_ip1
    g.restype = ctypes.c_int
    g.argtypes = [d, d, d, i, ctypes.c_int, i, ctypes.c_int, d, ctypes.c_double]
    return f, g


def make_grad_get_jk(fn, mol, tau=0.0):
    """Drop-in for pyscf.grad.rhf.get_jk, backed by intti::jk_deriv_ao_build."""
    nao = mol.nao_nr()
    atm = np.asarray(mol._atm, dtype=np.int32, order="C")
    bas = np.asarray(mol._bas, dtype=np.int32, order="C")
    env = np.asarray(mol._env, dtype=np.float64, order="C")

    def get_jk(mol_, dm):
        d = np.ascontiguousarray(np.asarray(dm, dtype=np.float64))
        vj = np.zeros((3, nao, nao))
        vk = np.zeros((3, nao, nao))
        dptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        iptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        rc = fn(dptr(vj), dptr(vk), dptr(d), iptr(atm), mol.natm, iptr(bas), mol.nbas,
                dptr(env), float(tau))
        if rc != 0:
            raise RuntimeError(f"intti_get_jk_ip1 failed, rc={rc}")
        return vj, vk

    return get_jk


def make_get_jk(fn, mol, tau=0.0):
    """A drop-in replacement for mf.get_jk backed by intti::jk_build."""
    nao = mol.nao_nr()
    atm = np.asarray(mol._atm, dtype=np.int32, order="C")
    bas = np.asarray(mol._bas, dtype=np.int32, order="C")
    env = np.asarray(mol._env, dtype=np.float64, order="C")

    def get_jk(mol_, dm, hermi=1, with_j=True, with_k=True, omega=None, **kwargs):
        dms = np.asarray(dm, dtype=np.float64, order="C")
        single = dms.ndim == 2
        dms = dms.reshape(-1, nao, nao)
        ndm = dms.shape[0]
        vj = np.zeros((ndm, nao, nao)) if with_j else np.zeros((1, 1, 1))
        vk = np.zeros((ndm, nao, nao)) if with_k else np.zeros((1, 1, 1))
        herm = np.full(ndm, int(hermi), dtype=np.int32)
        dptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        iptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        rc = fn(dptr(vj), dptr(vk), dptr(np.ascontiguousarray(dms)), ndm, iptr(herm),
                int(bool(with_j)), int(bool(with_k)), iptr(atm), mol.natm, iptr(bas),
                mol.nbas, dptr(env), float(omega or 0.0), float(tau))
        if rc == -1:
            raise RuntimeError("intti_get_jk: basis must be uncontracted (nctr=nprim=1)")
        if rc != 0:
            raise RuntimeError(f"intti_get_jk failed, rc={rc}")
        if single:
            return (vj[0] if with_j else None), (vk[0] if with_k else None)
        return (vj if with_j else None), (vk if with_k else None)

    return get_jk


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    fn, fn_ip1 = load(sys.argv[1])
    basis = {"O": uncontracted((0, O_S), (1, O_P)), "H": uncontracted((0, H_S))}
    mol = gto.M(atom=ATOM, basis=basis, unit="Bohr", cart=True, verbose=0)

    ref = scf.RHF(mol)
    ref.conv_tol = 1e-12
    e_ref = ref.kernel()
    assert ref.converged

    # the same driver, with intti supplying every J and K
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.get_jk = make_get_jk(fn, mol)
    e = mf.kernel()
    assert mf.converged, "PySCF SCF on intti J/K did not converge"

    print(f"nao          = {mol.nao_nr()}  (uncontracted, cart)")
    print(f"E(PySCF)     = {e_ref:.15f}")
    print(f"E(intti J/K) = {e:.15f}")
    print(f"difference   = {e - e_ref:.3e}")

    # J/K agreement on a NON-symmetric density, which the SCF itself never
    # exercises but the response machinery depends on
    rng = np.random.default_rng(0)
    nao = mol.nao_nr()
    dm = rng.standard_normal((nao, nao)) * 0.05
    ours = make_get_jk(fn, mol)(mol, dm, hermi=0)
    theirs = ref.get_jk(mol, dm, hermi=0)
    dj = np.abs(ours[0] - theirs[0]).max()
    dk = np.abs(ours[1] - theirs[1]).max()
    print(f"general-density J agreement = {dj:.3e}")
    print(f"general-density K agreement = {dk:.3e}")
    # hermi = 2: the ANTISYMMETRIC case. This is not a corner case -- NMR
    # shielding's CPHF response is driven exactly this way
    # (pyscf/prop/nmr/rhf.py: vresp = mf.gen_response(singlet=True, hermi=2)
    # with dm1 = d1 - d1.conj().T), and it is the case the fused exchange_build
    # cannot represent, since it builds an upper triangle and mirrors it.
    a = rng.standard_normal((nao, nao)) * 0.05
    a = a - a.T
    ours_a = make_get_jk(fn, mol)(mol, a, hermi=2)
    theirs_a = ref.get_jk(mol, a, hermi=2)
    dja = np.abs(ours_a[0] - theirs_a[0]).max()
    dka = np.abs(ours_a[1] - theirs_a[1]).max()
    ksc = np.abs(theirs_a[1]).max()
    print(f"antisym-density J agreement = {dja:.3e}  (|J| = {np.abs(theirs_a[0]).max():.3e})")
    print(f"antisym-density K agreement = {dka:.3e}  (|K| = {ksc:.3e})")
    # K must be genuinely nonzero and antisymmetric, or the check is vacuous
    print(f"   |K + K^T| = {np.abs(theirs_a[1] + theirs_a[1].T).max():.3e}")
    ok = (abs(e - e_ref) < 1e-9 and dj < 1e-9 and dk < 1e-9
          and dja < 1e-9 and dka < 1e-9 and ksc > 1e-4)
    # derivative J/K in the bra-gradient convention, against pyscf.grad.rhf
    dm0 = ref.make_rdm1()
    gj, gk = make_grad_get_jk(fn_ip1, mol)(mol, dm0)
    rj, rk = grad_rhf.get_jk(mol, dm0)
    dgj = np.abs(gj - rj).max()
    dgk = np.abs(gk - rk).max()
    print(f"grad-convention dJ/dA agreement = {dgj:.3e}  (|dJ| = {np.abs(rj).max():.3e})")
    print(f"grad-convention dK/dA agreement = {dgk:.3e}  (|dK| = {np.abs(rk).max():.3e})")
    # ...and the payoff: PySCF's OWN gradient assembly driven on intti's
    # derivative integrals. Gradients.get_jk is the hook get_veff calls.
    g_ref = ref.nuc_grad_method().kernel()
    gobj = ref.nuc_grad_method()
    gobj.get_jk = make_grad_get_jk(fn_ip1, mol)
    g_ours = gobj.kernel()
    dg = np.abs(np.asarray(g_ours) - np.asarray(g_ref)).max()
    print("gradient (Ha/bohr), PySCF assembly on intti derivative J/K:")
    for row in np.asarray(g_ours):
        print("    " + "  ".join(f"{x:14.10f}" for x in row))
    print(f"gradient agreement = {dg:.3e}  (|g| = {np.abs(g_ref).max():.3e})")
    ok = (ok and dgj < 1e-9 and dgk < 1e-9 and np.abs(rj).max() > 1e-3
          and dg < 1e-9 and np.abs(g_ref).max() > 1e-3)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
