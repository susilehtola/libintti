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
from pyscf.hessian import rhf as hess_rhf

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
    ip1 = lib.intti_ip1_h1_jk
    ip1.restype = ctypes.c_int
    ip1.argtypes = [d, d, d, d, d, ctypes.c_int, ctypes.c_int, i, ctypes.c_int, i,
                    ctypes.c_int, d, ctypes.c_double]
    hs = lib.intti_hess_skeleton
    hs.restype = ctypes.c_int
    hs.argtypes = [d, d, d, i, ctypes.c_int, i, ctypes.c_int, d, ctypes.c_double]
    return f, g, hs, ip1


def make_h1_intti(fn, mol, tau=0.0):
    """Drop-in for pyscf.hessian.rhf.make_h1: the CPHF right-hand side.

    Replaces the last use of PySCF's own two-electron integrals in the Hessian
    path. make_h1 needs four int2e_ip1 contractions per atom, with the
    differentiated index restricted to that atom's shells; intti_ip1_h1_jk
    returns all four from one pass. vj2/vk2 come back full and are sliced to the
    atom's AO rows, which is the shape PySCF's 's1ij' scripts already produce.
    """
    nao = mol.nao_nr()
    atm = np.asarray(mol._atm, dtype=np.int32, order="C")
    bas = np.asarray(mol._bas, dtype=np.int32, order="C")
    env = np.asarray(mol._env, dtype=np.float64, order="C")

    def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
        mol_ = hessobj.mol
        if atmlst is None:
            atmlst = range(mol_.natm)
        mocc = mo_coeff[:, mo_occ > 0]
        dm0 = np.dot(mocc, mocc.T) * 2
        hcore_deriv = hessobj.base.nuc_grad_method().hcore_generator(mol_)
        aoslices = mol_.aoslice_by_atom()
        dptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        iptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        h1ao = [None] * mol_.natm
        for ia in atmlst:
            shl0, shl1, p0, p1 = aoslices[ia]
            o = [np.zeros((3, nao, nao)) for _ in range(4)]
            rc = fn(dptr(o[0]), dptr(o[1]), dptr(o[2]), dptr(o[3]),
                    dptr(np.ascontiguousarray(dm0)), int(shl0), int(shl1), iptr(atm),
                    mol_.natm, iptr(bas), mol_.nbas, dptr(env), float(tau))
            if rc != 0:
                raise RuntimeError(f"intti_ip1_h1_jk failed, rc={rc}")
            vj1, vj2, vk1, vk2 = o[0], o[1][:, p0:p1, :], o[2], o[3][:, p0:p1, :]
            vhf = vj1 - vk1 * .5
            vhf[:, p0:p1] += vj2 - vk2 * .5
            h1 = vhf + vhf.transpose(0, 2, 1)
            h1 += hcore_deriv(ia)
            h1ao[ia] = h1
        return h1ao

    return make_h1


def make_partial_hess(fn, mol, tau=0.0):
    """Drop-in for pyscf.hessian.rhf.partial_hess_elec.

    intti returns the skeleton contracted per SHELL centre; folding onto atoms
    is the caller's job (it owns the shell-to-atom map), exactly as for the
    gradient. The CPHF response terms and hess_nuc are PySCF's and are added by
    hess_elec on top of what this returns.
    """
    nao = mol.nao_nr()
    nbas = mol.nbas
    atm = np.asarray(mol._atm, dtype=np.int32, order="C")
    bas = np.asarray(mol._bas, dtype=np.int32, order="C")
    env = np.asarray(mol._env, dtype=np.float64, order="C")
    sh_atom = np.asarray(mol._bas[:, 0], dtype=int)

    def partial_hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                          atmlst=None, max_memory=4000, verbose=None):
        mf = hessobj.base
        if mo_coeff is None: mo_coeff = mf.mo_coeff
        if mo_occ is None: mo_occ = mf.mo_occ
        if mo_energy is None: mo_energy = mf.mo_energy
        mocc = mo_coeff[:, mo_occ > 0]
        dm0 = np.dot(mocc, mocc.T) * 2
        # energy-weighted density W = 2 sum_i eps_i C_i C_i^T
        dme0 = np.einsum("pi,qi,i->pq", mocc, mocc, mo_energy[mo_occ > 0]) * 2
        hs = np.zeros((3 * nbas, 3 * nbas))
        dptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        iptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        rc = fn(dptr(hs), dptr(np.ascontiguousarray(dm0)),
                dptr(np.ascontiguousarray(dme0)), iptr(atm), mol.natm, iptr(bas), nbas,
                dptr(env), float(tau))
        if rc != 0:
            raise RuntimeError(f"intti_hess_skeleton failed, rc={rc}")
        # fold shell centres onto atoms
        natm = mol.natm
        out = np.zeros((natm, natm, 3, 3))
        for p in range(nbas):
            for q in range(nbas):
                out[sh_atom[p], sh_atom[q]] += hs[3 * p:3 * p + 3, 3 * q:3 * q + 3]
        return out

    return partial_hess_elec


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
    fn, fn_ip1, fn_hess, fn_h1 = load(sys.argv[1])
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
    # skeleton Hessian: what partial_hess_elec computes, before the CPHF response
    hobj = ref.Hessian()
    h_ref = hess_rhf.partial_hess_elec(hobj)
    h_ours = make_partial_hess(fn_hess, mol)(hobj)
    dh = np.abs(np.asarray(h_ours) - np.asarray(h_ref)).max()
    print(f"skeleton Hessian agreement = {dh:.3e}  (|H| = {np.abs(h_ref).max():.3e})")
    # full Hessian and harmonic frequencies. The skeleton and the CPHF RESPONSE
    # (hessian.rhf.gen_vind routes through get_jk) are both on intti integrals;
    # the CPHF right-hand side, make_h1, still uses PySCF's int2e_ip1 -- stated
    # rather than glossed.
    H_ref = ref.Hessian().kernel()
    mf2 = scf.RHF(mol)
    mf2.conv_tol = 1e-12
    mf2.get_jk = make_get_jk(fn, mol)
    mf2.kernel()
    h2 = mf2.Hessian()
    h2.partial_hess_elec = make_partial_hess(fn_hess, mol).__get__(h2, type(h2))
    h2.make_h1 = make_h1_intti(fn_h1, mol).__get__(h2, type(h2))
    H_ours = h2.kernel()
    dH = np.abs(np.asarray(H_ours) - np.asarray(H_ref)).max()
    print(f"full Hessian agreement = {dH:.3e}  (|H| = {np.abs(H_ref).max():.3e})")

    from pyscf.hessian import thermo
    f_ref = thermo.harmonic_analysis(mol, H_ref)["freq_wavenumber"]
    f_our = thermo.harmonic_analysis(mol, np.asarray(H_ours))["freq_wavenumber"]
    print("harmonic frequencies (cm^-1):")
    for a, b in zip(np.atleast_1d(f_ref), np.atleast_1d(f_our)):
        print(f"    PySCF {a:12.4f}    intti {b:12.4f}    diff {b - a:+.2e}")
    df = np.abs(np.atleast_1d(f_our) - np.atleast_1d(f_ref)).max()
    ok = (ok and dh < 1e-8 and np.abs(h_ref).max() > 1e-2
          and dH < 1e-7 and df < 1e-4)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
