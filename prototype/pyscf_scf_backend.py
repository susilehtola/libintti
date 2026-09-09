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

For the ONE-ELECTRON integrals there is a better level still: replacing
mol.intor itself (install_intti_intor). get_jk and friends intercept
CONTRACTIONS; mol.intor is PySCF's actual integral layer, which is what an
integrals library should be supplying. Patching it means PySCF's own, untouched
gradient and Hessian code runs on intti integrals with no hook at all, and it
reaches call sites that are not hookable -- hessian.rhf.hess_elec computes
s1a = -mol.intor('int1e_ipovlp', comp=3) inline.

Generally contracted basis sets are served natively by the contraction-aware
builders, at every density symmetry -- including the antisymmetric response
densities magnetic properties need. The remaining restriction is Cartesian only
(mol.cart = True); intti_get_jk reports a violation rather than answering
incorrectly.

usage: pyscf_scf_backend.py <path-to-libintti_cint.so>
"""
import ctypes
import sys

import numpy as np
from pyscf import df, gto, scf
from pyscf.grad import rhf as grad_rhf
from pyscf.hessian import rhf as hess_rhf
from pyscf.hessian import thermo

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
    e1 = lib.intti_int1e_ip
    e1.restype = ctypes.c_int
    e1.argtypes = [d, ctypes.c_int, ctypes.c_int, i, ctypes.c_int, i, ctypes.c_int, d]
    hs = lib.intti_hess_skeleton
    hs.restype = ctypes.c_int
    hs.argtypes = [d, d, d, i, ctypes.c_int, i, ctypes.c_int, d, ctypes.c_double]
    v = ctypes.c_void_p
    c2c = lib.intti_coulomb_2c
    c2c.restype = ctypes.c_int
    c2c.argtypes = [d, i, ctypes.c_int, i, ctypes.c_int, d]
    c3c = lib.intti_coulomb_3c
    c3c.restype = ctypes.c_int
    c3c.argtypes = [d, i, ctypes.c_int, i, ctypes.c_int, d, i, ctypes.c_int, i,
                    ctypes.c_int, d]
    rop = lib.intti_ri_open
    rop.restype = v
    rop.argtypes = [i, ctypes.c_int, i, ctypes.c_int, d, i, ctypes.c_int, i,
                    ctypes.c_int, d, ctypes.c_double]
    rjk = lib.intti_ri_get_jk
    rjk.restype = ctypes.c_int
    rjk.argtypes = [v, d, d, d, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    rcl = lib.intti_ri_close
    rcl.argtypes = [v]
    rid = lib.intti_ri_deriv_jk
    rid.restype = ctypes.c_int
    rid.argtypes = [d, d, d, d, ctypes.c_int, i, ctypes.c_int, i, ctypes.c_int, i,
                    ctypes.c_int, d, i, ctypes.c_int, i, ctypes.c_int, d, ctypes.c_double]
    rih = lib.intti_ri_hess_jk
    rih.restype = ctypes.c_int
    rih.argtypes = [d, d, d, d, ctypes.c_int, i, ctypes.c_int, i, ctypes.c_int, d,
                    i, ctypes.c_int, i, ctypes.c_int, d, ctypes.c_double]
    return f, g, hs, ip1, e1, rih, rid, (rop, rjk, rcl), (c2c, c3c)


def make_1e(fn, mol):
    """int1e_ipovlp / ipkin / ipnuc / iprinv from intti, PySCF's conventions."""
    nao = mol.nao_nr()
    atm = np.asarray(mol._atm, dtype=np.int32, order="C")
    bas = np.asarray(mol._bas, dtype=np.int32, order="C")
    env = np.asarray(mol._env, dtype=np.float64, order="C")

    def get(which, iatm=-1):
        o = np.zeros((3, nao, nao))
        rc = fn(o.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), int(which), int(iatm),
                atm.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), mol.natm,
                bas.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), mol.nbas,
                env.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        if rc != 0:
            raise RuntimeError(f"intti_int1e_ip failed, rc={rc}")
        return o

    return get


def install_intti_intor(get, mol):
    """Replace mol.intor itself for the names intti implements.

    A level below the get_jk / hcore_generator hooks: those intercept
    CONTRACTIONS, whereas mol.intor is PySCF's actual integral layer -- what an
    integrals library ought to be supplying. Patching it means PySCF's own,
    untouched one-electron gradient and Hessian code runs on intti integrals
    with no hook at all, and it reaches call sites that are not hookable:
    hessian.rhf.hess_elec computes s1a = -mol.intor('int1e_ipovlp', comp=3)
    inline, with no override point.

    Names we do not implement fall through to PySCF unchanged. Returns a counter
    so the caller can ASSERT the interception fired -- a monkeypatch that
    silently never triggers looks exactly like success.
    """
    orig = mol.intor
    served = {}

    def intor(intor_name, comp=None, hermi=0, aosym="s1", out=None, shls_slice=None,
              grids=None):
        base = intor_name.replace("_sph", "").replace("_cart", "")
        # POLICY: one-electron names only. A two-electron intor returns the
        # four-index tensor (or is consumed quartet by quartet), which is
        # exactly the per-quartet route the matrix-level API exists to avoid --
        # it would hand the device one quartet at a time and lose the batching.
        # Two-electron work goes through the matrix hooks (get_jk,
        # intti_get_jk_ip1, intti_hess_skeleton), never through here.
        assert not base.startswith("int2e"), (
            f"{base}: two-electron integrals must not be routed through "
            "mol.intor; use the matrix-level hooks")
        simple = shls_slice is None and out is None and grids is None
        table = {"int1e_ipovlp": 0, "int1e_ipkin": 1, "int1e_ipnuc": 2}
        if simple and base in table:
            served[base] = served.get(base, 0) + 1
            return get(table[base])
        if simple and base == "int1e_iprinv":
            # the origin is wherever mol.with_rinv_at_nucleus put it; match it to
            # a nucleus rather than trusting the caller's loop index
            org = mol._env[gto.PTR_RINV_ORIG:gto.PTR_RINV_ORIG + 3]
            ia = int(np.argmin(np.linalg.norm(mol.atom_coords() - org, axis=1)))
            if np.linalg.norm(mol.atom_coords()[ia] - org) < 1e-12:
                served[base] = served.get(base, 0) + 1
                return get(3, ia)
        return orig(intor_name, comp=comp, hermi=hermi, aosym=aosym, out=out,
                    shls_slice=shls_slice, grids=grids)

    mol.intor = intor
    return served


def make_h1_intti(fn, mol, hcore_gen=None, tau=0.0):
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
        hcore_deriv = (hcore_gen(mol_) if hcore_gen is not None
                       else hessobj.base.nuc_grad_method().hcore_generator(mol_))
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



def df_hessian_check(fn_rih, atoms, orb_basis, aux_basis, label):
    """The RI two-electron Hessians against pyscf.df.hessian.rhf, the seam.

    Until now ri_j_hessian and ri_k_hessian_occ were checked only against finite
    differences of OUR OWN RI gradients -- a real consistency check, but an
    internal one. This is the independent oracle.

    The conventions are derived, not fitted: both sides differentiate the same
    energy expressions, so with D = CL CR^T,

        ri_j_hessian(D)          = d^2[+1/2 Tr(D J)] ==  PySCF's ej
        ri_k_hessian_occ(CL, CR) = d^2[-1/4 Tr(D K)] == -PySCF's ek

    and a closed-shell dm0 = 2 C_occ C_occ^T is passed as CL = CR = sqrt(2)C_occ.
    ej and ek are compared SEPARATELY; summing them first would let an error in
    one hide inside the other.

    Our Hessian is indexed by shell centre, orbital shells then auxiliary
    shells. Folding both blocks onto atoms is what produces PySCF's
    auxbasis_response = 2 -- the auxiliary response is not a separate term for
    us, it is the auxiliary block of the same matrix.
    """
    from pyscf.df.hessian import rhf as dfhess
    dptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    iptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    mol = gto.M(atom=atoms, basis=orb_basis, unit="Bohr", cart=True, verbose=0)
    auxmol = df.addons.make_auxmol(mol, aux_basis)
    mf = scf.RHF(mol).density_fit(auxbasis=aux_basis)
    mf.conv_tol = 1e-12
    mf.kernel()
    hobj = dfhess.Hessian(mf)
    hobj.auxbasis_response = 2
    _, ej, ek = dfhess._partial_hess_ejk(hobj)

    nocc = int((mf.mo_occ > 0).sum())
    cocc = np.ascontiguousarray(mf.mo_coeff[:, mf.mo_occ > 0] * np.sqrt(2.0))
    dm0 = np.ascontiguousarray(mf.make_rdm1())
    ncen = mol.nbas + auxmol.nbas
    hj = np.zeros((3 * ncen, 3 * ncen))
    hk = np.zeros((3 * ncen, 3 * ncen))
    atmA = np.asarray(mol._atm, dtype=np.int32, order="C")
    basA = np.asarray(mol._bas, dtype=np.int32, order="C")
    envA = np.asarray(mol._env, dtype=np.float64, order="C")
    atmB = np.asarray(auxmol._atm, dtype=np.int32, order="C")
    basB = np.asarray(auxmol._bas, dtype=np.int32, order="C")
    envB = np.asarray(auxmol._env, dtype=np.float64, order="C")
    rc = fn_rih(dptr(hj), dptr(hk), dptr(dm0), dptr(cocc), nocc,
                iptr(atmA), mol.natm, iptr(basA), mol.nbas, dptr(envA),
                iptr(atmB), auxmol.natm, iptr(basB), auxmol.nbas, dptr(envB), 1e-12)
    assert rc == 0, f"intti_ri_hess_jk failed, rc={rc}"

    sh_atom = np.concatenate([basA[:, 0], basB[:, 0]])

    def fold(H):
        out = np.zeros((mol.natm, mol.natm, 3, 3))
        for p in range(ncen):
            for q in range(ncen):
                out[sh_atom[p], sh_atom[q]] += H[3 * p:3 * p + 3, 3 * q:3 * q + 3]
        return out

    dj = np.abs(fold(hj) - ej).max()
    dk = np.abs(fold(hk) + ek).max()
    print(f"    {label:22s} nao={mol.nao_nr():3d} naux={auxmol.nao_nr():3d}  "
          f"ej {dj:.2e} (|{np.abs(ej).max():.2e}|)  -ek {dk:.2e} (|{np.abs(ek).max():.2e}|)")
    return (dj < 1e-10 and dk < 1e-10
            and np.abs(ej).max() > 1e-2 and np.abs(ek).max() > 1e-2)



def df_full_hessian_check(fn_rih, fn_rid, fn_ri, atoms, orb_basis, aux_basis, label):
    """The whole DF Hessian on intti's RI derivative integrals.

    Two seams, both plain instance overrides:
      partial_hess_elec  <- intti_ri_hess_jk   (the ej / ek Hessians)
      make_h1            <- intti_ri_deriv_jk  (the CPHF right-hand side)

    The SCF itself and the CPHF RESPONSE additionally run on intti's cached RI
    J/K (intti_ri_open / intti_ri_get_jk), so every two-electron quantity in the
    DF pipeline is ours. The response densities are NOT symmetric, which is
    exactly why ri_jk contracts K as two GEMMs and assumes nothing about D.

    What is NOT ours, stated rather than glossed: the one-electron part e1 still
    comes from PySCF's _partial_hess_ejk -- though mol.intor is patched, so its
    derivative integrals are ours.

    make_h1 asks for 3*natm x nao^2, so the shell -> atom map is passed straight
    into the builders: the shell-resolved object is never formed.
    """
    from pyscf.df.hessian import rhf as dfhess
    from pyscf.hessian import thermo
    dptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    iptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    mol = gto.M(atom=atoms, basis=orb_basis, unit="Bohr", cart=True, verbose=0)
    auxmol = df.addons.make_auxmol(mol, aux_basis)
    ref = scf.RHF(mol).density_fit(auxbasis=aux_basis)
    ref.conv_tol = 1e-12
    ref.kernel()
    H_ref = dfhess.Hessian(ref).set(auxbasis_response=2).kernel()

    nao, natm = mol.nao_nr(), mol.natm
    atmA = np.asarray(mol._atm, dtype=np.int32, order="C")
    basA = np.asarray(mol._bas, dtype=np.int32, order="C")
    envA = np.asarray(mol._env, dtype=np.float64, order="C")
    atmB = np.asarray(auxmol._atm, dtype=np.int32, order="C")
    basB = np.asarray(auxmol._bas, dtype=np.int32, order="C")
    envB = np.asarray(auxmol._env, dtype=np.float64, order="C")
    grp = np.ascontiguousarray(np.concatenate([basA[:, 0], basB[:, 0]]).astype(np.int32))
    ncen = mol.nbas + auxmol.nbas

    # the CACHED RI fit: built once here, reused by every SCF iteration and every
    # CPHF response build. A one-shot get_jk would rebuild it each call.
    ri_open, ri_get_jk, ri_close = fn_ri
    handle = ri_open(iptr(atmA), natm, iptr(basA), mol.nbas, dptr(envA),
                     iptr(atmB), auxmol.natm, iptr(basB), auxmol.nbas, dptr(envB), 1e-12)
    assert handle, "intti_ri_open failed"

    def ri_jk(mol_, dm, hermi=1, with_j=True, with_k=True, omega=None):
        dm = np.asarray(dm)
        squeeze = dm.ndim == 2
        D = np.ascontiguousarray(dm.reshape(-1, nao, nao))
        vj = np.zeros_like(D)
        vk = np.zeros_like(D)
        rc = ri_get_jk(handle, dptr(vj), dptr(vk), dptr(D), D.shape[0],
                       int(with_j), int(with_k))
        assert rc == 0, f"intti_ri_get_jk rc={rc}"
        return (vj[0] if squeeze else vj), (vk[0] if squeeze else vk)

    mf = scf.RHF(mol).density_fit(auxbasis=aux_basis)
    mf.conv_tol = 1e-12
    mf.get_jk = ri_jk
    e_our = mf.kernel()
    assert mf.converged, "DF-SCF on intti RI J/K did not converge"

    def fold(H):
        out = np.zeros((natm, natm, 3, 3))
        for p in range(ncen):
            for q in range(ncen):
                out[grp[p], grp[q]] += H[3 * p:3 * p + 3, 3 * q:3 * q + 3]
        return out

    def partial(hessobj, *a, **kw):
        cocc = np.ascontiguousarray(mf.mo_coeff[:, mf.mo_occ > 0] * np.sqrt(2.0))
        dm0 = np.ascontiguousarray(mf.make_rdm1())
        e1, _, _ = dfhess._partial_hess_ejk(hessobj)   # 1e part only is reused
        hj = np.zeros((3 * ncen, 3 * ncen))
        hk = np.zeros((3 * ncen, 3 * ncen))
        rc = fn_rih(dptr(hj), dptr(hk), dptr(dm0), dptr(cocc), cocc.shape[1],
                    iptr(atmA), natm, iptr(basA), mol.nbas, dptr(envA),
                    iptr(atmB), auxmol.natm, iptr(basB), auxmol.nbas, dptr(envB), 1e-12)
        assert rc == 0, f"intti_ri_hess_jk rc={rc}"
        return e1 + fold(hj) + fold(hk)   # hk already carries its minus sign

    def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
        mocc = mo_coeff[:, mo_occ > 0]
        cocc = np.ascontiguousarray(mocc * np.sqrt(2.0))
        dm0 = np.ascontiguousarray(np.dot(mocc, mocc.T) * 2)
        vj = np.zeros((3 * natm, nao, nao))
        vk = np.zeros((3 * natm, nao, nao))
        rc = fn_rid(dptr(vj), dptr(vk), dptr(dm0), dptr(cocc), cocc.shape[1],
                    iptr(grp), natm, iptr(atmA), natm, iptr(basA), mol.nbas, dptr(envA),
                    iptr(atmB), auxmol.natm, iptr(basB), auxmol.nbas, dptr(envB), 1e-12)
        assert rc == 0, f"intti_ri_deriv_jk rc={rc}"
        hcore_deriv = hessobj.base.nuc_grad_method().hcore_generator(mol)
        return [hcore_deriv(ia) + vj[3 * ia:3 * ia + 3] - 0.5 * vk[3 * ia:3 * ia + 3]
                for ia in range(natm)]

    hobj = dfhess.Hessian(mf)
    hobj.auxbasis_response = 2
    hobj.partial_hess_elec = partial.__get__(hobj, type(hobj))
    hobj.make_h1 = make_h1.__get__(hobj, type(hobj))
    H_our = np.asarray(hobj.kernel())
    ri_close(handle)
    dE = abs(e_our - ref.e_tot)
    dH = np.abs(H_our - np.asarray(H_ref)).max()
    f_ref = thermo.harmonic_analysis(mol, H_ref)["freq_wavenumber"]
    f_our = thermo.harmonic_analysis(mol, H_our)["freq_wavenumber"]
    dfreq = np.abs(np.atleast_1d(f_our) - np.atleast_1d(f_ref)).max()
    print(f"    {label:22s} E {dE:.2e}  full-H {dH:.2e} (|{np.abs(H_ref).max():.2e}|)  "
          f"freq {dfreq:.2e} cm^-1")
    print(f"    {label:22s} freqs " + " ".join(f"{x:.4f}" for x in np.atleast_1d(f_our)))
    return (dE < 1e-10 and dH < 1e-9 and dfreq < 1e-4
            and np.abs(H_ref).max() > 1e-2)



def df_contracted_check(fn_ri, atoms, cases, label):
    """DF-RHF on CONTRACTED bases with a real Coulomb-fitting set.

    ri_fit is templated on the basis types, so contracted RI is the same code
    with the contraction-aware two- and three-centre builders underneath. The
    auxiliary basis is where this bites: cc-pVDZ-JKFIT carries angular momentum
    up to f, well above the orbital basis, and l >= 2 in the AUXILIARY position
    is what a hand-made sp test basis never reaches.
    """
    ri_open, ri_get_jk, ri_close = fn_ri
    dptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    iptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    rng = np.random.default_rng(5)
    ok = True
    for ob, ab in cases:
        mol = gto.M(atom=atoms, basis=ob, unit="Bohr", cart=True, verbose=0)
        auxmol = df.addons.make_auxmol(mol, ab)
        nao = mol.nao_nr()
        atmA = np.asarray(mol._atm, dtype=np.int32, order="C")
        basA = np.asarray(mol._bas, dtype=np.int32, order="C")
        envA = np.asarray(mol._env, dtype=np.float64, order="C")
        atmB = np.asarray(auxmol._atm, dtype=np.int32, order="C")
        basB = np.asarray(auxmol._bas, dtype=np.int32, order="C")
        envB = np.asarray(auxmol._env, dtype=np.float64, order="C")
        handle = ri_open(iptr(atmA), mol.natm, iptr(basA), mol.nbas, dptr(envA),
                         iptr(atmB), auxmol.natm, iptr(basB), auxmol.nbas, dptr(envB), 1e-12)
        assert handle, "intti_ri_open failed"

        def ri_jk(mol_, dm, hermi=1, with_j=True, with_k=True, omega=None):
            dm = np.asarray(dm)
            squeeze = dm.ndim == 2
            D = np.ascontiguousarray(dm.reshape(-1, nao, nao))
            vj = np.zeros_like(D)
            vk = np.zeros_like(D)
            rc = ri_get_jk(handle, dptr(vj), dptr(vk), dptr(D), D.shape[0],
                           int(with_j), int(with_k))
            assert rc == 0, f"intti_ri_get_jk rc={rc}"
            return (vj[0] if squeeze else vj), (vk[0] if squeeze else vk)

        ref = scf.RHF(mol).density_fit(auxbasis=ab)
        ref.conv_tol = 1e-12
        ref.kernel()
        g = rng.standard_normal((nao, nao)) * 0.05
        worst = {}
        for tag, dm in (("scf", ref.make_rdm1()), ("gen", g), ("anti", g - g.T)):
            oj, ok_ = ri_jk(mol, dm)
            rj, rk = ref.get_jk(mol, dm, hermi=0)
            worst[tag] = max(np.abs(oj - rj).max(), np.abs(ok_ - rk).max())
            ok = ok and worst[tag] < 1e-9
        mf = scf.RHF(mol).density_fit(auxbasis=ab)
        mf.conv_tol = 1e-12
        mf.get_jk = ri_jk
        e = mf.kernel()
        ri_close(handle)
        nctr = max(int(mol._bas[k, 3]) for k in range(mol.nbas))
        lmax = max(int(auxmol._bas[k, 1]) for k in range(auxmol.nbas))
        print(f"    {ob:8s}/{ab:14s} nao={nao:3d} naux={auxmol.nao_nr():3d} "
              f"nctr={nctr} aux_lmax={lmax}  E {e - ref.e_tot:+.1e}  "
              + "  ".join(f"{k} {v:.1e}" for k, v in worst.items()))
        ok = ok and mf.converged and abs(e - ref.e_tot) < 1e-9 and lmax >= 2
    return ok



def ncentre_check(fn_nc, atoms, cases):
    """The two- and three-centre Coulomb tensors against libcint directly.

    Per-builder localisation, which the RI energy cannot give: the fit
    T M^-1 T^T is INVARIANT to any diagonal rescaling of the auxiliary AOs
    (T -> TS, M -> SMS leaves it unchanged), so an auxiliary-side convention
    error cancels and only a genuine inconsistency BETWEEN M and T shows up --
    as a wrong energy, with no indication of which builder is at fault.

    That is not hypothetical. The contracted two-centre metric double-counted
    the off-diagonal elements of a diagonal shell block, and it took a bisection
    to find, because the unit tests compared contracted against contracted. It
    was invisible below l = 2 as well: a same-centre two-centre integral
    vanishes unless both components have even parity in every direction, so the
    first nonzero off-diagonals in a diagonal block are the (xx|yy)-type trace
    pairs. Hence the auxiliary angular momentum is pushed to g here.
    """
    c2c, c3c = fn_nc
    dptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    iptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    ok = True
    for ob, ab in cases:
        mol = gto.M(atom=atoms, basis=ob, unit="Bohr", cart=True, verbose=0)
        aux = df.addons.make_auxmol(mol, ab)
        nao, naux = mol.nao_nr(), aux.nao_nr()
        atmA = np.asarray(mol._atm, dtype=np.int32, order="C")
        basA = np.asarray(mol._bas, dtype=np.int32, order="C")
        envA = np.asarray(mol._env, dtype=np.float64, order="C")
        atmB = np.asarray(aux._atm, dtype=np.int32, order="C")
        basB = np.asarray(aux._bas, dtype=np.int32, order="C")
        envB = np.asarray(aux._env, dtype=np.float64, order="C")
        M = np.zeros((naux, naux))
        assert c2c(dptr(M), iptr(atmB), aux.natm, iptr(basB), aux.nbas, dptr(envB)) == 0
        Mref = aux.intor("int2c2e", aosym="s1")
        T = np.zeros((nao, nao, naux))
        assert c3c(dptr(T), iptr(atmA), mol.natm, iptr(basA), mol.nbas, dptr(envA),
                   iptr(atmB), aux.natm, iptr(basB), aux.nbas, dptr(envB)) == 0
        Tref = df.incore.aux_e2(mol, aux, intor="int3c2e", aosym="s1")
        d2 = np.abs(M - Mref).max() / np.abs(Mref).max()
        d3 = np.abs(T - Tref).max() / np.abs(Tref).max()
        lo = max(int(mol._bas[k, 1]) for k in range(mol.nbas))
        la = max(int(aux._bas[k, 1]) for k in range(aux.nbas))
        nctr = max(int(mol._bas[k, 3]) for k in range(mol.nbas))
        print(f"    {ob:8s}(l{lo},nctr{nctr}) / {ab:14s}(l{la}) naux={naux:3d}  "
              f"2c {d2:.2e}  3c {d3:.2e}  (relative)")
        ok = ok and d2 < 1e-12 and d3 < 1e-12
        ok = ok and la >= 3   # the aux angular momentum must actually be high
    return ok


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    (fn, fn_ip1, fn_hess, fn_h1, fn_1e, fn_rih, fn_rid, fn_ri,
     fn_nc) = load(sys.argv[1])
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
    H_ref_saved = ref.Hessian().kernel()
    # From here on mol.intor itself is intti's, so PySCF's OWN one-electron
    # gradient and Hessian code runs on our integrals with no hook at all -- the
    # 1e hooks written earlier become unnecessary. Installed after every
    # reference has been computed, so the comparisons stay honest.
    served = install_intti_intor(make_1e(fn_1e, mol), mol)
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
    H_ref = H_ref_saved
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

    f_ref = thermo.harmonic_analysis(mol, H_ref)["freq_wavenumber"]
    f_our = thermo.harmonic_analysis(mol, np.asarray(H_ours))["freq_wavenumber"]
    print("harmonic frequencies (cm^-1):")
    for a, b in zip(np.atleast_1d(f_ref), np.atleast_1d(f_our)):
        print(f"    PySCF {a:12.4f}    intti {b:12.4f}    diff {b - a:+.2e}")
    df = np.abs(np.atleast_1d(f_our) - np.atleast_1d(f_ref)).max()
    # CONTRACTED basis sets: everything above runs on an uncontracted basis,
    # which is what intti_get_jk used to accept. Standard basis sets are all
    # contracted, so this is the difference between a demonstrator and something
    # runnable. cc-pVDZ is generally contracted (nctr = 2), not merely segmented.
    print("contracted basis sets, PySCF's SCF on intti J/K:")
    for bname in ("sto-3g", "6-31g", "cc-pvdz"):
        cmol = gto.M(atom=ATOM, basis=bname, unit="Bohr", cart=True, verbose=0)
        cref = scf.RHF(cmol)
        cref.conv_tol = 1e-12
        ce_ref = cref.kernel()
        cmf = scf.RHF(cmol)
        cmf.conv_tol = 1e-12
        cmf.get_jk = make_get_jk(fn, cmol)
        ce = cmf.kernel()
        assert cmf.converged, f"{bname}: SCF on intti J/K did not converge"
        nctr = max(int(cmol._bas[i, 3]) for i in range(cmol.nbas))
        print(f"    {bname:9s} nao={cmol.nao_nr():3d} max_nctr={nctr}  "
              f"E {ce:.12f}  diff {ce - ce_ref:+.2e}")
        ok = ok and abs(ce - ce_ref) < 1e-9
        # Response densities are not symmetric. The contracted J/K route has no
        # hermiticity restriction, so check it where it matters: a general dm
        # (hermi=0) and the antisymmetric dm NMR response uses (hermi=2).
        rng = np.random.default_rng(0)
        g = rng.standard_normal((cmol.nao_nr(),) * 2) * 0.05
        cjk = make_get_jk(fn, cmol)
        for tag, dm, h in (("general", g, 0), ("antisym", g - g.T, 2)):
            oj, ok_ = cjk(cmol, dm, hermi=h)
            rj, rk = cref.get_jk(cmol, dm, hermi=h)
            dj = np.abs(oj - rj).max()
            dk = np.abs(ok_ - rk).max()
            print(f"    {bname:9s} {tag:8s} hermi={h}  dJ {dj:.2e}  dK {dk:.2e}")
            ok = ok and dj < 1e-11 and dk < 1e-11 and np.abs(rk).max() > 1e-3
        # One-electron DERIVATIVES on the contracted basis. These are what
        # mol.intor interception serves, so without them PySCF's own gradient
        # and Hessian code could only run on an uncontracted basis.
        cget = make_1e(fn_1e, cmol)
        checks = [("ipovlp", cget(0), cmol.intor("int1e_ipovlp", comp=3)),
                  ("ipkin", cget(1), cmol.intor("int1e_ipkin", comp=3)),
                  ("ipnuc", cget(2), cmol.intor("int1e_ipnuc", comp=3))]
        with cmol.with_rinv_at_nucleus(1):
            checks.append(("iprinv", cget(3, 1), cmol.intor("int1e_iprinv", comp=3)))
        for tag, o, r in checks:
            d = np.abs(o - r).max()
            print(f"    {bname:9s} {tag:8s} d {d:.2e}  |ref| {np.abs(r).max():.2e}")
            ok = ok and d < 1e-10 * max(np.abs(r).max(), 1.0)
        # THE WHOLE PIPELINE on a contracted basis: SCF, gradient, skeleton
        # Hessian, full Hessian and harmonic frequencies, with every integral
        # coming from intti (2e via the matrix hooks, 1e via mol.intor
        # interception). This is the capstone repeated on a real basis set
        # rather than the uncontracted demonstrator above; cc-pVDZ is generally
        # contracted (nctr = 2), which is the case that exercises the shared
        # primitive intermediates rather than mere segmentation.
        cg_ref = cref.nuc_grad_method().kernel()
        cH_ref = cref.Hessian().kernel()
        chobj = cref.Hessian()
        ch_ref = hess_rhf.partial_hess_elec(chobj)
        install_intti_intor(make_1e(fn_1e, cmol), cmol)  # after the references
        ch_our = make_partial_hess(fn_hess, cmol)(chobj)
        dch = np.abs(np.asarray(ch_our) - np.asarray(ch_ref)).max()
        cg = cmf.nuc_grad_method()
        cg.get_jk = make_grad_get_jk(fn_ip1, cmol)
        cg_our = cg.kernel()
        dcg = np.abs(np.asarray(cg_our) - np.asarray(cg_ref)).max()
        ch2 = cmf.Hessian()
        ch2.partial_hess_elec = make_partial_hess(fn_hess, cmol).__get__(ch2, type(ch2))
        ch2.make_h1 = make_h1_intti(fn_h1, cmol).__get__(ch2, type(ch2))
        cH_our = ch2.kernel()
        dcH = np.abs(np.asarray(cH_our) - np.asarray(cH_ref)).max()
        cf_ref = thermo.harmonic_analysis(cmol, cH_ref)["freq_wavenumber"]
        cf_our = thermo.harmonic_analysis(cmol, np.asarray(cH_our))["freq_wavenumber"]
        dcf = np.abs(np.atleast_1d(cf_our) - np.atleast_1d(cf_ref)).max()
        print(f"    {bname:9s} gradient {dcg:.2e}  skeleton-H {dch:.2e}  "
              f"full-H {dcH:.2e}  freq {dcf:.2e} cm^-1")
        print(f"    {bname:9s} freqs " +
              " ".join(f"{x:.4f}" for x in np.atleast_1d(cf_our)))
        # each reference must be non-trivial, or agreement means nothing
        ok = (ok and dcg < 1e-9 and dch < 1e-9 and dcH < 1e-8 and dcf < 1e-4
              and np.abs(cg_ref).max() > 1e-3 and np.abs(cH_ref).max() > 1e-2)
    # RI (density-fitted) two-electron Hessians vs pyscf.df.hessian.rhf. Both
    # bases uncontracted: the RI derivative layer has no contraction-aware
    # builders yet, unlike the direct J/K path.
    print("RI two-electron Hessians vs pyscf.df.hessian.rhf:")
    ok = df_hessian_check(
        fn_rih, ATOM,
        {"O": uncontracted((0, [3.0, 0.9, 0.3]), (1, [1.1, 0.35])),
         "H": uncontracted((0, [1.3, 0.35]))},
        {"O": uncontracted((0, [6.0, 1.8, 0.6]), (1, [2.0, 0.7])),
         "H": uncontracted((0, [2.4, 0.7]))},
        "H2O sp/sp-aux") and ok
    # a second system with different connectivity, and an auxiliary basis
    # carrying d functions -- the aux angular momentum is what the auxiliary
    # response terms actually exercise
    ok = df_hessian_check(
        fn_rih,
        [["N", (0.0, 0.0, 0.0)], ["H", (1.9, 0.0, 0.3)],
         ["H", (-0.9, 1.6, 0.3)], ["H", (-0.9, -1.6, 0.3)]],
        {"N": uncontracted((0, [4.0, 1.0]), (1, [0.9])),
         "H": uncontracted((0, [1.2, 0.3]))},
        {"N": uncontracted((0, [7.0, 2.0]), (1, [1.6]), (2, [1.1])),
         "H": uncontracted((0, [2.2, 0.6]), (1, [0.8]))},
        "NH3 d-aux") and ok
    # ...and the whole DF Hessian driven on those derivative integrals
    print("DF Hessian on intti RI derivative integrals:")
    ok = df_full_hessian_check(
        fn_rih, fn_rid, fn_ri, ATOM,
        {"O": uncontracted((0, [3.0, 0.9, 0.3]), (1, [1.1, 0.35])),
         "H": uncontracted((0, [1.3, 0.35]))},
        {"O": uncontracted((0, [6.0, 1.8, 0.6]), (1, [2.0, 0.7])),
         "H": uncontracted((0, [2.4, 0.7]))},
        "H2O RI freqs") and ok
    # per-builder oracle for the RI n-centre tensors, up to g in the auxiliary
    print("2c/3c Coulomb tensors vs libcint:")
    ok = ncentre_check(
        fn_nc, ATOM,
        [("sto-3g", "cc-pvdz-jkfit"), ("cc-pvdz", "cc-pvdz-jkfit"),
         ("cc-pvdz", "cc-pvtz-jkfit")]) and ok
    # contracted RI: standard orbital basis AND a standard Coulomb-fitting set,
    # which is where the auxiliary angular momentum finally exceeds l = 1
    print("DF-RHF on contracted bases, intti RI J/K:")
    ok = df_contracted_check(
        fn_ri, ATOM,
        [("sto-3g", "cc-pvdz-jkfit"), ("6-31g", "cc-pvdz-jkfit"),
         ("cc-pvdz", "cc-pvdz-jkfit")],
        "contracted RI") and ok
    print("mol.intor calls served by intti:", dict(sorted(served.items())))
    # a patch that never fired would look identical to success
    assert served.get("int1e_ipovlp", 0) > 0, "mol.intor interception never fired"
    ok = (ok and dh < 1e-8 and np.abs(h_ref).max() > 1e-2
          and dH < 1e-7 and df < 1e-4)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
