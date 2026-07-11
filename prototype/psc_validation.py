# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Susi Lehtola
"""Validation of the prolate-spheroidal (PSC) representation of orbital
products and of kernel-mediated interactions between them (libintti M2c).

A two-center orbital product phi_a(r-A) phi_b(r-B) has *exactly* finite
azimuthal content |m| <= l_a + l_b about its interfocal axis, so it is
exactly representable as sum_m f_m(xi, eta) e^{i m phi} on a 2D grid.
At each node of the t quadrature the Gaussian kernel couples two coaxial
products diagonally in m through modified Bessel factors:

  int dphi1 dphi2 e^{i m1 phi1 + i m2 phi2} e^{2 t^2 rho1 rho2 cos(phi1-phi2)}
      = (2 pi)^2 delta_{m1,-m2} I_{m1}(2 t^2 rho1 rho2),

with e^{-t^2(rho1^2+rho2^2)} I_m(2t^2 rho1 rho2) evaluated stably as
ive(m, 2t^2 rho1 rho2) * exp(-t^2 (rho1-rho2)^2).

Grid-represented products cannot resolve the kernel at arbitrarily large t,
so they use the truncated LinLog t grid with the delta-function tail
correction (tail = pi/t_c^2 * <chi|chi'>); this is the same regime split as
in the C++ TGrid (Mobius for analytic representations only).

Validations:
  1. representation: charge and self-overlap of s/p GTO products on the PSC
     grid vs analytic values
  2. coaxial Coulomb interaction via t quadrature + Bessel m-diagonal
     coupling vs the analytic ERI (ab|cd)
  3. non-coaxial products via uniform-phi grids: N_phi convergence
  4. mixed representation: analytic GTO bra x PSC-gridded ket
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ive, erf


# ----------------------------------------------------------------- analytic
def F0(x):
    x = np.asarray(x, float)
    out = np.ones_like(x)
    m = x > 1e-14
    out[m] = 0.5 * np.sqrt(np.pi / x[m]) * erf(np.sqrt(x[m]))
    return out


def eri_ssss(za, A, zb, B, zc, C, zd, D):
    p, q = za + zb, zc + zd
    P = (za * A + zb * B) / p
    Q = (zc * C + zd * D) / q
    K = np.exp(-za * zb / p * np.sum((A - B) ** 2)) * np.exp(
        -zc * zd / q * np.sum((C - D) ** 2))
    rho = p * q / (p + q)
    x = rho * np.sum((P - Q) ** 2)
    return 2 * np.pi**2.5 / (p * q * np.sqrt(p + q)) * K * float(F0(np.array([x]))[0])


# --------------------------------------------------- PSC product on a grid
class PSCProduct:
    """phi_a(r-A) phi_b(r-B) on a PSC grid; A, B on the z axis at -+ R/2.
    Shells: (exponent, cartesian powers (px,py,pz)). Stored as Fourier
    components f_m(xi, eta) with |m| <= max azimuthal degree (exact)."""

    def __init__(self, za, pa, zb, pb, R, n_xi=48, n_eta=48, xi_max=None):
        self.R = R
        mdeg = pa[0] + pa[1] + pb[0] + pb[1]  # max azimuthal degree
        n_phi = 2 * mdeg + 1
        # quadratures
        x, w = leggauss(n_eta)
        self.eta, self.w_eta = x, w
        if xi_max is None:
            zeff = min(za, zb)
            xi_max = 1.0 + 12.0 / np.sqrt(zeff * (R / 2) ** 2 + 1e-30) + 2.0
        x, w = leggauss(n_xi)
        self.xi = 0.5 * (xi_max - 1) * (x + 1) + 1
        self.w_xi = 0.5 * (xi_max - 1) * w
        self.phi = 2 * np.pi * np.arange(n_phi) / n_phi

        XI, ETA = np.meshgrid(self.xi, self.eta, indexing="ij")
        self.z = (R / 2) * XI * ETA
        self.rho = (R / 2) * np.sqrt(np.maximum((XI**2 - 1) * (1 - ETA**2), 0.0))
        # dV = (R/2)^3 (xi^2 - eta^2) dxi deta dphi
        self.w2d = ((R / 2) ** 3 * (XI**2 - ETA**2)
                    * np.outer(self.w_xi, self.w_eta))

        # evaluate the product on (xi, eta, phi) and FFT in phi (exact)
        vals = np.empty((len(self.xi), len(self.eta), n_phi))
        for k, ph in enumerate(self.phi):
            xg = self.rho * np.cos(ph)
            yg = self.rho * np.sin(ph)
            zA = self.z + R / 2   # z - A_z with A_z = -R/2
            zB = self.z - R / 2
            rA2 = xg**2 + yg**2 + zA**2
            rB2 = xg**2 + yg**2 + zB**2
            vals[:, :, k] = (xg ** pa[0] * yg ** pa[1] * zA ** pa[2]
                             * np.exp(-za * rA2)
                             * xg ** pb[0] * yg ** pb[1] * zB ** pb[2]
                             * np.exp(-zb * rB2))
        # f_m with the e^{+i m phi} convention: chi = sum_m f_m e^{i m phi}
        self.f = np.fft.ifft(vals, axis=2)  # index k -> m = k (mod n_phi)
        self.mlist = np.fft.fftfreq(n_phi, 1.0 / n_phi).astype(int)

    def charge(self):
        # int chi dV = 2 pi * int f_0
        return 2 * np.pi * np.sum(self.w2d * self.f[:, :, 0].real)

    def overlap(self, other):
        # int chi chi' dV for two products on the SAME grid:
        # int e^{i m phi} e^{i m' phi} dphi = 2 pi delta_{m,-m'}
        acc = 0.0
        for k, m in enumerate(self.mlist):
            k2s = np.where(other.mlist == -m)[0]
            if len(k2s):
                acc += 2 * np.pi * np.sum(
                    self.w2d * (self.f[:, :, k] * other.f[:, :, k2s[0]]).real)
        return acc


class CoaxialKernel:
    """(chi1 | exp(-t^2 r12^2) | chi2) for coaxial PSC products; the geometry
    matrices are t-independent and cached across the whole t sweep."""

    def __init__(self, f1, f2):
        z1 = f1.z.ravel(); r1 = f1.rho.ravel()
        z2 = f2.z.ravel(); r2 = f2.rho.ravel()
        self.sep2 = (z1[:, None] - z2[None, :]) ** 2 + \
                    (r1[:, None] - r2[None, :]) ** 2
        self.rr2 = 2 * r1[:, None] * r2[None, :]
        self.channels = []
        w1 = f1.w2d.ravel(); w2 = f2.w2d.ravel()
        for k, m in enumerate(f1.mlist):
            k2s = np.where(f2.mlist == -m)[0]
            if len(k2s):
                self.channels.append((abs(m), w1 * f1.f[:, :, k].ravel(),
                                      w2 * f2.f[:, :, k2s[0]].ravel()))

    def __call__(self, t):
        E = np.exp(-t * t * self.sep2)
        acc = 0.0
        for m, g1, g2 in self.channels:
            W = ive(m, t * t * self.rr2) * E
            acc += (g1 @ W @ g2).real
        return (2 * np.pi) ** 2 * acc



def s_product_overlap(za, A, zb, B, zc, C, zd, D):
    """<chi1 chi2> for chi1 = phi_a phi_b, chi2 = phi_c phi_d (s functions)."""
    p, q = za + zb, zc + zd
    P = (za * A + zb * B) / p
    Q = (zc * C + zd * D) / q
    K = np.exp(-za * zb / p * np.sum((A - B) ** 2)) * np.exp(
        -zc * zd / q * np.sum((C - D) ** 2))
    return K * (np.pi / (p + q)) ** 1.5 * np.exp(
        -p * q / (p + q) * np.sum((P - Q) ** 2))


def linlog_grid(t_lin=3.0, n_lin=40, n_log=60, t_c=30.0):
    x, w = leggauss(n_lin)
    t1 = 0.5 * t_lin * (x + 1); w1 = 0.5 * t_lin * w
    x, w = leggauss(n_log)
    u = 0.5 * (np.log(t_c) - np.log(t_lin)) * (x + 1) + np.log(t_lin)
    t2 = np.exp(u); w2 = 0.5 * (np.log(t_c) - np.log(t_lin)) * w * t2
    return np.r_[t1, t2], 2 / np.sqrt(np.pi) * np.r_[w1, w2], np.pi / t_c**2


def main():
    print("=" * 72)
    print("1. PSC representation: charge and overlap of GTO products")
    print("=" * 72)
    R = 1.4
    za, zb = 0.9, 1.3
    A = np.array([0, 0, -R / 2]); B = np.array([0, 0, R / 2])
    # s*s product: analytic charge = Gaussian product formula
    p = za + zb
    K = np.exp(-za * zb / p * R * R)
    q_exact = K * (np.pi / p) ** 1.5
    for n in [24, 32, 48]:
        f = PSCProduct(za, (0, 0, 0), zb, (0, 0, 0), R, n_xi=n, n_eta=n)
        print(f"  n_xi=n_eta={n:3d}: ss charge rel err "
              f"{abs(f.charge() - q_exact) / q_exact:.2e}", flush=True)
    # px*px product charge: <x^2> of the product Gaussian:
    # int x^2 e^{-p(r-P)^2} = (pi/p)^{3/2}/(2p)  (P on z axis, x_P = 0)
    q_exact_pp = K * (np.pi / p) ** 1.5 / (2 * p)
    f = PSCProduct(za, (1, 0, 0), zb, (1, 0, 0), R, n_xi=48, n_eta=48)
    print(f"  px*px charge rel err: {abs(f.charge() - q_exact_pp) / q_exact_pp:.2e}")
    # m-truncation exactness: the pz*s product must have only m=0
    f = PSCProduct(za, (0, 0, 1), zb, (0, 0, 0), R, n_xi=16, n_eta=16)
    print(f"  pz*s: max |f_m|, m!=0: "
          f"{max(np.max(np.abs(f.f[:, :, k])) for k, m in enumerate(f.mlist) if m != 0) if len(f.mlist) > 1 else 0.0:.2e}")

    print()
    print("=" * 72)
    print("2. Coaxial Coulomb via t quadrature + Bessel m-coupling vs analytic")
    print("=" * 72)
    # (ss|ss) with both pairs on the same axis, separated segments
    zc, zd = 1.1, 0.6
    R2 = 1.0
    shift = 3.0  # second diatomic segment center offset along z
    C = np.array([0, 0, shift - R2 / 2]); D = np.array([0, 0, shift + R2 / 2])
    exact = eri_ssss(za, A, zb, B, zc, C, zd, D)

    class Shifted(PSCProduct):
        def __init__(self, *a, zshift=0.0, **kw):
            super().__init__(*a, **kw)
            self.z = self.z + zshift

    for n in [16, 24, 32]:
        f1 = PSCProduct(za, (0, 0, 0), zb, (0, 0, 0), R, n_xi=n, n_eta=n)
        f2 = Shifted(zc, (0, 0, 0), zd, (0, 0, 0), R2, zshift=shift, n_xi=n, n_eta=n)
        # t_c must stay within what the spatial grid resolves; the
        # truncated tail is restored by the delta correction (analytic here)
        t, w, tail = linlog_grid(t_c=20.0)
        kern = CoaxialKernel(f1, f2)
        val = sum(wi * kern(ti) for ti, wi in zip(t, w))
        val += tail * s_product_overlap(za, A, zb, B, zc, C, zd, D)
        print(f"  n={n:3d}: coaxial (ss|ss) rel err {abs(val - exact) / exact:.2e}",
              flush=True)

    print()
    print("=" * 72)
    print("3. Non-coaxial: uniform-phi grids, N_phi convergence")
    print("=" * 72)
    # rotate the second segment axis by 60 degrees about y, keep separation
    ang = np.pi / 3
    Rot = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0],
                    [-np.sin(ang), 0, np.cos(ang)]])
    C2 = shift * np.array([np.sin(ang) * 0, 0, 1])  # segment center stays on z
    Cr = C2 + Rot @ np.array([0, 0, -R2 / 2])
    Dr = C2 + Rot @ np.array([0, 0, R2 / 2])
    exact_rot = eri_ssss(za, A, zb, B, zc, Cr, zd, Dr)

    def cart_points(f, frame_origin, frame_rot, n_phi):
        """3D cloud (points, weights, values) of a PSC product, frame-mapped."""
        phis = 2 * np.pi * np.arange(n_phi) / n_phi
        pts, wts, vals = [], [], []
        chi_grid = np.zeros((f.z.shape[0], f.z.shape[1], n_phi))
        # reconstruct chi on the phi grid from its Fourier components
        for k, m in enumerate(f.mlist):
            chi_grid += (f.f[:, :, k][:, :, None]
                         * np.exp(1j * m * phis)[None, None, :]).real
        for k, ph in enumerate(phis):
            x = f.rho * np.cos(ph); y = f.rho * np.sin(ph)
            local = np.stack([x, y, f.z], axis=-1).reshape(-1, 3)
            pts.append(local @ frame_rot.T + frame_origin)
            wts.append((f.w2d * 2 * np.pi / n_phi).ravel())
            vals.append(chi_grid[:, :, k].ravel())
        return np.vstack(pts), np.concatenate(wts), np.concatenate(vals)

    t, w, tail = linlog_grid(n_lin=24, n_log=36, t_c=20.0)
    tail_term = tail * s_product_overlap(za, A, zb, B, zc, Cr, zd, Dr)
    f1 = PSCProduct(za, (0, 0, 0), zb, (0, 0, 0), R, n_xi=16, n_eta=16)
    f2 = PSCProduct(zc, (0, 0, 0), zd, (0, 0, 0), R2, n_xi=16, n_eta=16)
    for n_phi in [4, 6, 8, 12]:
        p1, w1, v1 = cart_points(f1, np.zeros(3), np.eye(3), n_phi)
        p2, w2, v2 = cart_points(f2, C2, Rot, n_phi)
        g1 = w1 * v1
        g2 = w2 * v2
        # the distance matrix is t-independent: build once, sweep t
        d2 = np.sum((p1[:, None, :] - p2[None, :, :]) ** 2, axis=2)
        val = tail_term + sum(
            wi * (g1 @ np.exp(-ti * ti * d2) @ g2) for ti, wi in zip(t, w))
        print(f"  N_phi={n_phi:3d}: non-coaxial (ss|ss) rel err "
              f"{abs(val - exact_rot) / exact_rot:.2e}", flush=True)

    print()
    print("=" * 72)
    print("4. Mixed representation: analytic GTO bra x PSC ket")
    print("=" * 72)
    # (ab| t |chi): bra s-pair analytic per grid point:
    # int e^{-p(r-P)^2} e^{-t^2(r-r2)^2} dr
    #   = (pi/(p+t^2))^{3/2} exp(-p t^2/(p+t^2) |P-r2|^2)
    P = (za * A + zb * B) / p
    Kb = np.exp(-za * zb / p * R * R)
    f2 = PSCProduct(zc, (0, 0, 0), zd, (0, 0, 0), R2, n_xi=24, n_eta=24)
    p2, w2, v2 = cart_points(f2, C2, Rot, 6)
    g2 = w2 * v2
    t, w, tail = linlog_grid(n_lin=24, n_log=36, t_c=20.0)
    val = tail * s_product_overlap(za, A, zb, B, zc, Cr, zd, Dr)
    for ti, wi in zip(t, w):
        d2 = np.sum((P[None, :] - p2) ** 2, axis=1)
        bra = Kb * (np.pi / (p + ti * ti)) ** 1.5 * np.exp(
            -p * ti * ti / (p + ti * ti) * d2)
        val += wi * (bra @ g2)
    print(f"  mixed (ss|ss) rel err: {abs(val - exact_rot) / exact_rot:.2e}", flush=True)

    print()
    print("done.")


if __name__ == "__main__":
    main()
