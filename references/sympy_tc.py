#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026 Susi Lehtola
#
# Independent oracle for the transcorrelated NON-HERMITIAN two-body integral
#   <pq| grad_1 u_12 . grad_1 |rs>,  u = sum_k c_k e^{-g_k r12^2}
# (Haupt thesis Eq 2.29-2.31; the one genuinely new TC integral -- M-TC). This
# evaluates it by DIRECT Gaussian integration of the operator applied to the ket
# orbital chi_r -- no integration by parts, no libintti -- so it independently
# checks libintti's by-parts reduction to plain geminal integrals with
# derivative-shifted orbitals:
#   <pq|grad u.grad|rs> = - sum_k c_k sum_d [ G_k((d_d p)(d_d r),(q,s))
#                                            + G_k((p)(d_d^2 r),(q,s)) ].
# Directions decouple, so each axis is a 2D Gaussian integral (fast & exact).
import sympy as sp

i1, i2 = sp.symbols('i1 i2', real=True)

def axis_integral(al, P, la, be, Q, lb, gam, R, lc, de, S, ld, g, moment):
    # int di1 di2 (i1-P)^la (i1-R)^lc (i2-Q)^lb (i2-S)^ld * moment
    #   * e^{-al(i1-P)^2 -gam(i1-R)^2 -g(i1-i2)^2 -be(i2-Q)^2 -de(i2-S)^2}
    poly = (i1-P)**la * (i1-R)**lc * (i2-Q)**lb * (i2-S)**ld * moment
    gauss = sp.exp(-al*(i1-P)**2 - gam*(i1-R)**2 - g*(i1-i2)**2
                   - be*(i2-Q)**2 - de*(i2-S)**2)
    return sp.integrate(sp.integrate(poly*gauss, (i1,-sp.oo,sp.oo)),
                        (i2,-sp.oo,sp.oo))

def tc_nonherm(a,P,lp, b,Q,lq, c,R,lr, d,S,ls, gems):
    """<pq|grad u.grad|rs>. lp,lq,lr,ls are (lx,ly,lz) tuples; centres P,Q,R,S
    are 3-tuples; gems is a list of (g_k, c_k). p<->a (elec1), q<->b (elec2),
    r<->c (elec1, the ket orbital acted on), s<->d (elec2)."""
    ax = ['x','y','z']
    total = sp.Integer(0)
    for (g, ck) in gems:
        for k in range(3):  # gradient direction
            term = sp.Integer(0)
            # d_k u = -2 g (i1-i2); d_k chi_r = lr_k (i1-R)^{lr_k-1} - 2c (i1-R)^{lr_k+1}
            # -> operator along axis k: (-2 g (i1-i2)) * d_k[(i1-R)^{lr_k} e^{-c(..)^2}]/e
            # We fold the -2g(i1-i2) as a moment on axis k, and apply d_k to chi_r's power.
            # d_k chi_r contributes two power terms:
            for (dpow, dcoef) in ([(lr[k]-1, lr[k])] if lr[k] > 0 else []) + [(lr[k]+1, -2*c)]:
                prod = sp.sympify(dcoef)
                for j in range(3):
                    if j == k:
                        prod *= axis_integral(a,P[j],lp[j], b,Q[j],lq[j],
                                              c,R[j],dpow, d,S[j],ls[j], g,
                                              sp.sympify(-2*g)*(i1-i2))
                    else:
                        prod *= axis_integral(a,P[j],lp[j], b,Q[j],lq[j],
                                              c,R[j],lr[j], d,S[j],ls[j], g,
                                              sp.Integer(1))
                term += prod
            total += ck*term
    return total

if __name__ == '__main__':
    P=(sp.Integer(0),)*3; R=(sp.Rational(1,2),sp.Integer(0),sp.Integer(0))
    Q=(sp.Integer(0),sp.Integer(1),sp.Integer(0))
    S=(sp.Rational(1,5),sp.Rational(9,10),sp.Rational(3,10))
    a,b,c,d = sp.Rational(12,10),sp.Rational(9,10),sp.Rational(8,10),sp.Rational(11,10)
    gems=[(sp.Rational(7,10), sp.Integer(1))]
    # ssss
    v=tc_nonherm(a,P,(0,0,0), b,Q,(0,0,0), c,R,(0,0,0), d,S,(0,0,0), gems)
    print("ssss:", float(v))
    # r = p_x (ket orbital acted on carries angular momentum)
    v=tc_nonherm(a,P,(0,0,0), b,Q,(0,0,0), c,R,(1,0,0), d,S,(0,0,0), gems)
    print("r=px:", float(v))
    # p = p_y, r = p_x
    v=tc_nonherm(a,P,(0,1,0), b,Q,(0,0,0), c,R,(1,0,0), d,S,(0,0,0), gems)
    print("p=py,r=px:", float(v))
    # geminal sum
    gems2=[(sp.Rational(2,10),sp.Rational(1,2)),(sp.Rational(8,10),sp.Rational(3,10)),(sp.Integer(3),sp.Rational(1,5))]
    v=tc_nonherm(a,P,(0,0,0), b,Q,(0,0,0), c,R,(0,0,0), d,S,(0,0,0), gems2)
    print("ssss gemsum:", float(v))
