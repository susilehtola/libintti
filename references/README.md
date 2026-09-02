<!-- SPDX-License-Identifier: BSD-3-Clause -->
# Symbolic reference derivations

The C++ tests validate the integral engine against analytic values. Where those
values are closed forms, the "magic numbers" are **derived symbolically here
with SymPy** rather than asserted by hand, so every oracle is reproducible and a
drift between the code and the mathematics is caught. Each script asserts its
own results and is run in CI (the `references` job).

Run them directly:

```sh
python references/sympy_slater.py
python references/sympy_three_electron.py
```

## `sympy_slater.py` — Slater / STO references (`tests/test_sto.cpp`)

| Derived quantity | Closed form | Test |
| --- | --- | --- |
| 1s self-overlap | `pi/zeta^3` | overlap |
| n-s self-overlap | `4 pi (2n)!/(2 zeta)^{2n+1}` | higher-n overlap |
| 2p self-overlap | `pi/zeta^5` | l>0 overlap |
| Slater density potential | `(1/R)[1-(1+zeta R)e^{-2 zeta R}]` | `sto_slater_potential` |
| 1s self-repulsion | `5 zeta/8` | 2-centre Coulomb, `R=0` |
| delta-tail weight | `(8 pi/zeta^3)[1-(1+u)e^{-u}]`, `u=zeta^2/4 s_c` | delta-tail |
| 2-centre 1s Coulomb (equal exp.) | Roothaan `1/R - e^{-2zR}(1/R+11z/8+3z^2R/4+z^3R^2/6)`; limits checked | `sto_coulomb_2c` |

## `sympy_three_electron.py` — three-electron references (`tests/test_threeel.cpp`)

The six-index integral is `G_{abcdef} = <a(1)b(2)c(3)|r12^{-1} r13^{-1}|d(1)e(2)f(3)>`.

| Derived quantity | Closed form | Test |
| --- | --- | --- |
| one-centre Coulomb-Coulomb | `4 zeta/3` (via `int rho_G V_G^2`, `V_G=erf(sqrt(2z) r)/r`) | `OneCentreAnalytic` |
| Gaussian-geminal one-centre | `pi^{9/2}/[(a+LQ+LS)(a+g)(a+d)]^{3/2}` (via `det A` of the quadratic form) | `GaussianGeminalOneCentreAnalytic` |

The remaining oracles are non-closed-form and validated numerically instead:
the general s-type reference values come from the NumPy prototypes in
`prototype/`, and the one- and two-electron matrices are checked against PySCF
(`prototype/pyscf_*_validation.py`); PySCF is not installed in CI.
