"""Operator-split driver: accurate EP, Land distortion stepped at the coupling dt.

This isolates the question actually being asked. Integrating the whole ToRORd
state explicitly measures the EP stability limit instead, and fails around
t = 350 ms when a concentration reaches zero.

State indices (verified): XS=43, XW=44, Ca_TRPN=45, TmBlocked=46,
                          ZETAS=47, ZETAW=48
"""
import numpy as np
from scipy.integrate import solve_ivp
from ..models.torord_land import model_ToRORd_Land
from ..models.land_params import LAND2017

IXS, IXW, IZS, IZW = 43, 44, 47, 48

__all__ = ["run_split", "IXS", "IXW", "IZS", "IZW"]


def run_split(X0, bcl, n_beats, dt_couple, lambda_fn=None, land=LAND2017,
              zeta_method="exponential", method="LSODA",
              rtol=1e-6, atol=1e-8, record=True, zeta_bound=10.0,
              **model_kwargs):
    """
    zeta_bound : abort if |ZETA| exceeds this. Physiological |ZETAW| peaks
                 around 0.12, so 10 is ~80x headroom. Without this the run
                 hangs rather than failing: a diverged ZETAW makes
                 gamma_rate_w = gamma_wu*|ZETAW| enormous, the system becomes
                 too stiff for the inner solver, and it never returns.
    """
    X = np.array(X0, float)
    T, REC = [], []
    for b in range(n_beats):
        t = 0.0
        while t < bcl - 1e-9:
            h = min(dt_couple, bcl - t)
            lam, rate = (1.0, 0.0) if lambda_fn is None else lambda_fn(t + b * bcl)
            frozen = (lambda _t, _l=lam, _r=rate: (_l, _r))
            zs0, zw0 = X[IZS], X[IZW]

            def rhs(tt, xx):
                return model_ToRORd_Land(tt, xx, 1, lambda_fn=frozen,
                                         land=land, **model_kwargs)

            sol = solve_ivp(rhs, [t, t + h], X, method=method,
                            rtol=rtol, atol=atol)
            if not sol.success:
                raise FloatingPointError(
                    f"inner solver failed at t={t:.1f} ms, "
                    f"dt_couple={dt_couple}, {zeta_method}: {sol.message}")
            X = sol.y[:, -1].copy()

            if zeta_method == "euler":
                X[IZS] = zs0 + h * (land.A * rate - land.cds * zs0)
                X[IZW] = zw0 + h * (land.A * rate - land.cdw * zw0)
            elif zeta_method == "exponential":
                for idx, z0, c in ((IZS, zs0, land.cds), (IZW, zw0, land.cdw)):
                    e = np.exp(-c * h)
                    X[idx] = z0 * e + (land.A * rate / c) * (1.0 - e)

            t += h
            zmax = max(abs(X[IZS]), abs(X[IZW]))
            if not np.all(np.isfinite(X)) or zmax > zeta_bound:
                raise FloatingPointError(
                    f"diverged at t={t:.1f} ms, dt_couple={dt_couple}, "
                    f"{zeta_method}: max|ZETA| = {zmax:.3g}")
            if record and b == n_beats - 1:
                T.append(t); REC.append(X.copy())
    return np.array(T), np.array(REC)