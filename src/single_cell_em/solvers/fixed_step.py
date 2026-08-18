"""Explicit fixed-step drivers, for reproducing what a 3D solver does."""
import numpy as np
from ..models.torord_land import model_ToRORd_Land
from ..models.land_params import LAND2017

IZS, IZW = 47, 48   # ZETAS, ZETAW indices in the state vector


def run_fixed_step(X0, bcl, n_beats, dt, dt_couple=None,
                   lambda_fn=None, land=LAND2017,
                   zeta_method="euler", record_dt=1.0, **model_kwargs):
    """
    dt          : cell ODE step (ms)
    dt_couple   : interval at which lambda is refreshed and held (ms).
                  Defaults to dt. Set to 1-10 ms to mimic mechanics coupling.
    zeta_method : 'euler'       ZETAS/ZETAW stepped explicitly (unstable above
                                ~2*tau, i.e. ~4.9 ms with default Land params)
                  'exponential' exact update over the step, unconditionally stable
    """
    dt_couple = dt if dt_couple is None else dt_couple
    n_sub = max(1, int(round(dt_couple / dt)))

    X = np.array(X0, float)
    t = 0.0
    T, REC = [], []
    next_rec = 0.0

    while t < n_beats * bcl - 1e-9:
        lam, rate = (1.0, 0.0) if lambda_fn is None else lambda_fn(t)
        frozen = (lambda _t, _lam=lam, _r=rate: (_lam, _r))

        for _ in range(n_sub):
            dX = np.asarray(model_ToRORd_Land(t, X, 1, lambda_fn=frozen,
                                              land=land, **model_kwargs), float)
            zs, zw = X[IZS], X[IZW]
            X = X + dt * dX

            if zeta_method == "exponential":
                for idx, z0, c in ((IZS, zs, land.cds), (IZW, zw, land.cdw)):
                    e = np.exp(-c * dt)
                    X[idx] = z0 * e + (land.A * rate / c) * (1.0 - e)

            t += dt
            if t >= next_rec:
                T.append(t); REC.append(X.copy()); next_rec += record_dt

        if not np.all(np.isfinite(X)):
            raise FloatingPointError(f"diverged at t = {t:.2f} ms, dt = {dt}, "
                                     f"dt_couple = {dt_couple}")

    return np.array(T), np.array(REC)