import numpy as np


def apd(t, v, pct=90):
    vmax, vmin = v.max(), v.min()
    thr = vmax - (pct / 100.0) * (vmax - vmin)
    i_up = int(np.argmax(v))
    depol = np.where(v[:i_up] <= thr)[0]
    t0 = t[depol[-1]] if len(depol) else t[0]
    rep = np.where(v[i_up:] <= thr)[0]
    if not len(rep):
        return np.nan
    return t[i_up + rep[0]] - t0


def tension_metrics(t, Ta):
    i = int(np.argmax(Ta))
    peak, base = Ta[i], Ta.min()
    out = {"Ta_peak": peak, "Ta_rest": base, "Ta_TTP": t[i] - t[0]}
    for pct in (50, 90):
        thr = peak - (pct / 100.0) * (peak - base)
        rel = np.where(Ta[i:] <= thr)[0]
        out[f"Ta_RT{pct}"] = (t[i + rel[0]] - t[i]) if len(rel) else np.nan
    return out