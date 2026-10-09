"""Effect of the three lambda protocols on active tension.

Post hoc decomposition: Ta = Lfac * (Tref/dr) * ((ZETAS+1)*XS + ZETAW*XW)
so Lfac and the distortion terms can each be switched off after the fact.
The ca50 length dependence (beta_1) acts through XS/XW and cannot be removed
post hoc; set RUN_BETA1_ZERO to isolate it with a second pass.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import replace

from single_cell_em.models.torord_land import (
    get_starting_state, model_runner, get_currents_structure)
from single_cell_em.models.land_params import LAND2017
from single_cell_em.protocols.lambda_traces import (
    isometric, xin_cubic, from_cardiax_csv, fourier_fit, align_to_peak)
from single_cell_em.analysis.biomarkers import apd, tension_metrics
from single_cell_em.plotting.traces import plot_beat

BCL, BEATS = 1000.0, 50
CSV = "../../data/lambda_rodero_mesh.csv"
IXS, IXW, IZS, IZW = 43, 44, 47, 48
RUN_BETA1_ZERO = False          # doubles runtime; isolates the ca50 effect

# ---------------------------------------------------------------- protocols
d = np.load("../06_moulin_digitisation/moulin_lambda.npz")
moulin = fourier_fit(d["t"], d["lam_corrected"], BCL,
                     n_harmonics=int(d["n_harmonics"]))
cardiax, t_pk = align_to_peak(from_cardiax_csv(CSV, source_duration=800.0,
                                               bcl=BCL), bcl=BCL)
print(f"CardiaX peak stretch shifted from t = {t_pk:.1f} ms to 0\n")

protocols = {"isometric":    isometric(lam_rest=1.0),
             "moulin_lv":    moulin,
             "xin_cubic_m2": xin_cubic(mode=2, bcl=BCL),
             "cardiax":      cardiax}


def run(lfn, land):
    p = {"bcl": BCL, "land": land, "lambda_fn": lfn, "cellType": 0}
    t, X, p = model_runner(get_starting_state("m_endo"), p,
                           beats=BEATS, ignore_first=BEATS - 1)
    if p.get("isFailed"):
        return None, None, None
    return get_currents_structure(t, X, p), X[-1], t[-1]


def decompose(Xb, tb, lfn, land):
    """Ta as simulated, with Lfac forced to 1, and with distortion removed."""
    lam = np.array([lfn(x)[0] for x in tb])
    l0 = np.minimum(land.lambda_max, lam)
    Lf = np.maximum(0.0, 1 + land.beta_0 *
                    (l0 + np.minimum(land.lambda_min, l0) - (1 + land.lambda_min)))
    XS, XW = Xb[:, IXS], Xb[:, IXW]
    ZS, ZW = Xb[:, IZS], Xb[:, IZW]
    k = land.Tref / land.dr
    return {"Ta":          Lf * k * ((ZS + 1) * XS + ZW * XW),
            "Ta_noLfac":   1.0 * k * ((ZS + 1) * XS + ZW * XW),
            "Ta_noZeta":   Lf * k * (XS),
            "Lfac_min":    Lf.min(),
            "ZETAS_min":   ZS.min(),
            "ZETAW_absmax": np.abs(ZW).max()}


# ---------------------------------------------------------------- run
runs, rows = {}, []
for name, lfn in protocols.items():
    c, Xb, tb = run(lfn, LAND2017)
    if c is None:
        print(f"{name}: FAILED"); continue
    runs[name] = c
    dec = decompose(Xb, tb, lfn, LAND2017)
    m = tension_metrics(c["time"], c["Ta"])
    row = {"protocol": name,
           "APD90": apd(c["time"], c["V"]),
           "Cai_peak_uM": c["Cai"].max() * 1000,
           "lam_min": c["lambda"].min(),
           "Ta_peak": m["Ta_peak"],
           "Ta_TTP": m["Ta_TTP"],
           "Ta_RT90": m["Ta_RT90"],
           "Ta_peak_noLfac": dec["Ta_noLfac"].max(),
           "Ta_peak_noZeta": dec["Ta_noZeta"].max(),
           "Lfac_min": dec["Lfac_min"],
           "ZETAS_min": dec["ZETAS_min"],
           "ZETAW_absmax": dec["ZETAW_absmax"]}
    if RUN_BETA1_ZERO:
        c0, Xb0, tb0 = run(lfn, replace(LAND2017, beta_1=0.0))
        row["Ta_peak_noCa50"] = (tension_metrics(c0["time"], c0["Ta"])["Ta_peak"]
                                 if c0 is not None else np.nan)
    rows.append(row)

df = pd.DataFrame(rows).set_index("protocol")
iso = df.loc["isometric", "Ta_peak"]
df["Ta_vs_isometric_%"] = 100 * (df["Ta_peak"] / iso - 1)
pd.set_option("display.width", 200, "display.max_columns", 30)
print(df.round(4).to_string())
df.to_csv("protocol_tension.csv")

# ---------------------------------------------------------------- figures
fig, _ = plot_beat(runs, tmax=BCL)
fig.savefig("04_traces.png", dpi=150)

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
x = np.arange(len(df))
ax[0].bar(x - 0.25, df["Ta_peak"], 0.25, label="as simulated")
ax[0].bar(x,        df["Ta_peak_noLfac"], 0.25, label="length-dependence removed")
ax[0].bar(x + 0.25, df["Ta_peak_noZeta"], 0.25, label="velocity-dependence removed")
ax[0].set_xticks(x); ax[0].set_xticklabels(df.index, rotation=20)
ax[0].set_ylabel("Peak $T_a$ (kPa)"); ax[0].legend(fontsize=8)
ax[0].set_title("Where the tension goes")

for name, c in runs.items():
    ax[1].plot(c["lambda"], c["Ta"], lw=1.3, label=name)
ax[1].set_xlabel("$\\lambda$"); ax[1].set_ylabel("$T_a$ (kPa)")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
ax[1].set_title("Tension-length loops")

fig.tight_layout(); fig.savefig("04_decomposition.png", dpi=150); plt.show()