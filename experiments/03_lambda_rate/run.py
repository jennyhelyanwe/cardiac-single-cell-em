"""Effect of the coupling time step on active tension, and the stability of
the Land distortion variables.

Compares three ways of advancing ZETAS/ZETAW at the mechanics coupling step:
  reference   - integrated accurately along with everything else
  euler       - explicit, predicted to diverge above 2/cdw = 4.93 ms
  exponential - exact update, predicted stable at any dt
The isometric control has dlambda/dt = 0 and should show no dt dependence
under any scheme, which is what localises the instability to the rate terms.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from single_cell_em.models.torord_land import get_starting_state
from single_cell_em.models.land_params import LAND2017
from single_cell_em.protocols.lambda_traces import (
    isometric, from_cardiax_csv, align_to_peak)
from single_cell_em.solvers.fixed_step import run_split, IXS, IXW, IZS, IZW
import warnings
warnings.filterwarnings("ignore", message="lsoda:")

BCL, BEATS = 1000.0, 2
COUPLE = [0.5, 1.0, 5.0, 10.0]
CSV = "../../data/lambda_rodero_mesh.csv"

cardiax, t_pk = align_to_peak(
    from_cardiax_csv(CSV, source_duration=800.0, bcl=BCL, mode="hold"), bcl=BCL)
print(f"CardiaX peak stretch shifted from t = {t_pk:.1f} ms to 0")
print(f"predicted Euler limit  2/cdw = {2 / LAND2017.cdw:.2f} ms")
print(f"                       2/cds = {2 / LAND2017.cds:.2f} ms\n")

X0 = get_starting_state("m_endo")
protocols = {"isometric": isometric(), "cardiax": cardiax}


def compute_Ta(X, lam, land=LAND2017):
    l0 = np.minimum(land.lambda_max, lam)
    Lf = np.maximum(0.0, 1 + land.beta_0 *
                    (l0 + np.minimum(land.lambda_min, l0)
                     - (1 + land.lambda_min)))
    return Lf * (land.Tref / land.dr) * (
        (X[:, IZS] + 1) * X[:, IXS] + X[:, IZW] * X[:, IXW])


rows = []
for pname, lfn in protocols.items():
    for zmeth in ("reference", "euler", "exponential"):
        for dtc in COUPLE:
            try:
                t, X = run_split(X0, BCL, BEATS, dtc, lambda_fn=lfn,
                                 land=LAND2017, zeta_method=zmeth,
                                 method="LSODA")
                lam = np.array([lfn(x)[0] for x in t])
                Ta = compute_Ta(X, lam)
                rows.append(dict(protocol=pname, method=zmeth, dt=dtc,
                                 Ta_peak=Ta.max(), Ta_min=Ta.min(),
                                 zetas_min=X[:, IZS].min(),
                                 zetaw_absmax=np.abs(X[:, IZW]).max(),
                                 diverged=False))
                print(f"{pname:10s} {zmeth:12s} dt={dtc:5.1f}  "
                      f"Ta_peak {Ta.max():8.2f}  |ZETAW|max "
                      f"{np.abs(X[:, IZW]).max():9.4f}")
            except FloatingPointError as e:
                rows.append(dict(protocol=pname, method=zmeth, dt=dtc,
                                 Ta_peak=np.nan, diverged=True))
                print(f"{pname:10s} {zmeth:12s} dt={dtc:5.1f}  DIVERGED  ({e})")

df = pd.DataFrame(rows)
df.to_csv("lambda_rate_convergence.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
styles = {"reference": ("k", "-"), "euler": ("C3", "--"), "exponential": ("C0", "-")}
for (pr, me), g in df.groupby(["protocol", "method"]):
    col, ls = styles[me]
    mk = "o" if pr == "cardiax" else "s"
    ax[0].plot(g.dt, g.Ta_peak, ls, color=col, marker=mk, ms=4,
               alpha=1.0 if pr == "cardiax" else 0.4, label=f"{pr}, {me}")
    ax[1].plot(g.dt, g.zetaw_absmax, ls, color=col, marker=mk, ms=4,
               alpha=1.0 if pr == "cardiax" else 0.4, label=f"{pr}, {me}")

for a, yl in zip(ax, ["Peak $T_a$ (kPa)", "max $|\\zeta_w|$"]):
    a.axvline(2 / LAND2017.cdw, color="r", ls=":", lw=1.2)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("Coupling time step $\\Delta t$ (ms)")
    a.set_ylabel(yl); a.grid(alpha=0.3, which="both")
    a.legend(fontsize=7, ncol=2)
ax[0].set_title(f"Peak tension vs coupling step\n"
                f"(red line: $2/c_{{dw}}$ = {2 / LAND2017.cdw:.2f} ms)")
ax[1].set_title("Distortion variable magnitude")
fig.tight_layout()
fig.savefig("fig1_dt_convergence.png", dpi=150)
plt.show()