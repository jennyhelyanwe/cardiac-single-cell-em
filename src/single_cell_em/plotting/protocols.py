import numpy as np
import matplotlib.pyplot as plt
from ..models.land_params import LAND2017


def lfac(lam, land=LAND2017):
    l0 = np.minimum(land.lambda_max, lam)
    return np.maximum(0.0, 1 + land.beta_0 *
                      (l0 + np.minimum(land.lambda_min, l0) - (1 + land.lambda_min)))


def ca50_eff(lam, land=LAND2017):
    return land.ca50 + land.beta_1 * np.minimum(0.2, lam - 1.0)


def plot_protocols(protocols, bcl=1000.0, n=4001, land=LAND2017, figsize=(7, 10)):
    t = np.linspace(0, bcl, n)
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=figsize)
    for label, fn in protocols.items():
        lam  = np.array([fn(ti)[0] for ti in t])
        rate = np.array([fn(ti)[1] for ti in t])
        ax[0].plot(t, lam, lw=1.4, label=label)
        ax[1].plot(t, rate, lw=1.4)
        ax[2].plot(t, lfac(lam, land), lw=1.4)
        ax[3].plot(t, ca50_eff(lam, land), lw=1.4)
        print(f"{label:16s} lam [{lam.min():.4f}, {lam.max():.4f}]  "
              f"max|rate| {np.abs(rate).max():.5f}/ms  "
              f"Lfac_min {lfac(lam, land).min():.3f}  "
              f"jump@wrap {abs(fn(bcl - 1e-6)[0] - fn(0.0)[0]):.5f}")

    ax[0].axhline(land.lambda_min, color="k", ls=":", lw=0.8)
    ax[0].text(bcl * 0.02, land.lambda_min, " lambda_min", fontsize=7, va="bottom")
    for a, yl in zip(ax, ["$\\lambda$ (-)", "$d\\lambda/dt$ (ms$^{-1}$)",
                          "$L_{fac}$ (-)", "$Ca_{50}^{eff}$ ($\\mu$M)"]):
        a.set_ylabel(yl, fontsize=9); a.grid(alpha=0.3)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[3].set_xlabel("Time (ms)")
    ax[0].legend(fontsize=8, frameon=False)
    fig.align_ylabels(ax); fig.tight_layout()
    return fig, ax