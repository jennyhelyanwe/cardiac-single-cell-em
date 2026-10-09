import matplotlib.pyplot as plt
import numpy as np

PANELS = [("V",      "Membrane potential (mV)"),
          ("Cai",    "Intracellular Ca$^{2+}$ ($\\mu$M)"),
          ("Ta",     "Active tension (kPa)"),
          ("lambda", "$\\lambda$ (-)")]


def plot_beat(runs, t0=None, figsize=(7, 9), tmax=None):
    """runs: dict of label -> currents dict from get_currents_structure."""
    fig, axes = plt.subplots(len(PANELS), 1, sharex=True, figsize=figsize)
    for label, c in runs.items():
        t = c["time"] - (c["time"][0] if t0 is None else t0)
        for ax, (key, ylab) in zip(axes, PANELS):
            y = c[key] * 1000.0 if key == "Cai" else c[key]
            ax.plot(t, y, lw=1.4, label=label)
            ax.set_ylabel(ylab, fontsize=9)
            ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time (ms)")
    if tmax:
        axes[-1].set_xlim(0, tmax)
    axes[0].legend(fontsize=8, frameon=False)
    fig.align_ylabels(axes)
    fig.tight_layout()
    return fig, axes