"""
01_quickstart.py

Compare a control ToRORd-Land endocardial cell with one where the L-type calcium
current (ICaL) is halved, and plot the action potential and active tension.

Run from the repository root:
    python examples/01_quickstart.py

Expected output (approximately):
    control       APD90 = 273 ms, peak Ta = 24.0 kPa
    ICaL x 0.5    APD90 = 237 ms, peak Ta =  3.8 kPa
"""
import matplotlib.pyplot as plt

from single_cell_em.models.torord_land import (
    get_starting_state, model_runner, get_currents_structure)
from single_cell_em.analysis.biomarkers import apd, tension_metrics

BCL = 1000.0     # cycle length (ms)
BEATS = 20       # beats to simulate; changing a conductance moves the model away
                 # from its stored steady state, so it needs a few beats to settle

conditions = {
    "control": 1.0,
    "ICaL x 0.5": 0.5,
}

fig, (ax_v, ax_ta) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))

for label, scale in conditions.items():
    params = {"bcl": BCL, "cellType": 0, "ICaL_Multiplier": scale}
    t, X, params = model_runner(get_starting_state("m_endo"), params,
                                beats=BEATS, ignore_first=BEATS - 1)
    out = get_currents_structure(t, X, params)

    apd90 = apd(out["time"], out["V"])
    ta_peak = tension_metrics(out["time"], out["Ta"])["Ta_peak"]
    print(f"{label:12s}  APD90 = {apd90:.0f} ms, peak Ta = {ta_peak:5.1f} kPa")

    ax_v.plot(out["time"], out["V"], label=label)
    ax_ta.plot(out["time"], out["Ta"], label=label)

ax_v.set_ylabel("Membrane potential (mV)")
ax_ta.set_ylabel("Active tension (kPa)")
ax_ta.set_xlabel("Time (ms)")
ax_v.legend(frameon=False)
for ax in (ax_v, ax_ta):
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("quickstart.png", dpi=150)
plt.show()
