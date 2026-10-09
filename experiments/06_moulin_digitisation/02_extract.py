"""Stage 2: seeded Viterbi extraction of the dashed mean from the LV panel."""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from single_cell_em.analysis.curve_extraction import (
    black_ink_cost, apply_seeds, click_seeds, viterbi_trace, AxisCalibration)

panel = np.array(Image.open("panel_lv.png").convert("RGB"))
cal   = AxisCalibration.load("calibration.json")
H, W  = panel.shape[:2]

cost = black_ink_cost(panel, val_thr=200, sat_thr=100, bridge=15, vsig=1.5)
BIG = cost.max()
cost[:, :80]  = BIG      # y-axis spine, cols 75-78
cost[:75, :]  = BIG      # "LV" title, rows 32-66
cost[412:, :] = BIG      # x-axis and tick labels
cost[:, 424:] = BIG      # beyond phase 1

seeds = click_seeds(panel, n=6)
np.save("seeds.npy", np.array(seeds))
path = viterbi_trace(apply_seeds(cost, seeds, tol=6), k_max=6, alpha=0.06)

cols  = np.arange(W)
keep  = (cols >= 80) & (cols <= 423)
phase = cal.to_x(cols)[keep]
Eff   = cal.to_y(path)[keep]
np.savez("moulin_lv_extracted.npz", phase=phase, Eff_mean=Eff)

i = int(np.argmin(Eff))
print(f"Eff at phase 0 : {Eff[0]:+.4f}   (must be near 0)")
print(f"trough         : {Eff[i]:+.4f} at phase {phase[i]:.3f}")
print(f"Eff at phase 1 : {Eff[-1]:+.4f}   (drift, expect about -0.03)")

fig, ax = plt.subplots(1, 3, figsize=(16, 5))
ax[0].imshow(cost, cmap="gray"); ax[0].set_title("Ink cost (dark = candidate)")
ax[1].imshow(panel)
ax[1].plot(cols[keep], path[keep], "r", lw=1.4, label="extracted mean")
ax[1].plot(*zip(*seeds), "co", ms=6, label="seeds")
ax[1].legend(fontsize=8); ax[1].set_title("Seeded path")
ax[2].plot(phase, Eff, "k", lw=1.6)
ax[2].axhline(0, color="k", lw=0.6)
ax[2].axhline(-0.15, color="r", ls="--", lw=0.8)
ax[2].plot(phase[i], Eff[i], "ro", ms=5)
ax[2].set_xlabel("Normalised cardiac cycle"); ax[2].set_ylabel("$E_{ff}$")
ax[2].grid(alpha=0.3)
ax[2].set_title(f"trough {Eff[i]:.3f} at phase {phase[i]:.3f}")
fig.tight_layout(); fig.savefig("02_extraction.png", dpi=150); plt.show()