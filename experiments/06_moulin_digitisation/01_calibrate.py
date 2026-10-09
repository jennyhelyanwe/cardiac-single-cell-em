"""Stage 1: calibrate pixel -> data coordinates for the Moulin LV panel.

Input : a single cropped panel including both axes and their tick labels.
Output: panel_lv.png, calibration.json, 01_calibration.png
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from single_cell_em.analysis.curve_extraction import click_calibration, AxisCalibration

IMG      = "../../data/moulin_Eff.png"
RED_ROW  = None      # set to a row index to check the E_ff = -0.15 guide line,
                     # or leave None to auto-detect it
Y_RED    = -0.15

# ---------------------------------------------------------------- load
panel = np.array(Image.open(IMG).convert("RGB"))
H, W = panel.shape[:2]
print(f"panel is {W} x {H}")
assert panel.size and H > 20 and W > 20, f"empty or tiny image: {panel.shape}"
Image.fromarray(panel).save("panel_lv.png")

# ---------------------------------------------------------------- locate axes
dark = panel.mean(2) < 110
col_frac, row_frac = dark.sum(0) / H, dark.sum(1) / W


def _group(idx, gap=5):
    """Collapse runs of adjacent indices into their midpoints."""
    if len(idx) == 0:
        return []
    out, run = [], [int(idx[0])]
    for i in idx[1:]:
        if i - run[-1] <= gap:
            run.append(int(i))
        else:
            out.append(int(np.mean(run))); run = [int(i)]
    out.append(int(np.mean(run)))
    return out


spines    = _group(np.where(col_frac > 0.30)[0])
baselines = _group(np.where(row_frac > 0.30)[0])
print(f"y-axis spine(s) at x = {spines}")
print(f"x-axis / rules at y = {baselines}")
if not spines or not baselines:
    print("WARNING: no axis lines found. Lower the 0.30 threshold, or check "
          "the crop actually includes the spines.")

# ---------------------------------------------------------------- red guide line
rgb = panel.astype(int)
reddish = (rgb[:, :, 0] - np.maximum(rgb[:, :, 1], rgb[:, :, 2])) > 25
red_rows = _group(np.where(reddish.sum(1) > 0.10 * W)[0])
if RED_ROW is None and red_rows:
    RED_ROW = red_rows[0]
print(f"red guide line at row {RED_ROW}")

# ---------------------------------------------------------------- click
print("\nClick 4 TICK MARKS, in this order (not the number labels):")
print("  1. x = 0    tick on the bottom axis, at the y-axis spine")
print("  2. x = 1    rightmost tick on the bottom axis")
print("  3. y = 0    tick on the left spine, level with the '0' label")
print("  4. y = -0.3 tick on the left spine, level with the '-0.3' label\n")

cal = click_calibration(panel, x_vals=(0.0, 1.0), y_vals=(0.0, -0.3))
cal.save("calibration.json")

print(f"\npixels per unit phase  = {(cal.px1 - cal.px0):.1f}")
print(f"pixels per unit strain = {abs((cal.py1 - cal.py0) / 0.3):.1f}")
if RED_ROW is not None:
    got = cal.to_y(RED_ROW)
    print(f"CHECK: red guide line maps to E_ff = {got:.4f}, expected {Y_RED:.4f}, "
          f"error {got - Y_RED:+.4f}")
    if abs(got - Y_RED) > 0.01:
        print("  -> off by more than 0.01. You probably clicked a label rather "
              "than a tick. Re-run before continuing.")

# ---------------------------------------------------------------- verify
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(panel)
for v, lab in [(cal.px0, "x=0"), (cal.px1, "x=1")]:
    ax.axvline(v, color="r", lw=1)
    ax.text(v, 8, lab, color="r", fontsize=8, ha="center")
for v, lab in [(cal.py0, "y=0"), (cal.py1, "y=-0.3")]:
    ax.axhline(v, color="lime", lw=1)
    ax.text(8, v, lab, color="g", fontsize=8, va="bottom")
if RED_ROW is not None:
    ax.axhline(RED_ROW, color="cyan", ls="--", lw=1)
    ax.text(W - 8, RED_ROW, f"{cal.to_y(RED_ROW):.3f}", color="c",
            fontsize=8, ha="right", va="bottom")
ax.set_title("Calibration check: red/green lines must sit on the ticks")
fig.tight_layout()
fig.savefig("01_calibration.png", dpi=150)
plt.show()