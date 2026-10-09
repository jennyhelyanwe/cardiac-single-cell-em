"""Back-project the fitted lambda protocol onto the source figure."""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from single_cell_em.analysis.curve_extraction import AxisCalibration
from single_cell_em.protocols.lambda_traces import fourier_fit

panel = np.array(Image.open("panel_lv.png").convert("RGB"))
cal   = AxisCalibration.load("calibration.json")
d     = np.load("moulin_lambda.npz")
ext   = np.load("moulin_lv_extracted.npz")

BCL    = float(d["bcl"])
LAM_ED = float(d["lambda_ED"])
N      = int(d["n_harmonics"])

moulin = fourier_fit(d["t"], d["lam_corrected"], BCL, n_harmonics=N)

# dense fit -> E_ff -> pixels
tt   = np.linspace(0, BCL, 2001)
lam  = np.array([moulin(x)[0] for x in tt])
Eff  = 0.5 * ((lam / LAM_ED) ** 2 - 1.0)
px   = cal.from_x(tt / BCL)
py   = cal.from_y(Eff)

# raw extracted path, for comparison
px_r = cal.from_x(ext["phase"])
py_r = cal.from_y(ext["Eff_mean"])

fig, ax = plt.subplots(1, 2, figsize=(14, 6.5))
for a in ax:
    a.imshow(panel)
    a.set_xlim(0, panel.shape[1]); a.set_ylim(panel.shape[0], 0)

ax[0].plot(px_r, py_r, color="lime", lw=1.6, label="extracted (raw)")
ax[0].set_title("Viterbi extraction")
ax[0].legend(fontsize=9, loc="lower right")

ax[1].plot(px_r, py_r, color="lime", lw=1.0, alpha=0.5, label="extracted (raw)")
ax[1].plot(px, py, color="red", lw=2.0,
           label=f"fitted protocol, N={N}, $\\lambda_{{ED}}$={LAM_ED:.4f}")
ax[1].set_title("Final protocol back-projected")
ax[1].legend(fontsize=9, loc="lower right")

fig.tight_layout()
fig.savefig("04_overlay.png", dpi=170)
plt.show()

# quantitative agreement over the descent, before the drift correction bites
i_tr = int(d["i_trough"])
lam_at = np.array([moulin(x)[0] for x in d["t"][:i_tr]])
Eff_at = 0.5 * ((lam_at / LAM_ED) ** 2 - 1.0)
res_px = np.abs(cal.from_y(Eff_at) - cal.from_y(ext["Eff_mean"][:i_tr]))
print(f"pre-trough residual: mean {res_px.mean():.2f} px, max {res_px.max():.2f} px")
print(f"                     mean {res_px.mean() / 1114.4:.5f} in E_ff")