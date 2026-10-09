"""Stage 3: convert the digitised Moulin E_ff trace into a lambda protocol.

Inputs : moulin_lv_extracted.npz  (from 02_extract.py)
         ../../data/lambda_rodero_mesh.csv
Outputs: moulin_lambda.npz, 03_lambda.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from single_cell_em.protocols.lambda_traces import (
    fourier_fit, xin_cubic, from_cardiax_csv)

BCL    = 1000.0
N_HARM = 5
CSV    = "../../data/lambda_rodero_mesh.csv"

# ---------------------------------------------------------------- load
d = np.load("moulin_lv_extracted.npz")
phase, Eff = d["phase"], d["Eff_mean"]
t = phase * BCL

cx = pd.read_csv(CSV)["avg(lambda_f)"].values
LAM_ED = float(cx.max())
print(f"lambda_ED from CardiaX peak stretch = {LAM_ED:.4f}\n")

# ---------------------------------------------------------------- drift
# The tracking residual is ramped out over the relengthening limb only, so the
# descent and the trough are untouched. Adds ~4.6e-5/ms to the recovery slope,
# about 6% of the peak physiological rate.
# ASSUMPTION: drift accumulates only after peak shortening. Feature-tracking
# drift generally accumulates from the first frame, which would mean the trough
# is slightly affected too. This choice preserves the measured peak contraction
# at the cost of that assumption; whole-cycle detrending makes the opposite
# trade (trough inflated by ~0.013 in lambda). State which was used.
i_trough = int(np.argmin(Eff))
ramp = np.zeros_like(Eff)
ramp[i_trough:] = Eff[-1] * (t[i_trough:] - t[i_trough]) / (t[-1] - t[i_trough])
Eff_c = Eff - ramp

lam_meas = LAM_ED * np.sqrt(1 + 2 * Eff)
lam_c    = LAM_ED * np.sqrt(1 + 2 * Eff_c)

print(f"trough E_ff  {Eff[i_trough]:+.5f} -> {Eff_c[i_trough]:+.5f}  (preserved)")
print(f"end E_ff     {Eff[-1]:+.5f} -> {Eff_c[-1]:+.5f}")
print(f"drift slope  {Eff[-1] / (t[-1] - t[i_trough]):.2e} /ms on recovery limb\n")

# ---------------------------------------------------------------- fit
moulin = fourier_fit(t, lam_c, BCL, n_harmonics=N_HARM)
print(f"lam(0) = {moulin(0.0)[0]:.4f}   (target {LAM_ED:.4f})\n")

print("harmonic order selection:")
for N in (3, 4, 5, 6, 8, 10, 14):
    fN  = fourier_fit(t, lam_c, BCL, n_harmonics=N)
    fit = np.array([fN(x)[0] for x in t])
    rt  = np.array([fN(x)[1] for x in np.linspace(0, BCL, 4001)])
    turns = int((np.diff(np.sign(rt)) != 0).sum())
    print(f"  N={N:2d}  RMSE {np.sqrt(((lam_c - fit) ** 2).mean()):.5f}  "
          f"max|rate| {np.abs(rt).max():.5f}/ms  lam(0) {fN(0.0)[0]:.4f}  "
          f"rate sign changes {turns}   {'<-- ringing' if turns > 2 else ''}")
print()

# ---------------------------------------------------------------- compare
from single_cell_em.protocols.lambda_traces import align_to_peak

cardiax_raw = from_cardiax_csv(CSV, source_duration=800.0, bcl=BCL)
cardiax, t_pk = align_to_peak(cardiax_raw, bcl=BCL)
print(f"CardiaX peak stretch at t = {t_pk:.1f} ms, shifted to t = 0\n")

protocols = {
    "moulin_lv":    moulin,                       # already ED-referenced
    "xin_cubic_m2": xin_cubic(mode=2, bcl=BCL),
    "cardiax":      cardiax,
}

tt = np.linspace(0, BCL, 4001)
summary = {}
for name, fn in protocols.items():
    lam  = np.array([fn(x)[0] for x in tt])
    rate = np.array([fn(x)[1] for x in tt])
    summary[name] = (lam, rate)
    print(f"{name:14s} lam [{lam.min():.4f}, {lam.max():.4f}]  "
          f"trough phase {tt[lam.argmin()] / BCL:.3f}  "
          f"max|rate| {np.abs(rate).max():.5f}/ms  "
          f"wrap jump {abs(fn(BCL - 1e-6)[0] - fn(0.0)[0]):.5f}")

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(2, 2, figsize=(12, 8))

ax[0, 0].plot(phase, Eff, ".", color="0.6", ms=2, label="extracted")
ax[0, 0].plot(phase, Eff_c, "r-", lw=1.2, label="post-trough corrected")
ax[0, 0].axhline(0, color="k", lw=0.6)
ax[0, 0].axvline(phase[i_trough], color="k", ls=":", lw=0.8)
ax[0, 0].set_xlabel("Normalised cardiac cycle")
ax[0, 0].set_ylabel("$E_{ff}$")
ax[0, 0].legend(fontsize=8)
ax[0, 0].set_title("Drift correction applied after the trough only")

ax[0, 1].plot(t, lam_meas, ".", color="0.6", ms=2, label="measured")
ax[0, 1].plot(t, lam_c, "k.", ms=2, label="corrected")
ax[0, 1].plot(tt, summary["moulin_lv"][0], "b-", lw=1.6,
              label=f"Fourier N={N_HARM}")
ax[0, 1].axhline(LAM_ED, color="g", ls=":", lw=0.8)
ax[0, 1].set_xlabel("Time (ms)")
ax[0, 1].set_ylabel("$\\lambda$")
ax[0, 1].legend(fontsize=8)
ax[0, 1].set_title("$E_{ff}$ to $\\lambda$ via $\\lambda_{ED}\\sqrt{1+2E_{ff}}$")

for name, (lam, rate) in summary.items():
    ax[1, 0].plot(tt, lam, lw=1.4, label=name)
    ax[1, 1].plot(tt, rate, lw=1.4, label=name)

ax[1, 0].axhline(0.87, color="k", ls=":", lw=0.8)
ax[1, 0].text(20, 0.872, " lambda_min", fontsize=7, va="bottom")
ax[1, 0].set_ylabel("$\\lambda$")
ax[1, 0].set_title("Protocol comparison")
ax[1, 1].axhline(0, color="k", lw=0.6)
ax[1, 1].set_ylabel("$d\\lambda/dt$ (ms$^{-1}$)")
ax[1, 1].set_title("Rate: drives ZETAS / ZETAW")
for a in ax[1, :]:
    a.set_xlabel("Time (ms)")
    a.legend(fontsize=8)
for a in ax.ravel():
    a.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("03_lambda.png", dpi=150)
plt.show()

# ---------------------------------------------------------------- save
np.savez("moulin_lambda.npz",
         t=t, phase=phase, Eff_raw=Eff, Eff_corrected=Eff_c,
         lam_corrected=lam_c, lambda_ED=LAM_ED, bcl=BCL,
         n_harmonics=N_HARM, i_trough=i_trough)