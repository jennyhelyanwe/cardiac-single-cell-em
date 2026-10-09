# single-cell-em: ToRORd-Land in Python

A Python implementation of the ToRORd human ventricular cell model (Tomek et al. 2019)
coupled to the Land contraction model (Land et al. 2017), as used in
Margara et al. (2021). It simulates the action potential, calcium transient and active
tension of a single cardiomyocyte.

## Install

Requires Python 3.10 or later.

```bash
git clone https://github.com/jennyhelyanwe/cardiac-single-cell-em.git
cd cardiac-single-cell-em
pip install -e .
```

## Quickstart: control vs. reduced L-type calcium current

```python
import matplotlib.pyplot as plt
from single_cell_em.models.torord_land import get_starting_state, model_runner, get_currents_structure
from single_cell_em.analysis.biomarkers import apd, tension_metrics

fig, (ax_v, ax_ta) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
for label, scale in [("control", 1.0), ("ICaL x 0.5", 0.5)]:
    p = {"bcl": 1000.0, "cellType": 0, "ICaL_Multiplier": scale}
    t, X, p = model_runner(get_starting_state("m_endo"), p, beats=20, ignore_first=19)
    c = get_currents_structure(t, X, p)
    m = tension_metrics(c["time"], c["Ta"])
    print(f"{label}: APD90 = {apd(c['time'], c['V']):.0f} ms, peak Ta = {m['Ta_peak']:.1f} kPa")
    ax_v.plot(c["time"], c["V"], label=label)
    ax_ta.plot(c["time"], c["Ta"], label=label)
ax_v.set_ylabel("Vm (mV)"); ax_ta.set_ylabel("Active tension (kPa)")
ax_ta.set_xlabel("Time (ms)"); ax_v.legend()
plt.tight_layout(); plt.show()
```

Each run takes under a minute on a laptop. Twenty beats is enough for a first look,
but check that your biomarkers have stopped changing before trusting them.
More beats (e.g. 200) get you closer to steady state.

## Changing the model

- **Ion channel conductances:** pass multipliers in the parameter dictionary, e.g.
  `"ICaL_Multiplier"`, `"IKr_Multiplier"`, `"IKs_Multiplier"`, `"INaL_Multiplier"`,
  `"Jup_Multiplier"` (SERCA). A value of 1 is baseline. This is how hormone or drug
  effects are represented.
- **Cell type:** `"cellType"`: 0 = endocardial, 1 = epicardial, 2 = mid-myocardial,
  with the matching starting state from `get_starting_state`.
- **Pacing:** `"bcl"` is the cycle length in ms.
- **Stretch:** `"lambda_fn"` takes a function of time for the fibre stretch ratio
  (see `protocols/lambda_traces.py`). The default is isometric.
- **Contraction parameters:** use the default `LAND2017` parameter set unless told otherwise.

## Repository layout

```
src/single_cell_em/
    models/      ToRORd-Land equations and Land parameter sets
    protocols/   stretch (lambda) protocols
    analysis/    biomarkers: APD, tension metrics
    solvers/     fixed-step solver for coupling studies
experiments/     example research scripts
```

## References

- Tomek J, et al. (2019) eLife 8:e48890. Reference MATLAB/CellML implementation:
  https://github.com/jtmff/torord
- Land S, et al. (2017) J Mol Cell Cardiol 106:68–83.
- Margara F, Wang ZJ, et al. (2021) Prog Biophys Mol Biol 159:58–74.
