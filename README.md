# single-cell-em: ToRORd-Land in Python

A Python implementation of the ToRORd human ventricular cell model (Tomek et al. 2019)
coupled to the Land contraction model (Land et al. 2017), as used in Margara et al. (2021).
It simulates the action potential, calcium transient and active tension of a single
cardiomyocyte.

## Install

Requires Python 3.10 or later.

```bash
git clone https://github.com/jennyhelyanwe/cardiac-single-cell-em.git
cd cardiac-single-cell-em
pip install -e .
```

## Getting started

Run the examples from the repository root, in order:

```bash
python examples/00_check_install.py
```

This simulates two beats in a few seconds. You should see:

```
APD90 = 273 ms
Peak active tension = 24.0 kPa
```

```bash
python examples/01_quickstart.py
```

This compares a control cell with one where the L-type calcium current is halved,
prints APD90 and peak tension for each, and plots the action potential and active
tension. It takes one to two minutes.

## Run to steady state before trusting results

A cell model needs several beats to settle after you change something. Ion
concentrations inside the cell (sodium, potassium, calcium in the SR) change slowly,
and each beat shifts them a little until they reach a stable cycle. Results taken
before then reflect the transition, not the new condition.

The stored starting state (`get_starting_state`) is already at steady state for the
default model at a 1000 ms cycle length. As soon as you change a conductance, the
cycle length or the cell type, the model moves away from it.

To check you have reached steady state:

1. Run the model for an increasing number of beats (e.g. 20, 50, 100, 200).
2. Record your biomarkers (APD90, peak tension) from the last beat of each run.
3. Use the smallest number of beats after which they stop changing meaningfully
   (e.g. less than 1% between runs).

Do this once for each type of change you make, and report the number of beats
you used.

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
examples/        start here
experiments/     research scripts (not needed to get started)
```

## References

- Tomek J, et al. (2019) eLife 8:e48890. Reference MATLAB/CellML implementation:
  https://github.com/jtmff/torord
- Land S, et al. (2017) J Mol Cell Cardiol 106:68–83.
- Margara F, Wang ZJ, et al. (2021) Prog Biophys Mol Biol 159:58–74.
