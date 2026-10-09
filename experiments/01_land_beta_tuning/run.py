import numpy as np
from single_cell_em.models.torord_land import (
    get_starting_state, model_runner, get_currents_structure)
from single_cell_em.models.land_params import LAND2017
from single_cell_em.protocols.lambda_traces import isometric, ramp_hold_release
from single_cell_em.analysis.biomarkers import apd, tension_metrics
from single_cell_em.plotting.traces import plot_beat

BCL, BEATS = 1000.0, 50
runs = {}

for lname, lfn in (("isometric", isometric()),
                   ("dynamic",   ramp_hold_release(bcl=BCL))):
    p = {"bcl": BCL, "land": LAND2017, "lambda_fn": lfn, "cellType": 0}
    t, X, p = model_runner(get_starting_state("m_endo"), p,
                           beats=BEATS, ignore_first=BEATS - 1)
    c = get_currents_structure(t, X, p)
    runs[lname] = c
    m = tension_metrics(c["time"], c["Ta"])
    print(f"{lname:10s} APD90={apd(c['time'], c['V']):6.1f} ms  "
          f"Ta_peak={m['Ta_peak']:7.2f} kPa  TTP={m['Ta_TTP']:6.1f} ms  "
          f"lam range=[{c['lambda'].min():.3f}, {c['lambda'].max():.3f}]")

fig, _ = plot_beat(runs, tmax=600)
fig.savefig("traces.png", dpi=150)