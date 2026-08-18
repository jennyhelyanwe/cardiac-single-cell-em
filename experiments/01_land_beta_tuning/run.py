import numpy as np
from single_cell_em.models.torord_land import get_starting_state, model_runner
from single_cell_em.models.land_params import REGISTRY
from single_cell_em.protocols.lambda_traces import isometric, ramp_hold_release
from single_cell_em.analysis.biomarkers import apd, tension_metrics

BCL, BEATS = 1000.0, 50
IV, ITA = 0, None   # Ta is not a state; recompute via get_currents_structure

for pname, land in REGISTRY.items():
    for lname, lfn in (("isometric", isometric()),
                       ("dynamic",   ramp_hold_release(bcl=BCL))):
        p = {"bcl": BCL, "land": land, "lambda_fn": lfn, "cellType": 0}
        t, X, p = model_runner(get_starting_state("m_endo"), p,
                               beats=BEATS, ignore_first=BEATS - 1)
        from single_cell_em.models.torord_land import get_currents_structure
        c = get_currents_structure(t, X, p)
        m = tension_metrics(c["time"], c["Ta"])
        print(f"{pname:14s} {lname:10s} APD90={apd(c['time'], c['V']):6.1f} ms  "
              f"Ta_peak={m['Ta_peak']:7.2f} kPa  TTP={m['Ta_TTP']:6.1f} ms  "
              f"hash={land.hash()}")