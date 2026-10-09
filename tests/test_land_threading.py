# tests/test_land_threading.py
from dataclasses import replace
import numpy as np
from single_cell_em.models.land_params import LAND2017
from single_cell_em.models.torord_land import (
    get_starting_state, model_runner, get_currents_structure)


def _peak_Ta(land, beats=20):
    p = {"bcl": 1000.0, "land": land, "cellType": 0}
    t, X, p = model_runner(get_starting_state("m_endo"), p,
                           beats=beats, ignore_first=beats - 1)
    return get_currents_structure(t, X, p)["Ta"].max()


def test_Tref_scales_tension_linearly():
    base = _peak_Ta(LAND2017)
    dbl  = _peak_Ta(replace(LAND2017, Tref=2 * LAND2017.Tref))
    assert np.isclose(dbl, 2 * base, rtol=1e-6), f"land arg not threaded: {base}, {dbl}"


def test_ca50_changes_tension():
    assert not np.isclose(_peak_Ta(replace(LAND2017, ca50=1.2)),
                          _peak_Ta(LAND2017), rtol=1e-3)