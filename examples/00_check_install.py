from single_cell_em.models.torord_land import get_starting_state, model_runner, get_currents_structure
from single_cell_em.analysis.biomarkers import apd, tension_metrics

t, X, p = model_runner(get_starting_state("m_endo"), {"bcl": 1000.0}, beats=2, ignore_first=1)
c = get_currents_structure(t, X, p)
print(f"APD90 = {apd(c['time'], c['V']):.0f} ms")
print(f"Peak active tension = {tension_metrics(c['time'], c['Ta'])['Ta_peak']:.1f} kPa")
