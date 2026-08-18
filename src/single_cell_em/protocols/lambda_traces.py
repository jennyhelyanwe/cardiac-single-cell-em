"""Analytical lambda(t) protocols. Each returns (lam, dlam_dt) so the
distortion terms never depend on finite differencing."""
import numpy as np


def isometric(lam_rest=1.0):
    def f(t):
        return lam_rest, 0.0
    return f


def ramp_hold_release(lam_rest=1.0, dlam=-0.10, t_on=50.0, tau_on=30.0,
                      t_off=250.0, tau_off=60.0, bcl=1000.0):
    """Smooth shortening then relengthening, periodic with the pacing cycle."""
    def f(t):
        tt    = np.mod(t, bcl)
        s_on  = 0.5 * (1 + np.tanh((tt - t_on) / tau_on))
        s_off = 0.5 * (1 + np.tanh((tt - t_off) / tau_off))
        lam   = lam_rest + dlam * (s_on - s_off)
        d_on  = 0.5 / tau_on  * (1 - np.tanh((tt - t_on) / tau_on) ** 2)
        d_off = 0.5 / tau_off * (1 - np.tanh((tt - t_off) / tau_off) ** 2)
        return lam, dlam * (d_on - d_off)
    return f


def sinusoid(lam_rest=1.0, amp=0.05, period=1000.0, phase=0.0):
    w = 2 * np.pi / period
    def f(t):
        return lam_rest + amp * np.sin(w * t + phase), amp * w * np.cos(w * t + phase)
    return f


def from_trace(t_data, lam_data, bcl=1000.0, smooth=True):
    """Wrap a measured or simulated lambda trace. Derivative from a spline,
    not from raw differences, to avoid injecting noise into dZETA."""
    from scipy.interpolate import UnivariateSpline
    s = UnivariateSpline(t_data, lam_data, s=1e-6 if smooth else 0.0, k=3)
    ds = s.derivative()
    def f(t):
        tt = np.mod(t, bcl)
        return float(s(tt)), float(ds(tt))
    return f