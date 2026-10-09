"""Analytical and data-derived lambda(t) protocols.

Every protocol is a closure taking t in ms and returning (lam, dlam_dt),
so the Land distortion terms never depend on finite differencing.
"""
import numpy as np
from scipy.interpolate import CubicSpline, UnivariateSpline

__all__ = ["isometric", "ramp_hold_release", "sinusoid", "xin_cubic",
           "periodic_spline", "from_cardiax_csv", "from_ef_strain", "from_trace",
           "fourier_fit", "shift_phase", "scale_timing", "align_to_peak"]

# --- parametric ------------------------------------------------------------

def isometric(lam_rest=1.0):
    def f(t):
        return lam_rest, 0.0
    return f


def ramp_hold_release(lam_rest=1.0, dlam=-0.10, t_on=50.0, tau_on=30.0,
                      t_off=250.0, tau_off=60.0, bcl=1000.0):
    """Smooth shortening then relengthening. t_on / t_off are the knobs for
    the lambda-timing sweep."""
    def f(t):
        tt    = np.mod(t, bcl)
        s_on  = 0.5 * (1 + np.tanh((tt - t_on) / tau_on))
        s_off = 0.5 * (1 + np.tanh((tt - t_off) / tau_off))
        lam   = lam_rest + dlam * (s_on - s_off)
        d_on  = 0.5 / tau_on  * (1 - np.tanh((tt - t_on) / tau_on) ** 2)
        d_off = 0.5 / tau_off * (1 - np.tanh((tt - t_off) / tau_off) ** 2)
        return float(lam), float(dlam * (d_on - d_off))
    return f


def sinusoid(lam_rest=1.0, amp=0.05, period=1000.0, phase=0.0):
    w = 2 * np.pi / period
    def f(t):
        return lam_rest + amp * np.sin(w * t + phase), amp * w * np.cos(w * t + phase)
    return f


def xin_cubic(mode=2, bcl=1000.0):
    """Polynomial lambda(t) from the MATLAB implementation Jenny and Xin wrote.

    Faithful to the original, including two quirks worth knowing about:
      - not periodic: mode 2 jumps ~0.009 at each beat boundary
      - normalises by a literal 1000 ms, not by bcl
    Keep for replication; do not reuse for timing sweeps.
    """
    P = {0: [0.0, 0.0, 0.0, 1.0],
         1: [-2.30163145848258, 3.75764888453562,
             -1.47838300037886, 0.929123799870826],   # HFpEF, no stretch
         2: [-2.27433215072184, 4.10389824231470,
             -1.83869939727443, 1.08642479298336]}    # with stretch
    p  = P[mode]
    dp = [3 * p[0], 2 * p[1], p[2]]
    def f(t):
        x = np.mod(t, bcl) / 1000.0
        return float(np.polyval(p, x)), float(np.polyval(dp, x)) / 1000.0
    return f


# --- data-derived ----------------------------------------------------------

def periodic_spline(t_data, lam_data, bcl, enforce_periodic=True):
    """Exact, C2, periodic by construction. Preferred for data-derived lambda."""
    t_data   = np.asarray(t_data, float)
    lam_data = np.asarray(lam_data, float)
    drift = lam_data[-1] - lam_data[0]
    if enforce_periodic and abs(drift) > 1e-12:
        lam_data = lam_data - drift * (t_data - t_data[0]) / (t_data[-1] - t_data[0])
    cs = CubicSpline(t_data, lam_data, bc_type="periodic")
    ds = cs.derivative()
    def f(t):
        tt = np.mod(t, bcl)
        return float(cs(tt)), float(ds(tt))
    return f


def from_cardiax_csv(path, source_duration=800.0, bcl=1000.0,
                     column="avg(lambda_f)", mode="scale"):
    """mode='scale': stretch the cycle to fill bcl (shape kept, rates slowed by
       bcl/source_duration). mode='hold': keep absolute timings, hold the final
       value out to bcl. Use 'hold' whenever the rate magnitude matters."""
    import pandas as pd
    lam = pd.read_csv(path)[column].values
    t = np.linspace(0.0, source_duration, len(lam))
    if mode == "scale":
        t = t * (bcl / source_duration)
    elif mode == "hold" and bcl > source_duration:
        t   = np.append(t, bcl)
        lam = np.append(lam, lam[-1])
    return periodic_spline(t, lam, bcl)


def from_ef_strain(t_data, Eff, bcl, lambda_ED=1.038):
    """Imaging strain referenced to end diastole -> stretch referenced to the
    slack state, via F composition. lambda_ED restores the inflation phase that
    an ED-referenced imaging study cannot see."""
    lam = lambda_ED * np.sqrt(1.0 + 2.0 * np.asarray(Eff, float))
    return periodic_spline(np.asarray(t_data, float), lam, bcl)


def from_trace(t_data, lam_data, bcl=1000.0, smooth=1e-6):
    """Non-periodic smoothing spline. Use only for noisy data where you want
    the smoothing; periodic_spline is the better default."""
    s  = UnivariateSpline(t_data, lam_data, s=smooth, k=3)
    ds = s.derivative()
    def f(t):
        tt = np.mod(t, bcl)
        return float(s(tt)), float(ds(tt))
    return f

def fourier_fit(t_data, lam_data, bcl, n_harmonics=6, weights=None,
                return_coeffs=False):
    t_data = np.asarray(t_data, float)
    lam_data = np.asarray(lam_data, float)
    w = 2 * np.pi / bcl
    cols = [np.ones_like(t_data)]
    for k in range(1, n_harmonics + 1):
        cols += [np.cos(k * w * t_data), np.sin(k * w * t_data)]
    A, y = np.array(cols).T, lam_data
    if weights is not None:
        sw = np.sqrt(np.asarray(weights, float))
        A, y = A * sw[:, None], y * sw
    c, *_ = np.linalg.lstsq(A, y, rcond=None)

    def f(t):
        tt = np.mod(t, bcl)
        lam = c[0]
        rate = 0.0
        for k in range(1, n_harmonics + 1):
            a, b = c[2 * k - 1], c[2 * k]
            lam += a * np.cos(k * w * tt) + b * np.sin(k * w * tt)
            rate += k * w * (-a * np.sin(k * w * tt) + b * np.cos(k * w * tt))
        return float(lam), float(rate)

    return (f, c) if return_coeffs else f

def shift_phase(fn, shift, bcl=1000.0):
    """Move a protocol earlier (shift<0) or later in the cycle. Rate magnitude
    is preserved, only the timing moves. This is the knob for the lambda-timing
    sweep."""
    def f(t):
        return fn(np.mod(t - shift, bcl))
    return f


def scale_timing(fn, factor, bcl=1000.0, source_bcl=None):
    """Compress (factor<1) or dilate the protocol about t=0. Rates scale by
    1/factor, so this changes dlambda/dt as well as timing, deliberately.
    Use source_bcl when the wrapped protocol has a different native cycle
    length from the target bcl."""
    src = bcl if source_bcl is None else source_bcl
    def f(t):
        lam, rate = fn(np.mod(t / factor, src))
        return lam, rate / factor
    return f

def align_to_peak(fn, bcl=1000.0, n=10001, mode="max"):
    """Shift a protocol so its peak (or trough) stretch sits at t=0.

    Moulin-style imaging protocols are referenced to end diastole, so their
    t=0 is peak stretch by construction. Use this to put simulation-derived
    protocols on the same convention before comparing trough timings.

    Only meaningful for periodic protocols: shifting a non-periodic one moves
    its wrap discontinuity into the middle of the cycle.
    """
    tt = np.linspace(0.0, bcl, n)
    v = np.array([fn(x)[0] for x in tt])
    t_ref = tt[int(np.argmax(v) if mode == "max" else np.argmin(v))]
    jump = abs(fn(bcl - 1e-6)[0] - fn(0.0)[0])
    if jump > 1e-4:
        import warnings
        warnings.warn(f"align_to_peak: protocol has a wrap discontinuity of "
                      f"{jump:.4f}; shifting moves it to phase "
                      f"{((0 - t_ref) % bcl) / bcl:.3f}")
    return shift_phase(fn, -t_ref, bcl=bcl), t_ref