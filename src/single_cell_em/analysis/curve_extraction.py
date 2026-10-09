"""Viterbi curve extraction from published figures.

Cost = per-pixel ink mismatch; transition = smoothness penalty. The smoothness
term is what bridges the gaps in a dashed line, so alpha is the main knob:
too low and the path jumps between curves, too high and it cuts corners.
"""
import json
import numpy as np
import matplotlib.pyplot as plt


# def black_ink_cost(rgb, val_thr=110, sat_thr=45, cap=1.0):
#     """Low cost on dark, unsaturated (black/grey) pixels; high on colour and white.
#     Returns cost in [0, cap]."""
#     a = rgb.astype(float)
#     mx, mn = a.max(2), a.min(2)
#     darkness = np.clip((val_thr - mx) / val_thr, 0, 1)
#     greyness = np.clip((sat_thr - (mx - mn)) / sat_thr, 0, 1)
#     return cap * (1.0 - darkness * greyness)


def viterbi_trace(cost, k_max=6, alpha=0.06):
    """One row index per column. k_max = max vertical jump per column (px)."""
    H, W = cost.shape
    D = np.full((H, W), np.inf)
    ptr = np.zeros((H, W), dtype=int)
    rows = np.arange(H)
    D[:, 0] = cost[:, 0]
    for j in range(1, W):
        best = D[:, j - 1].copy()
        arg = rows.copy()
        for d in range(1, k_max + 1):
            up = np.full(H, np.inf); up[d:] = D[:-d, j - 1] + alpha * d
            m = up < best; best[m] = up[m]; arg[m] = rows[m] - d
            dn = np.full(H, np.inf); dn[:-d] = D[d:, j - 1] + alpha * d
            m = dn < best; best[m] = dn[m]; arg[m] = rows[m] + d
        D[:, j] = best + cost[:, j]
        ptr[:, j] = arg
    path = np.zeros(W, dtype=int)
    path[-1] = int(D[:, -1].argmin())
    for j in range(W - 1, 0, -1):
        path[j - 1] = ptr[path[j], j]
    return path


def extract_curves(cost, n_curves=3, exclude_px=6, **kw):
    """Extract n curves by masking each found path and repeating.
    In a mean-plus-CI figure, sort by mean row and take the middle one."""
    c = cost.copy()
    paths = []
    for _ in range(n_curves):
        p = viterbi_trace(c, **kw)
        paths.append(p)
        for j, i in enumerate(p):
            lo, hi = max(0, i - exclude_px), min(c.shape[0], i + exclude_px + 1)
            c[lo:hi, j] = c.max()
    return sorted(paths, key=lambda p: p.mean())


class AxisCalibration:
    """Affine pixel -> data map, assuming axes are not rotated."""

    def __init__(self, px0, px1, x0, x1, py0, py1, y0, y1):
        self.px0, self.px1, self.x0, self.x1 = px0, px1, x0, x1
        self.py0, self.py1, self.y0, self.y1 = py0, py1, y0, y1

    def to_x(self, px):
        return self.x0 + (np.asarray(px) - self.px0) * (self.x1 - self.x0) / (self.px1 - self.px0)

    def to_y(self, py):
        return self.y0 + (np.asarray(py) - self.py0) * (self.y1 - self.y0) / (self.py1 - self.py0)

    def save(self, path):
        json.dump(self.__dict__, open(path, "w"), indent=2)

    @classmethod
    def load(cls, path):
        return cls(**json.load(open(path)))

    def from_x(self, x):
        return self.px0 + (np.asarray(x) - self.x0) * (self.px1 - self.px0) / (self.x1 - self.x0)

    def from_y(self, y):
        return self.py0 + (np.asarray(y) - self.y0) * (self.py1 - self.py0) / (self.y1 - self.y0)


def click_calibration(img, x_vals=(0.0, 1.0), y_vals=(0.0, -0.3)):
    """Click 4 points in order: x=x_vals[0], x=x_vals[1], y=y_vals[0], y=y_vals[1]."""
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.imshow(img)
    ax.set_title(f"Click: x={x_vals[0]}, then x={x_vals[1]}, "
                 f"then y={y_vals[0]}, then y={y_vals[1]}", fontsize=10)
    pts = fig.ginput(4, timeout=0)
    plt.close(fig)
    (px0, _), (px1, _), (_, py0), (_, py1) = pts
    return AxisCalibration(px0, px1, x_vals[0], x_vals[1],
                           py0, py1, y_vals[0], y_vals[1])

from scipy.ndimage import maximum_filter1d, gaussian_filter1d


def black_ink_cost(rgb, val_thr=200, sat_thr=100, bridge=15, vsig=1.5):
    """Low cost on dark, unsaturated pixels. `bridge` is a horizontal dilation
    that closes the gaps in a dashed line; without it a flat path is cheaper
    than the true curve. `vsig` softens the ink vertically so near-misses in
    the transition still cost less than background."""
    a = rgb.astype(float)
    mx, mn = a.max(2), a.min(2)
    score = (np.clip((val_thr - mx) / val_thr, 0, 1)
             * np.clip((sat_thr - (mx - mn)) / sat_thr, 0, 1))
    if bridge:
        score = maximum_filter1d(score, size=bridge, axis=1)
    if vsig:
        score = gaussian_filter1d(score, vsig, axis=0)
    return 1.0 - score / max(score.max(), 1e-9)


def apply_seeds(cost, seeds, tol=6, penalty=10.0):
    """Constrain the path to pass within tol rows of each (col, row) seed."""
    c = cost.copy()
    big = c.max()
    for col, row in seeds:
        col, row = int(round(col)), int(round(row))
        mask = np.ones(c.shape[0], dtype=bool)
        mask[max(0, row - tol): row + tol + 1] = False
        c[mask, col] = big * penalty
    return c


def click_seeds(img, n=6):
    """Click n points along the target curve, left to right."""
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img)
    ax.set_title(f"Click {n} points along the DASHED MEAN, left to right.\n"
                 "Include the start, the trough, and the end.", fontsize=10)
    pts = fig.ginput(n, timeout=0)
    plt.close(fig)
    return [(x, y) for x, y in pts]

