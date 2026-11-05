"""Mass estimate from a top-view mask of the material on the spatula (Methods, Vision-Based Mass Estimation, Eq. 1).

    n_px = the number of mask pixels in the full frame of the top camera
    m = a * n_px ** b

a and b are fitted per material by least squares in log-log coordinates, on scoops weighed on a balance that returned
a mask (30 to 37 per material in the paper, Table 4). Every error the paper reports is a leave-one-out error: the
scoop it predicts was left out of the fit.
"""
from __future__ import annotations

import numpy as np


def pixel_count(mask) -> int:
    """n_px: the number of non-zero pixels of the mask."""
    return int(np.count_nonzero(mask))


def mass_from_pixels(n_px: int, a: float, b: float) -> float:
    """m = a * n_px ** b (Eq. 1); 0 mg for an empty mask."""
    return a * n_px ** b if n_px > 0 else 0.0


def fit_power_law(n_px, mass_mg) -> tuple[float, float]:
    """a, b of m = a * n_px ** b, by least squares in log-log coordinates."""
    b, log_a = np.polyfit(np.log(n_px), np.log(mass_mg), 1)
    return float(np.exp(log_a)), float(b)


def power_law_fit(x: np.ndarray, y: np.ndarray):
    """m = a * px^b by log-log least squares.

    Returns a, b, R2, in-sample abs errors, LOO abs errors, LOO predictions.
    R2 is computed in LOGARITHMIC coordinates, the same ones the fit minimises in, so it
    weights every trial equally over the 0.3-91 mg range instead of following the largest
    scoops (paper sec:stats).
    """
    b, la = np.polyfit(np.log(x), np.log(y), 1)
    a = float(np.exp(la))
    pred = a * x ** b
    log_y, log_pred = np.log(y), np.log(pred)
    r2 = 1.0 - np.sum((log_y - log_pred) ** 2) / np.sum((log_y - log_y.mean()) ** 2)
    loo = np.empty(len(x))
    loo_pred = np.empty(len(x))
    for k in range(len(x)):
        keep = np.ones(len(x), dtype=bool)
        keep[k] = False
        bk, lk = np.polyfit(np.log(x[keep]), np.log(y[keep]), 1)
        loo_pred[k] = np.exp(lk) * x[k] ** bk
        loo[k] = abs(loo_pred[k] - y[k])
    return a, float(b), float(r2), np.abs(pred - y), loo, loo_pred
