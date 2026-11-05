"""Shared loading and the analysis constants that the paper quotes verbatim."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"        # paper/data: trial-level data; the run logs are downloaded here
IMAGES = HERE.parent / "output"    # paper/output: where the figure scripts write (not in git)
PHOTOS = DATA / "photos"           # the photographs of Figures 1 to 3, not published (paper/README.md)

MATERIALS = ["salt", "sugar", "polenta"]
LABEL = {"salt": "NaCl", "sugar": "Sugar", "polenta": "Polenta"}

# --- constants quoted in the paper, sec:stats (keep in sync) ----------------
EMPTY_SCOOP_MG = 0.15      # empty scoop: carried mass below this, which the balance cannot tell from an empty spatula.
                           # Basis from the balance alone, printed by table3_vision_fit.py: the most negative deposit
                           # loss, the readings of the empty scoops, and the gap above them (any value in it gives the
                           # same set). Empty scoops stay in f_depth and leave the vision and deposit statistics.
MISSEG_REL = 0.5           # mis-segmentation: LOO estimate off by > 50 % relative ...
MISSEG_ABS_MG = 1.0        # ... AND by > 1 mg absolute
SURFACE_JUMP_MM = 10.0     # surface-estimate failure: estimate differs from the run median by more than this
SMALL_SCOOP_MG = 10.0      # "scoops of up to 10 mg": deposit loss vs vision error in the Deposit Action results
# The deposit has no success criterion: it is reported as measured, the delivered fraction and the mass lost
# (table1_depth_mass.py, fig_deposit.py).

# --- constant quoted in the fig:surface_height caption ---------------------------
# Lip-to-floor distance used to turn depth below the lip into height above the floor, for BOTH methods.
# 54.0 mm is the probe script's conversion and the config's contact_z_min_m (-0.054, "glass-base floor");
# the vision script's own height column uses 56.70 mm, so its raw heights sit 2.7 mm above the probe's.
VIAL_FLOOR_BELOW_LIP_MM = 54.0
NS = 1e9


def load(material: str) -> dict[str, np.ndarray]:
    p = DATA / f"{material}_trials.csv"
    with p.open() as f:
        rows = list(csv.DictReader(f))
    out: dict[str, np.ndarray] = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        try:
            out[k] = np.array([float(v) for v in vals])
        except ValueError:
            out[k] = np.array(vals)
    out["iteration"] = out["iteration"].astype(int)
    out["surface_fail"] = surface_failures(out)
    out["empty_scoop"] = empty_scoops(out)
    return out


def surface_failures(d: dict[str, np.ndarray]) -> np.ndarray:
    """Trials whose surface estimate departs from the run median by > SURFACE_JUMP_MM."""
    z = d["surface_z_lip_mm"]
    if not np.any(np.isfinite(z)):
        return np.zeros(len(z), dtype=bool)
    med = np.nanmedian(z)
    return np.isfinite(z) & (np.abs(z - med) > SURFACE_JUMP_MM)


def empty_scoops(d: dict[str, np.ndarray]) -> np.ndarray:
    """Trials with a valid surface estimate whose carried mass is below EMPTY_SCOOP_MG.

    An outcome of the sampling action: the commanded offset was run and the scoop came back empty, so these
    trials stay in the f_depth statistics and leave the vision and deposit statistics (paper sec:stats).
    """
    return ~d["surface_fail"] & (d["carried_mg"] < EMPTY_SCOOP_MG)


def loaded_scoops(d: dict[str, np.ndarray]) -> np.ndarray:
    """The scoops the vision and deposit statistics use: a valid surface estimate and not an empty scoop."""
    return ~d["surface_fail"] & ~d["empty_scoop"]


# The power law m = a * n_px ** b and its leave-one-out fit (power_law_fit) are in the ssvision package
# (ssvision/mass.py), which the demo uses too.


def fmt(v: float, nd: int = 2) -> str:
    return "---" if not np.isfinite(v) else f"{v:.{nd}f}"
