"""The demo data in demo/data/: one stored trial per material (the ~10 mg scoop of Figure 5) and the calibration
scoops of the paper's power-law fit. Written by scripts/make_demo_data.py from the paper's run logs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

DATA = Path(__file__).resolve().parent / "data"
OUT = Path(__file__).resolve().parent / "out"
TRIALS = {"salt": "salt_t06", "sugar": "sugar_t04", "polenta": "polenta_t40"}
LABEL = {"salt": "NaCl", "sugar": "Sugar", "polenta": "Polenta"}
VIAL_FLOOR_BELOW_LIP_MM = 54.0   # Figure 7: depth below the lip -> height above the vial floor


def load_trial(stem: str, full: bool = False) -> dict:
    """The stored record of one trial and its four images (side, side_mask, top, top_mask).

    By default the images are the crops in demo/data, each with the full-frame position of its top-left pixel
    (<name>_origin). With full=True they are the uncropped frames in demo/data/full (origin 0, 0), which SAM 3 needs;
    they are downloaded with scripts/download_data.py demo."""
    rec = json.loads((DATA / f"{stem}.json").read_text())
    out = {"rec": rec}
    for name in ("side", "side_mask", "top", "top_mask"):
        if full:
            path, origin = DATA / rec["full_frames"][name]["file"], (0, 0)
            if not path.exists():
                raise SystemExit(f"{path} not found: the full frames are in the Zenodo record, "
                                 f"download them with: python scripts/download_data.py demo")
        else:
            path, origin = DATA / rec["files"][name]["file"], tuple(rec["files"][name]["origin_px"])
        img = Image.open(path)
        out[name] = np.asarray(img) > 0 if name.endswith("_mask") else np.asarray(img.convert("RGB"))
        out[f"{name}_origin"] = origin
    return out


def calibration_scoops(path: Path = DATA / "calibration.csv") -> dict[str, dict[str, np.ndarray]]:
    """Per material: the trial numbers, pixel counts and balance masses of the calibration scoops in `path`
    (columns pixel_count and mass_mg; material and trial if present)."""
    with Path(path).open() as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"pixel_count", "mass_mg"} <= set(rows[0]):
        raise SystemExit(f"{path}: needs the columns pixel_count and mass_mg (and optionally material, trial)")
    out: dict[str, dict[str, list]] = {}
    for k, r in enumerate(rows, 1):
        m = out.setdefault(r.get("material") or "material", {"trial": [], "pixel_count": [], "mass_mg": []})
        m["trial"].append(int(r["trial"]) if r.get("trial") else k)
        m["pixel_count"].append(float(r["pixel_count"]))
        m["mass_mg"].append(float(r["mass_mg"]))
    return {mat: {k: np.array(v) for k, v in m.items()} for mat, m in out.items()}
