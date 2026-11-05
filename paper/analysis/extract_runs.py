"""Extract one tidy per-trial CSV per material from the 2026-09-10 benchmark zarrs.

Usage:
    python3 extract_runs.py                # uses RUNS below
    python3 extract_runs.py salt=/path/a.zarr sugar=/path/b.zarr

Output: paper/data/<material>_trials.csv, one row per trial:
    iteration, sz_mm, carried_mg (source before - after), delivered_mg (goal net),
    lost_mg (carried - delivered), source_before_mg, source_after_mg,
    px_obs0..px_obs4 (spatula mask pixel counts; obs2 = after scoop = the estimate),
    surface_z_lip_mm (estimated surface height in the vial-lip frame, NaN if absent),
    t_* event timestamps in ns.

Conventions (verified against the run snapshots):
    - depth is scoop_contact/surface_offset_m[:, 2]; positive = ABOVE the estimated surface
    - weigh_source_vial/phase: 0 = before the scoop, 1 = after the deposit; gross readings
    - the material label comes from the folder name, NOT from config_json.material (always HS1)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import zarr

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

RUNS = {
    "salt": DATA / "HS1_vision" / "benchmark_2026-09-10-18-52.zarr",
    "sugar": DATA / "sugar_vision" / "benchmark_2026-09-10-13-46.zarr",
    "polenta": DATA / "polenta_vision" / "benchmark_2026-09-11-09-08.zarr",  # 40 trials, "yellow material on the scoop"
}

OBS = [0, 1, 2, 3, 4]  # trial start, before scoop, after scoop, before deposit, after deposit


def _arr(group, name, dtype=None):
    if name not in group:
        return None
    a = np.array(group[name])
    return a.astype(dtype) if dtype else a


def _mm_per_px(cfg: dict) -> float:
    (a, b) = np.asarray(cfg["surface"]["vial_clicks"], float)
    return float(cfg["vial"]["outer_clickable_length_mm"] / np.linalg.norm(b - a))


def _prompt_fractions(cfg: dict) -> dict:
    """Position of the two SAM prompt points along the click axis, as a fraction from click 0 to click 1."""
    (a, b) = np.asarray(cfg["surface"]["vial_clicks"], float)
    ax = (b - a) / np.linalg.norm(b - a)
    out = {}
    for name, key in (("positive", "material_click"), ("negative", "negative_click")):
        p = np.asarray(cfg["surface"][key], float)[0]
        out[name] = round(float(np.dot(p - a, ax) / np.linalg.norm(b - a)), 3)
    return out


def extract(zarr_path: Path) -> tuple[list[dict], dict]:
    g = zarr.open_group(str(zarr_path), mode="r")
    attrs = dict(g.attrs)
    cfg = json.loads(attrs["config_json"])
    E = g["events"]

    w = E["weigh_goal_vial"]
    w_it = _arr(w, "iteration", int)
    w_m = _arr(w, "mass_mg", float)
    w_t = _arr(w, "time_ns", int)
    goal = {int(i): (float(m), int(t)) for i, m, t in zip(w_it, w_m, w_t)}

    c = E["scoop_contact"]
    depth = {int(i): float(o[2]) * 1000.0 for i, o in zip(_arr(c, "iteration", int), _arr(c, "surface_offset_m"))}
    t_contact = {int(i): int(t) for i, t in zip(_arr(c, "iteration", int), _arr(c, "time_ns", int))}

    s = E["weigh_source_vial"]
    src: dict[int, dict[int, tuple[float, int]]] = {}
    for i, ph, m, t in zip(_arr(s, "iteration", int), _arr(s, "phase", int), _arr(s, "gross_mass_mg", float), _arr(s, "time_ns", int)):
        src.setdefault(int(i), {})[int(ph)] = (float(m), int(t))

    sp = E["spatula_observer"]
    spat: dict[int, dict[int, tuple[int, int]]] = {}
    for i, k, px, t in zip(_arr(sp, "iteration", int), _arr(sp, "observation_index", int), _arr(sp, "pixel_count", int), _arr(sp, "time_ns", int)):
        spat.setdefault(int(i), {})[int(k)] = (int(px), int(t))

    surf: dict[int, tuple[float, int]] = {}
    if "surface_observer" in E:
        su = E["surface_observer"]
        for i, pose, t in zip(_arr(su, "iteration", int), _arr(su, "T_vialLip_surface_pose7"), _arr(su, "time_ns", int)):
            surf[int(i)] = (float(pose[2]) * 1000.0, int(t))

    rows = []
    for i in sorted(goal):
        sb, tb = src.get(i, {}).get(0, (np.nan, 0))
        sa, ta = src.get(i, {}).get(1, (np.nan, 0))
        carried = sb - sa
        delivered = goal[i][0]
        row = {
            "iteration": i,
            "sz_mm": round(depth.get(i, np.nan), 2),
            "carried_mg": carried,
            "delivered_mg": delivered,
            "lost_mg": carried - delivered,
            "source_before_mg": sb,
            "source_after_mg": sa,
            "surface_z_lip_mm": surf.get(i, (np.nan, 0))[0],
        }
        for k in OBS:
            row[f"px_obs{k}"] = spat.get(i, {}).get(k, (-1, 0))[0]
        row["t_source_before_ns"] = tb
        row["t_surface_ns"] = surf.get(i, (np.nan, 0))[1]
        for k in OBS:
            row[f"t_obs{k}_ns"] = spat.get(i, {}).get(k, (0, 0))[1]
        row["t_contact_ns"] = t_contact.get(i, 0)
        row["t_source_after_ns"] = ta
        row["t_goal_ns"] = goal[i][1]
        rows.append(row)

    so = cfg["spatula_observer"]
    meta = {
        "created_at": attrs.get("created_at"),
        "resumed_at": attrs.get("resumed_at"),
        "n_trials": len(rows),
        "depths_mm": [d * 1000 for d in cfg["benchmark"]["sweep"]["scoop_sites"]["depths_m"]],
        "repeats": cfg["benchmark"]["repeats_per_point"],
        "order": cfg["benchmark"]["order"],
        "seed": cfg["benchmark"]["random_seed"],
        "prompts": so["prompts"],
        "thresholds": so["confidence_thresholds"],
        "edge_fit": cfg["surface"].get("edge_fit"),
        "ransac_tol_mm": cfg["surface"].get("ransac_tol_mm"),
        "surface_height_limit_mm": cfg["surface"].get("surface_height_limit_mm"),
        # side-camera geometry: two fixed reference clicks on the ends of the vial body set the axis and the scale
        "vial_clicks_px": cfg["surface"]["vial_clicks"],
        "vial_length_mm": cfg["vial"]["outer_clickable_length_mm"],
        "mm_per_px": _mm_per_px(cfg),
        "prompt_clicks_px": {"positive": cfg["surface"]["material_click"], "negative": cfg["surface"]["negative_click"]},
        "prompt_fraction_from_click0": _prompt_fractions(cfg),
        "deposit": cfg["subskills"].get("deposit_variant"),
        "scoop": cfg["subskills"].get("remove_variant"),
    }
    return rows, meta


def main(argv: list[str]) -> int:
    runs = dict(RUNS)
    for a in argv:
        k, v = a.split("=", 1)
        runs[k] = Path(v)
    DATA.mkdir(exist_ok=True)
    for material, path in runs.items():
        if not path.exists():
            print(f"[skip] {material}: {path} not found", file=sys.stderr)
            continue
        rows, meta = extract(path)
        out = DATA / f"{material}_trials.csv"
        with out.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        with (DATA / f"{material}_run_meta.json").open("w") as f:
            json.dump(meta, f, indent=1)
        n_nan = sum(1 for r in rows if not np.isfinite(r["carried_mg"]))
        print(f"{material:8s} {meta['n_trials']:3d} trials -> {out.name}  (created {meta['created_at']}, resumed {meta['resumed_at']}, "
              f"prompts {meta['prompts']}, thr {meta['thresholds']}, NaN carried: {n_nan})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
