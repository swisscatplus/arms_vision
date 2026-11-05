"""Write the demo data (demo/data/) from the paper's trial-level data and run logs.

    python scripts/make_demo_data.py                     # calibration.csv, and the three demo trials if the run logs are present
    python scripts/make_demo_data.py salt=/path/to/benchmark_2026-09-10-18-52.zarr ...   # run logs elsewhere
    python scripts/make_demo_data.py --trial salt:23     # one more trial (then: python demo/estimate_mass.py --sample salt_t23)

Writes:
    demo/data/calibration.csv        the pixel count after the scoop and the balance mass of every scoop the paper's
                                     power-law fit uses (Table 4): a valid surface estimate, not an empty scoop, and a
                                     mask (paper/analysis/table3_vision_fit.py). From paper/data, no run logs needed.
For each trial (needs the run logs, scripts/download_data.py runs):
    demo/data/<material>_t<NN>_side.png        side-camera frame of the surface estimate, cropped (RGB)
    demo/data/<material>_t<NN>_side_mask.png   its stored SAM 3 mask, same crop (0/255)
    demo/data/<material>_t<NN>_top.png         top-camera frame after the scoop (read 2), cropped (RGB)
    demo/data/<material>_t<NN>_top_mask.png    its stored SAM 3 mask, same crop (0/255)
    demo/data/<material>_t<NN>.json            what the robot stored for the trial (surface pose, reference and prompt
                                               pixels, pixel count, balance masses, the run's settings) and the crops
    demo/data/full/<material>_t<NN>_*.png      the same four images uncropped (1936 x 1216), for the --sam3 option of
                                               the demo scripts; in the Zenodo record as demo_full_frames.zip

The crops are fixed boxes around the material in the vial and the blade of the spatula, widened when a mask reaches
past them, so that each crop holds its whole mask with a margin. The demo scripts add the crop origin back, so the
computation runs in full-frame pixels and gives the same result as on the full frame.

Channel order in the run logs: side frames (events/surface_observer/image) are stored BGR, top frames
(events/spatula_observer/image_camera44) RGB. All PNGs are written RGB. The masks are stored 0/1 and are written
0/255; nothing else is changed.

The default trials are the ~10 mg scoop of each material, the side views of Figure 5 (top row): the detected trial
with a valid surface estimate whose carried mass is closest to 10 mg (fig3_examples.pick_trial).
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper" / "analysis"))
from common import load, loaded_scoops  # noqa: E402
from extract_runs import RUNS  # noqa: E402

OUT = ROOT / "demo" / "data"
FULL = OUT / "full"
TRIALS = {"salt": 6, "sugar": 4, "polenta": 40}   # fig3_examples.pick_trial(load(m), 10.0) for each material
AFTER_SCOOP = 2                                    # spatula read 2: after the scoop, the read the estimate uses
SIDE_BOX = (320, 170, 1420, 930)                   # x0, y0, x1, y1: the material in the vial and its surface
TOP_BOX = (240, 380, 760, 720)                     # the blade of the spatula
MARGIN_PX = 8                                      # a mask reaching past its box widens the crop to this margin


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_png(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(np.ascontiguousarray(arr)).save(path, optimize=True)


def crop_box(mask: np.ndarray, box) -> tuple[int, int, int, int]:
    """`box`, widened to hold the mask's bounding box plus MARGIN_PX, within the frame."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    x0, y0, x1, y1 = box
    if len(xs):
        x0, y0 = min(x0, xs.min() - MARGIN_PX), min(y0, ys.min() - MARGIN_PX)
        x1, y1 = max(x1, xs.max() + 1 + MARGIN_PX), max(y1, ys.max() + 1 + MARGIN_PX)
    return max(int(x0), 0), max(int(y0), 0), min(int(x1), w), min(int(y1), h)


def write_calibration() -> Path:
    """The fit set of the paper's power law, with the values as written in paper/data/<material>_trials.csv."""
    path = OUT / "calibration.csv"
    rows = []
    for material in TRIALS:
        d = load(material)
        fit = loaded_scoops(d) & (d["px_obs2"] > 0) & (d["carried_mg"] > 0)     # as table3_vision_fit.analyse
        keep = {int(i) for i in d["iteration"][fit]}
        with (ROOT / "paper" / "data" / f"{material}_trials.csv").open() as f:
            for r in csv.DictReader(f):
                if int(r["iteration"]) in keep:
                    rows.append([material, r["iteration"], r["px_obs2"], r["carried_mg"]])
    OUT.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["material", "trial", "pixel_count", "mass_mg"])
        w.writerows(rows)
    print(f"calibration: {len(rows)} scoops ({', '.join(f'{m} {sum(r[0] == m for r in rows)}' for m in TRIALS)}) -> "
          f"{path.relative_to(ROOT)}")
    return path


def extract(material: str, trial: int, zarr_path: Path) -> dict:
    import zarr
    from scipy.spatial.transform import Rotation

    g = zarr.open_group(str(zarr_path), mode="r")
    cfg = json.loads(dict(g.attrs)["config_json"])
    E = g["events"]
    stem = f"{material}_t{trial:02d}"

    su = E["surface_observer"]
    i = int(np.where(np.array(su["iteration"]) == trial)[0][0])
    side = np.array(su["image"][i])[:, :, ::-1]                    # stored BGR -> RGB
    side_mask = (np.array(su["mask"][i]) > 0).astype(np.uint8) * 255
    pose = np.array(su["T_vialLip_surface_pose7"][i], dtype=float)
    angle_x = float(Rotation.from_quat(pose[3:]).as_rotvec()[0])  # the pose is a rotation about x_lip only

    sp = E["spatula_observer"]
    it, oi = np.array(sp["iteration"]), np.array(sp["observation_index"])
    j = int(np.where((it == trial) & (oi == AFTER_SCOOP))[0][0])
    top = np.array(sp["image_camera44"][j])                         # stored RGB
    top_mask = (np.array(sp["mask_camera44"][j]) > 0).astype(np.uint8) * 255

    d = load(material)
    k = int(np.where(d["iteration"] == trial)[0][0])

    files, full = {}, {}
    FULL.mkdir(parents=True, exist_ok=True)
    boxes = {"side": crop_box(side_mask > 0, SIDE_BOX), "top": crop_box(top_mask > 0, TOP_BOX)}
    for name, arr in (("side", side), ("side_mask", side_mask), ("top", top), ("top_mask", top_mask)):
        p = FULL / f"{stem}_{name}.png"
        save_png(arr, p)
        full[name] = {"file": f"full/{p.name}", "sha256": sha256(p)}
        x0, y0, x1, y1 = boxes[name.split("_")[0]]
        p = OUT / f"{stem}_{name}.png"
        save_png(arr[y0:y1, x0:x1], p)
        files[name] = {"file": p.name, "sha256": sha256(p), "origin_px": [x0, y0], "size_px": [x1 - x0, y1 - y0]}

    rec = {
        "material": material,
        "trial": trial,
        "run": {"zarr": zarr_path.name, "created_at": dict(g.attrs).get("created_at")},
        "commanded_offset_sz_mm": float(d["sz_mm"][k]),
        "balance": {"carried_mg": float(d["carried_mg"][k]), "delivered_mg": float(d["delivered_mg"][k]),
                    "note": "carried = source vial before - after (differential weighing); paper/data/<material>_trials.csv"},
        "files": files,
        "files_note": ("crops of the camera frames: origin_px is the crop's top-left pixel in the full frame, which every "
                       "pixel coordinate below refers to"),
        "full_frames": {"size_px": [int(side.shape[1]), int(side.shape[0])], **full,
                        "note": "the uncropped frames, for --sam3: in the Zenodo record (python scripts/download_data.py demo)"},
        "channel_order_in_zarr": {"side": "BGR", "top": "RGB", "png": "RGB (both)"},
        "surface_stored": {
            "vial_clicks_image_px": np.array(su["vial_clicks_image"][i], dtype=float).tolist(),
            "vial_clicks_note": "two fixed reference pixels on the vial axis, click 0 at the lip end",
            "vial_length_mm": float(cfg["vial"]["outer_clickable_length_mm"]),
            "material_click_px": cfg["surface"]["material_click"],
            "negative_click_px": cfg["surface"]["negative_click"],
            "edge_fit": cfg["surface"].get("edge_fit"),
            "ransac_tol_mm": cfg["surface"].get("ransac_tol_mm"),
            "T_vialLip_surface_pose7": pose.tolist(),
            "pose7_note": "[x, y, z, qx, qy, qz, qw] of the surface frame in the vial-lip frame, metres, float32 in the zarr",
            "surface_z_lip_mm": float(pose[2] * 1000.0),
            "tilt_about_x_lip_deg": float(np.degrees(angle_x)),
            "plane_center_image_px": np.array(su["plane_center_image"][i], dtype=float).tolist(),
            "mask_area_px": int(np.count_nonzero(side_mask)),
        },
        "mass_stored": {
            "observation_index": AFTER_SCOOP,
            "pixel_count": int(np.array(sp["pixel_count"])[j]),
            "success": bool(np.array(sp["success"])[j]),
            "notes": str(np.array(sp["notes"])[j]),
            "csv_px_obs2": int(d["px_obs2"][k]),
            "prompts": cfg["spatula_observer"]["prompts"],
            "confidence_thresholds": cfg["spatula_observer"]["confidence_thresholds"],
            "paddings_px": cfg["spatula_observer"].get("paddings"),
            "single_detection_only": cfg["spatula_observer"].get("single_detection_only"),
        },
    }
    (OUT / f"{stem}.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def main(argv: list[str]) -> int:
    runs = dict(RUNS)
    trials = dict(TRIALS)
    args = list(argv)
    if "--trial" in args:
        spec = args[args.index("--trial") + 1]
        args = [a for a in args if a not in ("--trial", spec)]
        m, t = spec.split(":")
        trials = {m: int(t)}
    for a in args:
        k, v = a.split("=", 1)
        runs[k] = Path(v)
    write_calibration()
    for material, trial in trials.items():
        if not runs[material].exists():
            print(f"[skip] {material} trial {trial}: run log {runs[material]} not found "
                  f"(python scripts/download_data.py runs)", file=sys.stderr)
            continue
        rec = extract(material, trial, runs[material])
        s, m = rec["surface_stored"], rec["mass_stored"]
        print(f"{material} trial {trial}: side z = {s['surface_z_lip_mm']:.3f} mm, tilt {s['tilt_about_x_lip_deg']:.3f} deg; "
              f"top {m['pixel_count']} px; carried {rec['balance']['carried_mg']:.3f} mg; crops "
              f"{rec['files']['side']['size_px']} and {rec['files']['top']['size_px']} px")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
