"""Recompute the SAM 3 masks of the three runs with the service's processing, and compare them with the stored masks.

    python paper/analysis/tools/sam3_recompute.py [--device auto|cpu|mps|cuda] [--sam3-model facebook/sam3]
                                                  [salt=PATH sugar=PATH polenta=PATH]

Needs the run logs (python scripts/download_data.py runs, or the paths as above) and SAM 3 (pip install -e ".[sam3]"
and the gated facebook/sam3 weights). The masks are recomputed with ssvision.sam3_pipeline, the processing of the
segmentation service that produced the stored masks (arms_sam3 commit bc48b4f; docs/segmentation_service.md), with
the prompts and thresholds stored in each run log. For each of the 40 trials of each run:

  top view after the scoop (spatula read 2, the read the mass estimate uses):
    - the mask as the service computed it: each stage keeps the union of its detections above 0.4;
    - its IoU with the stored mask, both pixel counts, and the change in the mass estimate with the material's a and
      b of Table 4 (the fit of paper/analysis/table3_vision_fit.py);
    - the number of detections above the threshold in each stage. Where a stage had more than one, the mask is also
      recomputed with single_detection_only=True, the behaviour of the service since 2026-09-23 and what the robot's
      configuration asked for, and its change in mass against the union is given.
  side view of the surface estimate:
    - the mask as the service computed it, the best of three candidates by predicted IoU;
    - its IoU with the stored mask, and the change in depth and tilt against the stored estimate (ssvision.surface);
    - the predicted IoU of each candidate and the IoU of each with the stored mask. Where the two best predicted IoUs
      are within 0.01 (a near tie), the choice can differ between this machine and the service's GPU.

Prints a summary (its record, made on Apple silicon with MPS: paper/expected/sam3_recompute.txt) and writes one row
per read to paper/output/sam3_recompute_top.csv and sam3_recompute_side.csv (records: paper/expected/). On another
device or with other library versions the masks can differ by a few pixels, so the record is a reference, not a
check: no number of the paper depends on it, since the analysis uses the stored masks.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")     # before torch: CPU for any op MPS lacks

import numpy as np
from PIL import Image

ANALYSIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS))
from common import IMAGES, LABEL, MATERIALS, load, loaded_scoops  # noqa: E402
from extract_runs import RUNS  # noqa: E402

AFTER_SCOOP = 2           # spatula read 2: after the scoop
NEAR_TIE = 0.01           # predicted-IoU gap below which the best-of-three choice can flip between machines
EDGE = 0.01               # a score this close above the threshold can fall below it on another device
EVIDENCE = 0.01           # the union beats the top detection alone, in IoU with the stored mask, by more than this


def table4_power_law(material: str) -> tuple[float, float]:
    """a, b of Table 4: the fit set of table3_vision_fit.analyse (loaded scoops with a mask)."""
    from ssvision.mass import power_law_fit
    d = load(material)
    fit = loaded_scoops(d) & (d["px_obs2"] > 0) & (d["carried_mg"] > 0)
    a, b, *_ = power_law_fit(d["px_obs2"][fit], d["carried_mg"][fit])
    return a, b


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else 1.0


def stored_surface(su, i: int) -> tuple[float, float]:
    """Depth (mm) and tilt (deg) of the stored surface pose, a rotation about x_lip (as scripts/make_demo_data.py)."""
    from scipy.spatial.transform import Rotation
    pose = np.array(su["T_vialLip_surface_pose7"][i], dtype=float)
    return pose[2] * 1000.0, float(np.degrees(Rotation.from_quat(pose[3:]).as_rotvec()[0]))


def top_view(processor, model, device, rgb, cfg, single: bool):
    from ssvision import sam3_pipeline as sp
    sc = cfg["spatula_observer"]
    stages = [sp.StageConfig(p, float(t), int(pad), single)
              for p, t, pad in zip(sc["prompts"], sc["confidence_thresholds"], sc["paddings"])]
    return sp.segment_text(processor, model, Image.fromarray(rgb), stages, device)


def run_material(material: str, zarr_path: Path, text, tracker, device: str, rows_top: list, rows_side: list) -> None:
    import zarr
    from ssvision import sam3_pipeline as sp
    from ssvision.mass import mass_from_pixels
    from ssvision.surface import estimate_surface

    g = zarr.open_group(str(zarr_path), mode="r")
    cfg = json.loads(dict(g.attrs)["config_json"])
    a, b = table4_power_law(material)
    spo, su = g["events"]["spatula_observer"], g["events"]["surface_observer"]
    it, oi = np.array(spo["iteration"]), np.array(spo["observation_index"])
    su_it = np.array(su["iteration"])
    s = cfg["surface"]
    clicks = [s["material_click"][0], s["negative_click"][0]]
    length, tol = float(cfg["vial"]["outer_clickable_length_mm"]), float(s["ransac_tol_mm"])
    t0 = time.time()
    for trial in sorted(set(int(x) for x in it)):
        j = int(np.where((it == trial) & (oi == AFTER_SCOOP))[0][0])
        rgb = np.array(spo["image_camera44"][j])                       # stored RGB
        stored = np.array(spo["mask_camera44"][j]) > 0
        res = top_view(*text, device, rgb, cfg, single=False)
        ok = res.stages.all_stage_success_flags
        n = [len(p.accepted) if p is not None else 0 for p in res.stages.passes]
        sc = [[round(d.score, 4) for d in p.accepted] if p is not None else [] for p in res.stages.passes]
        mask = res.mask > 0
        n_st, n_re = int(stored.sum()), int(mask.sum())
        row = {"material": material, "trial": trial, "stored_px": n_st, "recomputed_px": n_re,
               "iou": round(iou(mask, stored), 5),
               "d_mass_mg": round(mass_from_pixels(n_re, a, b) - mass_from_pixels(n_st, a, b), 4),
               "stage1_ok": int(ok[0]), "stage2_ok": int(ok[1]), "stage1_n": n[0], "stage2_n": n[1],
               "stage1_thr": float(cfg["spatula_observer"]["confidence_thresholds"][0]),
               "stage2_thr": float(cfg["spatula_observer"]["confidence_thresholds"][1]),
               "stage1_scores": " ".join(map(str, sc[0])), "stage2_scores": " ".join(map(str, sc[1])),
               "stage1_box": " ".join(map(str, res.stages.all_stage_bboxes[0] or ())),
               "single_px": "", "single_iou": "", "single_d_mass_mg": ""}
        if max(n) > 1:                       # the 2026-09-23 behaviour differs only where a stage had several
            single = top_view(*text, device, rgb, cfg, single=True).mask > 0
            row.update(single_px=int(single.sum()), single_iou=round(iou(single, stored), 5),
                       single_d_mass_mg=round(mass_from_pixels(int(single.sum()), a, b)
                                              - mass_from_pixels(n_re, a, b), 4))
        rows_top.append(row)

        i = int(np.where(su_it == trial)[0][0])
        side = np.ascontiguousarray(np.array(su["image"][i])[:, :, ::-1])   # stored BGR -> RGB
        side_stored = np.array(su["mask"][i]) > 0
        pr = sp.segment_points(*tracker, Image.fromarray(side), clicks, [1, 0], device)
        cands = [c > 0 for c in pr.candidates]          # the three candidates of the same pass, for the near ties
        m = pr.mask > 0
        depth_st, tilt_st = stored_surface(su, i)
        try:
            est = estimate_surface(m, np.array(su["vial_clicks_image"][i], float), length, tol)
            dd, dt = est["depth_mm"] - depth_st, est["tilt_deg"] - tilt_st
        except (ValueError, RuntimeError) as exc:
            dd = dt = float("nan")
            print(f"  {LABEL[material]} trial {trial}: surface fit failed on the recomputed mask: {exc}")
        order = np.argsort(pr.iou_scores)[::-1]
        rows_side.append({"material": material, "trial": trial, "stored_px": int(side_stored.sum()),
                          "recomputed_px": int(m.sum()), "iou": round(iou(m, side_stored), 5),
                          "d_depth_mm": round(dd, 4), "d_tilt_deg": round(dt, 4),
                          "chosen": pr.index, "predicted_ious": " ".join(f"{v:.4f}" for v in pr.iou_scores),
                          "gap_top2": round(float(pr.iou_scores[order[0]] - pr.iou_scores[order[1]]), 4),
                          "iou_with_stored": " ".join(f"{iou(c, side_stored):.4f}" for c in cands),
                          "best_match": int(np.argmax([iou(c, side_stored) for c in cands])),
                          "stored_depth_mm": round(depth_st, 4), "stored_tilt_deg": round(tilt_st, 4)})
    print(f"  [{LABEL[material]}: 40 top and 40 side views in {time.time() - t0:.0f} s]", file=sys.stderr)


def q(values, f) -> str:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return f"{f(v):.3f}" if len(v) else "---"


def summarise(rows_top: list, rows_side: list) -> None:
    print("\nTop view after the scoop (spatula read 2), 40 reads per material; mass change with a and b of Table 4")
    for material in MATERIALS:
        r = [x for x in rows_top if x["material"] == material]
        if not r:
            continue
        ious = [x["iou"] for x in r]
        dm = np.abs([x["d_mass_mg"] for x in r])
        worst = r[int(np.argmin(ious))]
        equal = sum(x["stored_px"] == x["recomputed_px"] for x in r)
        print(f"  {LABEL[material]:8s} IoU median {np.median(ious):.4f}, min {min(ious):.4f} (trial {worst['trial']}); "
              f"pixel count equal to the stored one on {equal} of {len(r)}; |change in mass| median "
              f"{np.median(dm):.3f} mg, max {dm.max():.3f} mg (trial {r[int(np.argmax(dm))]['trial']})")
        empty_st = [x["trial"] for x in r if x["stored_px"] == 0]
        empty_re = [x["trial"] for x in r if x["recomputed_px"] == 0]
        print(f"           empty mask: stored on trials {empty_st or 'none'}, recomputed on trials {empty_re or 'none'}")
    print("\nDetections above the threshold (0.4) in each stage, after-scoop reads")
    for material in MATERIALS:
        r = [x for x in rows_top if x["material"] == material]
        if not r:
            continue
        for k in (1, 2):
            n = [x[f"stage{k}_n"] for x in r]
            several = [x["trial"] for x in r if x[f"stage{k}_n"] > 1]
            print(f"  {LABEL[material]:8s} stage {k}: none on {sum(v == 0 for v in n)}, one on {sum(v == 1 for v in n)}, "
                  f"more than one on {len(several)}" + (f" (trials {', '.join(map(str, several))})" if several else ""))
        multi = [x for x in r if x["single_px"] != ""]
        if multi:
            kept = [x["trial"] for x in multi if x["iou"] - x["single_iou"] > EVIDENCE]
            edge = [x["trial"] for x in multi
                    if any(x[f"stage{k}_n"] > 1 and sorted(map(float, x[f"stage{k}_scores"].split()))[-2]
                           < x[f"stage{k}_thr"] + EDGE for k in (1, 2))]
            print(f"           the stored mask matches the union better than the top detection alone (IoU higher by "
                  f"more than {EVIDENCE}) on {len(kept)} of {len(multi)}: there the service kept more than one detection")
            print(f"           second-best score within {EDGE} above the threshold on {len(edge)} of {len(multi)}"
                  + (f" (trials {', '.join(map(str, edge))}): their count can differ on another device" if edge else ""))
        for x in multi:
            print(f"           trial {x['trial']}: stage 1 scores {x['stage1_scores'] or '-'}, stage 2 scores "
                  f"{x['stage2_scores'] or '-'}; union {x['recomputed_px']} px (IoU {x['iou']:.4f} with the stored "
                  f"mask, {x['stored_px']} px), top detection only {x['single_px']} px (IoU {x['single_iou']:.4f}), "
                  f"mass {x['single_d_mass_mg']:+.3f} mg")
    print("\nSide view of the surface estimate, 40 reads per material; best of three candidates by predicted IoU")
    for material in MATERIALS:
        r = [x for x in rows_side if x["material"] == material]
        if not r:
            continue
        ious = [x["iou"] for x in r]
        worst = r[int(np.argmin(ious))]
        ties = [x["trial"] for x in r if x["gap_top2"] < NEAR_TIE]
        other = [x["trial"] for x in r if x["chosen"] != x["best_match"]]
        big = [x["trial"] for x in r if not abs(x["d_depth_mm"]) <= 1.0]
        print(f"  {LABEL[material]:8s} IoU median {np.median(ious):.4f}, min {min(ious):.4f} (trial {worst['trial']}); "
              f"|change in depth| median {q(np.abs([x['d_depth_mm'] for x in r]), np.median)} mm, max "
              f"{q(np.abs([x['d_depth_mm'] for x in r]), np.max)} mm; |change in tilt| median "
              f"{q(np.abs([x['d_tilt_deg'] for x in r]), np.median)} deg, max {q(np.abs([x['d_tilt_deg'] for x in r]), np.max)} deg")
        print("           the stored mask is closest to candidate 0, 1, 2 on "
              + ", ".join(str(sum(x["best_match"] == k for x in r)) for k in range(3)) + " reads; "
              f"near ties (two best predicted IoUs within {NEAR_TIE}): {len(ties)}"
              + (f" (trials {', '.join(map(str, ties))})" if ties else ""))
        print(f"           chosen candidate not the one closest to the stored mask: {len(other)}"
              + (f" (trials {', '.join(map(str, other))})" if other else "")
              + f"; |change in depth| above 1 mm: {len(big)}" + (f" (trials {', '.join(map(str, big))})" if big else ""))
        good = [x for x in r if x["chosen"] == x["best_match"]]
        if other and good:
            gi = [x["iou"] for x in good]
            print(f"           where the chosen candidate is the closest one: IoU median {np.median(gi):.4f}, min "
                  f"{min(gi):.4f}")
        d = load(material)
        for trial in d["iteration"][d["surface_fail"]]:
            x = next(y for y in r if y["trial"] == int(trial))
            print(f"           trial {x['trial']}, a surface failure of the paper: stored {x['stored_depth_mm']:+.2f} mm, "
                  f"{x['stored_tilt_deg']:+.2f} deg; recomputed {x['stored_depth_mm'] + x['d_depth_mm']:+.2f} mm, "
                  f"{x['stored_tilt_deg'] + x['d_tilt_deg']:+.2f} deg (candidate {x['chosen']}; the stored mask is "
                  f"closest to candidate {x['best_match']}; predicted IoUs {x['predicted_ious']})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="auto", help="auto, cpu, mps or cuda")
    ap.add_argument("--sam3-model", default="facebook/sam3", help="Hugging Face id or local folder of the SAM 3 weights")
    ap.add_argument("runs", nargs="*", help="material=PATH of a run log, to read it from elsewhere")
    args = ap.parse_args()
    runs = {m: RUNS[m] for m in MATERIALS}
    for spec in args.runs:
        m, p = spec.split("=", 1)
        runs[m] = Path(p).expanduser()
    missing = [str(p) for p in runs.values() if not Path(p).exists()]
    if missing:
        print(f"run logs not found: {', '.join(missing)} (python scripts/download_data.py runs)", file=sys.stderr)
        return 2

    import torch
    import transformers
    from ssvision.segment import _text_model, _tracker, model_label, pick_device
    device = pick_device(args.device)
    text, tracker = _text_model(args.sam3_model, device), _tracker(args.sam3_model, device)
    print(f"SAM 3 masks recomputed with ssvision.sam3_pipeline (the service's processing, arms_sam3 bc48b4f): "
          f"device {device}, torch {torch.__version__}, transformers {transformers.__version__}, "
          f"{model_label(args.sam3_model)}")
    rows_top, rows_side = [], []
    for m in MATERIALS:
        run_material(m, Path(runs[m]), text, tracker, device, rows_top, rows_side)
    IMAGES.mkdir(parents=True, exist_ok=True)
    for name, rows in (("top", rows_top), ("side", rows_side)):
        path = IMAGES / f"sam3_recompute_{name}.csv"
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print(f"  [wrote {path}]", file=sys.stderr)
    summarise(rows_top, rows_side)
    return 0


if __name__ == "__main__":
    sys.exit(main())
