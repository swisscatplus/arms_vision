"""Segment the material in the 2026-10-02 side-camera captures with the runs' two point prompts.

In the runs the side-view mask came from SAM 3, served on the GPU machine as the ROS 2 service
`/segment_image_clicks` (arms_ws `hardware/sam2_clicks_real.py`; the `sam2d` name is historical).
That service cannot be reached from this Mac, so this script runs the same prompts offline with Hugging Face
`transformers`:

    sam3    facebook/sam3                 Sam3TrackerModel + Sam3TrackerProcessor   (gated: accept Meta's terms)
    sam2.1  facebook/sam2.1-hiera-large   Sam2Model + Sam2Processor                 (public, Apache-2.0)

`--model sam3` (the default) tries SAM 3 once. If Hugging Face refuses the download (HTTP 403, e.g.
"awaiting a review from the repo authors"), it says so and falls back to SAM 2.1, unless
`--no-fallback` is given. No other source of the SAM 3 weights is used.

Prompts: the surface observer's positive and negative click, in pixels of the 1936 x 1216 frame, read
from the runs' own config snapshots (data/<run>_run_meta.json, written by extract_runs.py from each
zarr's config_json): positive (638, 718), negative (1198, 272), identical in the three runs.
Mask choice: three candidate masks (multimask_output=True), the one with the highest predicted IoU. This is
what the service did (arms_sam3 commit bc48b4f, ImageSegmentation.run_point_segmentation;
docs/segmentation_service.md) and what ssvision.sam3_pipeline.segment_points does, so Figure 3's masks, made with
this script on 2026-10-02, follow the service's rule.

Writes, next to each image in paper/data/photos/:
    <stem>.mask_<model>.png        0/255 mono mask, same size as the image
and one provenance file per run: paper/data/photos/segment_side_views_<model>.json (model id and
revision, transformers and torch versions, device, prompts, scores, mask areas, image SHA-256, and the
validation below when it was run).

--validate RUN:TRIAL ... (e.g. salt:24) also segments the stored side frame of each run trial (stored
BGR, converted to RGB) and compares every candidate with the stored SAM 3 mask: pixel IoU, and the surface
line fitted to each mask by the paper's code path (ssvision.surface.surface_line): angle difference, and the offset of p (where the line meets the click axis) along the axis in px
and mm. It marks the chosen candidate and a near tie (predicted IoUs within NEAR_TIE), where the choice
can flip between this Mac and the GPU machine.

Run with a Python that has torch, transformers and this repository's package (pip install -e ".[sam3]"):
    VENV/bin/python paper/analysis/tools/segment_side_views.py [--model sam3|sam2.1] [--device auto|mps|cpu]
                                                    [--validate salt:24 ...] [--no-fallback]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")     # before torch: CPU for ops MPS lacks

import numpy as np
from PIL import Image

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from common import DATA, PHOTOS  # noqa: E402

NEW_IMAGES = ("vialSideView.png", "vial_sideVewWithSpatula.png", "vial_sideView_deposite.png")
MODELS = {
    "sam3": ("facebook/sam3", "Sam3TrackerModel", "Sam3TrackerProcessor"),
    "sam2.1": ("facebook/sam2.1-hiera-large", "Sam2Model", "Sam2Processor"),
}
RUN_METAS = ("salt", "sugar", "polenta")


class AccessRefused(RuntimeError):
    pass


def run_prompts() -> tuple[list[float], list[float], dict]:
    """The surface observer's clicks, from the runs' config snapshots; checked identical across runs."""
    metas = {m: json.loads((DATA / f"{m}_run_meta.json").read_text()) for m in RUN_METAS}
    pos = {m: tuple(metas[m]["prompt_clicks_px"]["positive"][0]) for m in metas}
    neg = {m: tuple(metas[m]["prompt_clicks_px"]["negative"][0]) for m in metas}
    clicks = {m: tuple(map(tuple, metas[m]["vial_clicks_px"])) for m in metas}
    if len(set(pos.values())) != 1 or len(set(neg.values())) != 1 or len(set(clicks.values())) != 1:
        raise ValueError(f"the runs' prompts differ: {pos} {neg} {clicks}")
    p, n = list(map(float, pos["salt"])), list(map(float, neg["salt"]))
    return p, n, {"source": [f"data/{m}_run_meta.json" for m in RUN_METAS],
                  "vial_clicks_px": [list(c) for c in clicks["salt"]]}


def pick_device(want: str) -> str:
    import torch
    if want != "auto":
        return want
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(key: str, device: str):
    """(model, processor, info) for `key`; raises AccessRefused on a gated-repo refusal."""
    import transformers
    repo, model_cls, proc_cls = MODELS[key]
    try:
        processor = getattr(transformers, proc_cls).from_pretrained(repo)
        model = getattr(transformers, model_cls).from_pretrained(repo).to(device).eval()
    except Exception as exc:                                    # noqa: BLE001 - inspect the refusal below
        text = f"{type(exc).__name__}: {exc}"
        if "403" in text or "gated" in text.lower() or "awaiting a review" in text:
            raise AccessRefused(text) from exc
        raise
    return model, processor, {"repo": repo, "revision": cached_revision(repo), "model_class": model_cls,
                              "processor_class": proc_cls}


def cached_revision(repo: str) -> str | None:
    """The commit of the snapshot the weights were loaded from (the cache folder is named by it)."""
    from huggingface_hub import try_to_load_from_cache
    path = try_to_load_from_cache(repo, "config.json")
    return Path(path).parent.name if isinstance(path, str) else None


def candidates(model, processor, device: str, rgb: np.ndarray, pos, neg) -> tuple[np.ndarray, np.ndarray]:
    """The three candidate masks (3, H, W) and their predicted IoUs for one positive and one negative click."""
    import torch
    image = Image.fromarray(np.ascontiguousarray(rgb))
    inputs = processor(images=image, input_points=[[[pos, neg]]], input_labels=[[[1, 0]]],
                       return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, multimask_output=True)
    masks = processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
    scores = out.iou_scores.detach().float().cpu().numpy().reshape(-1)
    return np.asarray(masks[0]).astype(bool), scores


def segment(model, processor, device: str, rgb: np.ndarray, pos, neg) -> tuple[np.ndarray, list[float], int]:
    """Best-of-three mask (highest predicted IoU) for one positive and one negative click."""
    masks, scores = candidates(model, processor, device, rgb, pos, neg)
    k = int(np.argmax(scores))
    return masks[k], [float(s) for s in scores], k


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def line_at_axis(mask: np.ndarray, clicks: np.ndarray) -> dict:
    """The paper's surface line (ssvision.surface.surface_line) and p, where it meets the click axis."""
    from ssvision.surface import surface_line
    c, v, pts, inl = surface_line(mask.astype(np.uint8), clicks)
    a, b = clicks                                    # click 0 = lip ("top of vial"), click 1 = base
    ax = (a - b) / np.linalg.norm(a - b)
    s, _ = np.linalg.solve(np.array([ax, -v]).T, c - b)
    p = b + s * ax
    ang = np.degrees(np.arctan2(v[1], v[0]))
    ang = (ang + 90.0) % 180.0 - 90.0                # undirected line angle in (-90, 90]
    return {"p": p, "s_px": float(s), "angle_deg": float(ang), "n_inliers": int(inl.sum()), "n_pts": len(pts)}


NEAR_TIE = 0.01      # predicted-IoU gap below which the best-of-three choice can flip between machines


def validate(model, processor, device, spec: str, pos, neg, check_dir: Path | None) -> dict:
    """One stored run frame: each candidate against the stored SAM 3 mask (IoU, surface line)."""
    import zarr
    from extract_runs import RUNS
    run, trial = spec.split(":")
    trial = int(trial)
    su = zarr.open_group(str(RUNS[run]), mode="r")["events"]["surface_observer"]
    i = int(np.where(np.array(su["iteration"]) == trial)[0][0])
    rgb = np.array(su["image"][i])[:, :, ::-1]                 # stored BGR -> RGB
    stored = np.array(su["mask"][i]) > 0
    clicks = np.array(su["vial_clicks_image"][i], float)
    masks, scores = candidates(model, processor, device, rgb, pos, neg)
    order = np.argsort(scores)[::-1]
    k = int(order[0])
    gap = float(scores[order[0]] - scores[order[1]])
    a = line_at_axis(stored, clicks)
    mm_per_px = 57.0 / float(np.linalg.norm(clicks[0] - clicks[1]))   # vial.outer_clickable_length_mm
    per = []
    for j in range(len(masks)):
        m = masks[j]
        row = {"candidate": j, "score": float(scores[j]), "iou": float((m & stored).sum() / (m | stored).sum()),
               "area_px": int(m.sum()), "only_new_px": int((m & ~stored).sum()),
               "only_stored_px": int((~m & stored).sum())}
        try:
            b = line_at_axis(m, clicks)
            row.update(line_angle_deg=b["angle_deg"],
                       line_angle_diff_deg=(b["angle_deg"] - a["angle_deg"] + 90.0) % 180.0 - 90.0,
                       p_offset_along_axis_px=b["s_px"] - a["s_px"],
                       p_offset_along_axis_mm=(b["s_px"] - a["s_px"]) * mm_per_px)
        except Exception as exc:                                       # noqa: BLE001 - report, do not stop
            row.update(line_error=str(exc))
        per.append(row)
    best_match = int(np.argmax([r["iou"] for r in per]))
    out = {"run": run, "trial": trial, "stored_area_px": int(stored.sum()),
           "stored_line_angle_deg": a["angle_deg"], "stored_p_px": a["p"].tolist(), "mm_per_px": mm_per_px,
           "chosen": k, "score_gap_top2": gap, "near_tie": gap < NEAR_TIE, "best_matching_candidate": best_match,
           "candidates": per}
    print(f"validation on {run} trial {trial} (stored SAM 3 mask, {int(stored.sum())} px; stored line "
          f"{a['angle_deg']:+.3f} deg): chosen candidate #{k}, gap to the next {gap:.4f}"
          f"{' (near tie)' if gap < NEAR_TIE else ''}; best match to the stored mask #{best_match}")
    for r in per:
        line = (f"line {r['line_angle_deg']:+.3f} deg ({r['line_angle_diff_deg']:+.3f}), p along the axis "
                f"{r['p_offset_along_axis_px']:+.2f} px = {r['p_offset_along_axis_mm']:+.3f} mm"
                if "line_angle_deg" in r else f"line fit failed: {r['line_error']}")
        print(f"  #{r['candidate']}{'*' if r['candidate'] == k else ' '} score {r['score']:.4f}: IoU {r['iou']:.4f}, "
              f"{r['area_px']} px ({r['only_new_px']} only new, {r['only_stored_px']} only stored); {line}")
    if check_dir is not None:
        check_dir.mkdir(parents=True, exist_ok=True)
        for j, m in enumerate(masks):
            Image.fromarray((m * 255).astype(np.uint8)).save(check_dir / f"validate_{run}_t{trial}_c{j}.png")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", choices=sorted(MODELS), default="sam3")
    ap.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    ap.add_argument("--no-fallback", action="store_true", help="do not fall back to SAM 2.1 if SAM 3 is refused")
    ap.add_argument("--validate", nargs="*", default=[], help="RUN:TRIAL ..., e.g. salt:24 polenta:32")
    ap.add_argument("--check-dir", type=Path, default=None, help="also write the validation mask here")
    ap.add_argument("--images", nargs="*", default=list(NEW_IMAGES), help="file names in paper/data/photos")
    args = ap.parse_args()

    import torch
    import transformers
    pos, neg, prompt_src = run_prompts()
    device = pick_device(args.device)
    key = args.model
    refused = None
    try:
        model, processor, info = load_model(key, device)
    except AccessRefused as exc:
        refused = str(exc).splitlines()
        print(f"[segment] {MODELS[key][0]} refused by Hugging Face: {' '.join(refused)[:400]}")
        if key != "sam3" or args.no_fallback:
            return 2
        key = "sam2.1"
        print("[segment] falling back to SAM 2.1 (facebook/sam2.1-hiera-large, public)")
        model, processor, info = load_model(key, device)
    print(f"[segment] model {info['repo']} @ {info['revision']} ({info['model_class']}), transformers "
          f"{transformers.__version__}, torch {torch.__version__}, device {device}; prompts + {pos} - {neg}")

    log = {"model_key": key, **info, "transformers": transformers.__version__, "torch": torch.__version__,
           "device": device, "requested_model": args.model, "sam3_refused": refused,
           "prompts": {"positive_xy": pos, "negative_xy": neg, "labels": [1, 0], **prompt_src},
           "mask_choice": "multimask_output=True, highest predicted IoU, as the service (arms_sam3 bc48b4f, "
                          "ImageSegmentation.run_point_segmentation)",
           "date": time.strftime("%Y-%m-%d %H:%M:%S"), "images": {}}
    for name in args.images:
        path = PHOTOS / name
        rgb = np.asarray(Image.open(path).convert("RGB"))       # PNGs are RGB (checked against a run frame)
        if rgb.shape[:2] != (1216, 1936):
            raise ValueError(f"{name}: expected a 1936 x 1216 side-camera frame, got {rgb.shape[1]} x {rgb.shape[0]}")
        t0 = time.time()
        mask, scores, k = segment(model, processor, device, rgb, pos, neg)
        dt = time.time() - t0
        out = path.with_name(f"{path.stem}.mask_{key}.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(out)
        gap = float(np.sort(scores)[-1] - np.sort(scores)[-2])
        log["images"][name] = {"mask": out.name, "area_px": int(mask.sum()), "scores": scores, "chosen": k,
                               "score_gap_top2": gap, "image_sha256": sha256(path), "seconds": round(dt, 2)}
        print(f"[segment] {name}: mask {out.name}, {int(mask.sum())} px, scores {np.round(scores, 4).tolist()} "
              f"-> #{k} (gap {gap:.4f}{', near tie' if gap < NEAR_TIE else ''}), {dt:.1f} s")
    log["validation"] = [validate(model, processor, device, spec, pos, neg, args.check_dir)
                         for spec in args.validate]
    (PHOTOS / f"segment_side_views_{key}.json").write_text(json.dumps(log, indent=1))
    print(f"[segment] wrote {PHOTOS / f'segment_side_views_{key}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
