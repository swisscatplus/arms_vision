"""Optional: the two masks from SAM 3, with the paper's prompts and thresholds, processed as the service of the runs.

Needs torch and transformers (pip install -e ".[sam3]") and access to the gated facebook/sam3 weights on Hugging
Face (request access on the model page, then `hf auth login`). SAM 3 is distributed by Meta under its own licence.

    side view (surface):  one positive and one negative point prompt at fixed pixels   -> side_view_mask
    top view (mass):      text prompt "scoop" for the spatula, then the material's prompt on the crop to the
                          spatula's mask, both at confidence 0.4                        -> top_view_mask

In the paper's runs the masks came from SAM 3 on a GPU machine, through a service in the private workspace
swisscatplus/arms_gpu_ws: a ROS 2 node and its backend, arms_sam3, by Jasper Tan. ssvision.sam3_pipeline ports
the backend's processing from arms_sam3 commit bc48b4f, and these functions call it (docs/segmentation_service.md):
  - top view: each stage keeps every detection above 0.4 and takes the union of their masks. The second stage runs
    on the crop to the box of the first stage's union mask (padding 0), and its union mask is pasted back into the
    full frame. The robot asked for one detection per stage (single_detection_only), but the service ignored that
    flag until 2026-09-23; single_detection_only=True gives the behaviour since then.
  - side view: the best of SAM 3's three candidate masks by predicted IoU, binarised at probability 0.5.
  - FP32 on CPU, MPS or CUDA, without autocast (TF32 allowed on CUDA, as on the service).
On the same frames, device and precision, the port and arms_sam3's own functions give identical masks. Masks
recomputed here still differ slightly from the stored ones, because the service ran on a CUDA GPU in TF32, with
library versions and a weights revision that were not recorded: on the three demo trials, IoU 0.986-0.997 (top view)
and 0.966-0.995 (side view).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from . import sam3_pipeline as pipeline

# The prompts and thresholds of the paper's runs (Table 1, and the configuration stored in each run log)
SURFACE_POSITIVE_PX = (638, 718)      # side view: a point where material always was (lower part of the vial) ...
SURFACE_NEGATIVE_PX = (1198, 272)     # ... and a point where it never was; pixels of the 1936 x 1216 frame
SPATULA_PROMPT = "scoop"              # top view, first stage: the spatula
MATERIAL_PROMPTS = {"salt": "white material on the scoop",      # top view, second stage, per material (Table 1)
                    "sugar": "white material on the scoop",
                    "polenta": "yellow material on the scoop"}
CONFIDENCE = (0.4, 0.4)               # detection threshold of the two stages
PADDING_PX = (0, 0)                   # the second stage runs on the crop to the first stage's box, without padding
MODEL_ID = pipeline.MODEL_ID


def pick_device(want: str = "auto") -> str:
    if want != "auto":
        return want
    import torch
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def model_label(model_id: str) -> str:
    """The model id as printed: a local folder of weights is shown by its name only (a cache snapshot hash)."""
    p = Path(model_id).expanduser()
    return f"local weights {p.name}" if p.is_dir() else model_id


@lru_cache(maxsize=None)
def _tracker(model_id: str, device: str):
    return pipeline.load_tracker_model(model_id, device)


@lru_cache(maxsize=None)
def _text_model(model_id: str, device: str):
    return pipeline.load_text_model(model_id, device)


def side_view_mask(rgb: np.ndarray, positive=SURFACE_POSITIVE_PX, negative=SURFACE_NEGATIVE_PX,
                   model_id: str = MODEL_ID, device: str = "auto") -> tuple[np.ndarray, dict]:
    """The material in the vial, from one positive and one negative point prompt (the tracker head of SAM 3).

    `rgb` is the full side-view frame (H, W, 3, RGB) and the points are in its pixels. As on the service
    (sam3_pipeline.segment_points), SAM 3 returns three candidate masks and the one with the highest predicted IoU
    is kept; the runs set no score threshold. -> (mask, info with the predicted IoUs of the three candidates, the
    chosen one and a printable note)."""
    device = pick_device(device)
    processor, model = _tracker(model_id, device)
    res = pipeline.segment_points(processor, model, Image.fromarray(np.ascontiguousarray(rgb)), [positive, negative],
                                  [1, 0], device)
    ious = ", ".join(f"{v:.3f}{'*' if k == res.index else ''}" for k, v in enumerate(res.iou_scores))
    note = f"SAM 3 ({model_label(model_id)}, best of 3 candidate masks by predicted IoU: {ious})"
    return res.mask.astype(bool), {"predicted_iou": res.predicted_iou, "candidate": res.index,
                                   "candidate_ious": res.iou_scores, "presence": res.presence,
                                   "message": res.message, "note": note}


def top_view_mask(rgb: np.ndarray, prompts, thresholds=CONFIDENCE, paddings=PADDING_PX, model_id: str = MODEL_ID,
                  device: str = "auto", single_detection_only: bool = False) -> tuple[np.ndarray, dict]:
    """The material on the spatula, from the two-stage text prompt (the concept head of SAM 3): prompts[0] finds the
    spatula, prompts[1] the material on the crop to the spatula's box.

    `rgb` is the full top-view frame (H, W, 3, RGB); the mask is returned in its pixels. As on the service
    (sam3_pipeline.segment_text): each stage keeps every detection above its threshold and takes the union of their
    masks, binarised at 0.5; the next stage runs on the crop to the box of that union mask plus the padding.
    single_detection_only=True keeps only the top detection of each stage instead, as the service has since
    2026-09-23. As on the robot, an empty first stage is a failed reading (the spatula was not found), not 0 mg:
    RuntimeError. An empty second stage gives an empty mask (0 mg).
    -> (mask, info with the scores and number of detections above each threshold, the box of the first stage, the
    service's message and a printable note)."""
    device = pick_device(device)
    processor, model = _text_model(model_id, device)
    pads = list(paddings) if paddings else [0] * len(prompts)
    stages = [pipeline.StageConfig(p, float(t), int(pad), single_detection_only)
              for p, t, pad in zip(prompts, thresholds, pads)]
    res = pipeline.segment_text(processor, model, Image.fromarray(np.ascontiguousarray(rgb)), stages, device)
    passes = res.stages.passes if res.stages is not None else [res.single]
    if res.stages is not None:
        failed = [i for i, ok in enumerate(res.stages.all_stage_success_flags) if not ok]
        failed_at = failed[0] if failed else None
    else:
        failed_at = None if res.single.is_positive else 0
    if failed_at == 0:
        raise RuntimeError(f"stage 1 ({prompts[0]!r}) found nothing above {thresholds[0]}: a failed reading, "
                           f"not 0 mg ({res.message})")

    def scores(i):
        p = passes[i] if i < len(passes) else None
        return sorted((d.score for d in p.accepted), reverse=True) if p is not None else []

    s = [scores(i) for i in range(len(stages))]
    box = list(res.stages.all_stage_bboxes[0]) if res.stages is not None else None
    kept = "top detection only" if single_detection_only else "union"

    def describe(i):
        if not s[i]:
            return f"stage {i + 1} {prompts[i]!r} no detection above {thresholds[i]}: 0 mg"
        return (f"stage {i + 1} {prompts[i]!r} {len(s[i])} above {thresholds[i]} ("
                + ", ".join(f"{v:.3f}" for v in s[i][:4]) + (", ..." if len(s[i]) > 4 else "") + ")")

    note = (f"SAM 3 ({model_label(model_id)}, {kept}): " + describe(0)
            + (f", box of the mask {box}" if box else "") + "".join(f"; {describe(i)}" for i in range(1, len(stages))))
    info = {"stage1_score": s[0][0] if s[0] else None, "stage1_n": len(s[0]), "box": box,
            "stage2_score": s[1][0] if len(s) > 1 and s[1] else None, "stage2_n": len(s[1]) if len(s) > 1 else 0,
            "stage_scores": s, "message": res.message, "note": note}
    return res.mask.astype(bool), info


def screen_prompt(frames_with, frames_without, material_prompt: str, spatula_prompt: str = SPATULA_PROMPT,
                  model_id: str = MODEL_ID, device: str = "auto") -> dict:
    """The paper's criterion for choosing a material prompt (Methods, Vision-Based Mass Estimation): the confidence
    of `material_prompt` in the second stage on top-view frames with material on the spatula, and on frames with an
    empty spatula. The margin is the lowest confidence with material minus the highest without. A phrasing qualifies
    when the margin is positive (one threshold then separates loaded from empty spatulas), and the paper kept the
    qualifying phrasing with the largest margin. Each frame must show the spatula (the first stage must find it).
    The phrasings of Table 1 were screened before this repository existed, by a coding agent that applied this
    criterion; this function is a re-implementation of the criterion, not the code that produced Table 1."""
    def score(rgb) -> float:
        _, info = top_view_mask(rgb, (spatula_prompt, material_prompt), thresholds=(CONFIDENCE[0], 0.0),
                                model_id=model_id, device=device)
        return info["stage2_score"] if info["stage2_score"] is not None else 0.0
    with_material = [score(f) for f in frames_with]
    without = [score(f) for f in frames_without]
    return {"prompt": material_prompt, "lowest_with_material": min(with_material),
            "highest_without": max(without), "margin": min(with_material) - max(without),
            "with_material": with_material, "without": without}
