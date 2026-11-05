# Derived from arms_sam3 (Jasper Tan, Swiss Cat+; see docs/segmentation_service.md). segment_text's dispatch and
# png_from_bgr8 are derived from sam3_ros2 (Jasper Tan), licensed under the Apache License 2.0
# (LICENSE-sam3_ros2.txt in this folder), and were changed: rewritten as functions on in-memory images, without ROS
# messages or HTTP.
"""The SAM 3 processing of the service that produced the paper's masks, without ROS, HTTP server or GPU machine.

In the runs (2026-09-10 and 11) the robot sent every image to a SAM 3 service on a GPU machine: a ROS 2 node,
sam3_ros2, forwarded each request over HTTP to a Flask backend, arms_sam3. Both are by Jasper Tan, in the private
workspace github.com/swisscatplus/arms_gpu_ws. This module reproduces the backend's processing, function by function:

    text request, two stages or more   segment_text -> orchestrate_multistage   stage 1 finds the spatula; stage 2 runs
                                                                                on the crop to the box of stage 1's mask
    text request, one stage            segment_text -> segment_one_stage         the request without stages
    click request                      segment_points                            best of three masks by predicted IoU
    image decoding                     decode_request_image                      PNG -> RGB
    precision                          load_text_model, load_tracker_model       FP32, TF32 allowed on CUDA, no autocast

Source: arms_sam3 at commit bc48b4f (2026-09-15), scripts/image_segmentation.py, multistage_orchestrator.py,
segmentation_filtering.py and mask_geometry.py. The run logs show that the backend that served the runs already
wrote that commit's per-stage messages and debug dumps, and none of the later fixes. Before 2026-09-23 the flag
single_detection_only had no effect, so every stage kept the union of its accepted detections; the option
single_detection_only=True reproduces the fix of commit 1238357 (2026-09-23), which keeps the top detection only.
docs/segmentation_service.md gives the evidence and what each step did. Each function below names the function it
reproduces. The dispatch of one-stage requests (segment_text) and the PNG encoding of the robot's frames
(png_from_bgr8) follow sam3_ros2, src/sam3_ros2/sam3_ros2/backend_client.py and image_converter.py, Apache License
2.0 (LICENSE-sam3_ros2.txt in this folder).

Not reproduced: the ROS node, the Flask server, the browser GUI, the YAML crop presets and the debug dumps.

Needs torch and transformers (pip install -e ".[sam3]") and the gated facebook/sam3 weights. The functions take
loaded models, so the caller chooses the weights and the device (ssvision.segment loads them once per run).
"""
from __future__ import annotations

import base64
import io
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from PIL import Image

BBox = tuple[int, int, int, int]          # (x1, y1, x2, y2), x2 and y2 exclusive, as PIL's crop
MODEL_ID = "facebook/sam3"                # the backend loaded this id without a revision (ImageSegmentation.__init__)


# --- image decoding and precision -------------------------------------------------------------------------------

def png_from_bgr8(bgr: np.ndarray) -> bytes:
    """The PNG the ROS node sent to the backend for a bgr8 frame (sam3_ros2 image_converter.ros_image_to_bytes,
    Apache-2.0). OpenCV writes the BGR array as a PNG with true colours, so a frame the robot sent as
    BGR reaches the model as RGB (decode_request_image). PNG is lossless: decoding gives back the frame exactly."""
    import cv2
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(bgr))
    if not ok:
        raise ValueError("failed to encode the image to PNG")
    return buf.tobytes()


def decode_request_image(data: bytes | str) -> Image.Image:
    """The request image as the backend decoded it (arms_sam3 bc48b4f, image_segmentation.py,
    ImageSegmentation.handle_request and handle_points_request): base64 or raw PNG bytes -> PIL image in RGB."""
    raw = base64.b64decode(data) if isinstance(data, str) else data
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _allow_tf32(device: str) -> None:
    """ImageSegmentation.__init__ (arms_sam3 bc48b4f) set torch.backends.cuda.matmul.allow_tf32 and
    torch.backends.cudnn.allow_tf32 to True. They act on CUDA only, so they are set only there."""
    if str(device).startswith("cuda"):
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def load_text_model(model_id: str = MODEL_ID, device: str = "cpu", revision: Optional[str] = None):
    """(Sam3Processor, Sam3Model) in FP32 on `device`, as ImageSegmentation.__init__ (arms_sam3 bc48b4f) loaded them.

    The constructor also entered torch.autocast(bfloat16), but only in its own thread: autocast state is per
    thread, and Flask served each request on a worker thread, so every served request ran in FP32 (TF32 on CUDA).
    arms_gpu_ws design_docs/GUI_DESIGN.md section 3.6 records this. No autocast is entered here."""
    from transformers import Sam3Model, Sam3Processor
    _allow_tf32(device)
    model = Sam3Model.from_pretrained(model_id, revision=revision).to(device).eval()
    return Sam3Processor.from_pretrained(model_id, revision=revision), model


def load_tracker_model(model_id: str = MODEL_ID, device: str = "cpu", revision: Optional[str] = None):
    """(Sam3TrackerProcessor, Sam3TrackerModel) in FP32 on `device`, the click head, as
    ImageSegmentation._load_tracker (arms_sam3 bc48b4f) loaded it from the same checkpoint."""
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    _allow_tf32(device)
    model = Sam3TrackerModel.from_pretrained(model_id, revision=revision).to(device).eval()
    return Sam3TrackerProcessor.from_pretrained(model_id, revision=revision), model


# --- one text pass: filtering and union (segmentation_filtering.py) ------------------------------------------------

def extract_confidence(label: str) -> Optional[float]:
    """The score written into a label like "scoop (0.87)" (segmentation_filtering.extract_confidence). The backend
    thresholds and ranks detections by this two-decimal value, not by the model's full-precision score."""
    match = re.search(r"\((\d*\.?\d+)\)$", label)
    return float(match.group(1)) if match else None


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    """A strict uint8 {0, 1} mask (segmentation_filtering.binarize_mask)."""
    arr = np.asarray(mask)
    if arr.dtype == bool:
        return arr.astype(np.uint8)
    if np.issubdtype(arr.dtype, np.floating):
        return (arr > 0.5).astype(np.uint8)
    if np.issubdtype(arr.dtype, np.integer) and arr.max(initial=0) > 1:
        return (arr >= 128).astype(np.uint8)
    return (arr > 0).astype(np.uint8)


@dataclass(frozen=True)
class Detection:
    """One instance of a text pass: its binary mask in the pass's image, its label "<prompt> (<score:.2f>)" as the
    backend built it, and the model's full-precision score (kept here for reporting only)."""

    mask: np.ndarray
    label: str
    score: float


@dataclass(frozen=True)
class SinglePass:
    """One text pass over one image (segmentation_filtering.SinglePassSegmentationResult)."""

    image_size: tuple[int, int]                       # (height, width)
    prompt: str
    confidence_threshold: float
    accepted: list[Detection]                         # after the threshold and the empty-mask check
    union_mask: np.ndarray                            # uint8 {0, 1}, the union of every accepted mask
    confidence_interval: Optional[tuple[float, float]]  # (lowest, highest) two-decimal score of the accepted ones
    is_positive: bool
    rejected_confidences: list[float] = field(default_factory=list)


def build_single_pass(image_size: tuple[int, int], prompt: str, confidence_threshold: float,
                      detections: Sequence[Detection]) -> SinglePass:
    """Threshold, binarise and unite the detections of one pass (segmentation_filtering.build_single_pass_result,
    with filter_annotations and _union_masks). A detection is dropped when its two-decimal score is below the
    threshold or its mask is empty; every other one enters the union."""
    accepted: list[Detection] = []
    confidences: list[float] = []
    for det in detections:
        conf = extract_confidence(det.label)
        if conf is not None and conf < confidence_threshold:
            continue
        binary = binarize_mask(det.mask)
        if not np.any(binary):
            continue
        accepted.append(Detection(binary, det.label, det.score))
        if conf is not None:
            confidences.append(conf)
    height, width = image_size
    if accepted:
        shape = accepted[0].mask.shape
        for i, det in enumerate(accepted[1:], start=1):
            if det.mask.shape != shape:
                raise ValueError(f"Mask shape mismatch: mask 0 has shape {shape}, mask {i} has shape {det.mask.shape}")
        union = np.zeros(shape, dtype=np.uint8)
        for det in accepted:
            union = np.logical_or(union, det.mask).astype(np.uint8)
    else:
        union = np.zeros((height, width), dtype=np.uint8)
    rejected = [c for det in detections if (c := extract_confidence(det.label)) is not None
                and c < confidence_threshold]
    return SinglePass(image_size=image_size, prompt=prompt, confidence_threshold=confidence_threshold,
                      accepted=accepted, union_mask=union,
                      confidence_interval=(min(confidences), max(confidences)) if confidences else None,
                      is_positive=bool(np.any(union)), rejected_confidences=rejected)


def run_text_pass(processor, model, image: Image.Image, prompt: str, threshold: float, device: str) -> SinglePass:
    """One forward pass of the concept head on `image` with `prompt` (ImageSegmentation.run_image_segmentation,
    then build_single_pass_result as in handle_request's run_single_pass).

    The post-processor keeps a query when its score, sigmoid(logit) times sigmoid(presence), is above `threshold`,
    resizes its mask to the image bilinearly and binarises it at 0.5. The backend then sorted the detections by
    score and wrote each score into the label with two decimals."""
    import torch
    pil_image = image.convert("RGB")
    inputs = processor(images=pil_image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    res = processor.post_process_instance_segmentation(
        outputs, threshold=threshold, mask_threshold=0.5, target_sizes=inputs.get("original_sizes").tolist())[0]
    masks = res["masks"].cpu().numpy()
    scores = res["scores"].float().cpu().numpy()
    detections = [Detection(masks[i], f"{prompt} ({scores[i]:.2f})", float(scores[i])) for i in np.argsort(-scores)]
    return build_single_pass((pil_image.height, pil_image.width), prompt, threshold, detections)


# --- boxes, crops and the remapping into the full frame (mask_geometry.py, multistage_orchestrator.py) -------------

def mask_bbox(mask: np.ndarray, padding: int = 0) -> Optional[BBox]:
    """The box of the mask's foreground plus `padding`, clamped to the mask (mask_geometry.derive_mask_bounding_boxes,
    its clamped_bbox); x2 and y2 exclusive. None for an empty mask."""
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {mask.shape}")
    if padding < 0:
        raise ValueError("padding must be >= 0")
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    x1, y1, x2, y2 = int(xs.min()) - padding, int(ys.min()) - padding, int(xs.max()) + 1 + padding, int(ys.max()) + 1 + padding
    h, w = mask.shape
    return max(0, min(x1, w)), max(0, min(y1, h)), max(0, min(x2, w)), max(0, min(y2, h))


def _clamp_crop(bbox: BBox, width: int, height: int) -> BBox:
    """multistage_orchestrator._clamp_bbox_to_image_bounds: clamp, and refuse an empty box."""
    x1, y1, x2, y2 = bbox
    c = (max(0, min(x1, width)), max(0, min(y1, height)), max(0, min(x2, width)), max(0, min(y2, height)))
    if c[2] <= c[0] or c[3] <= c[1]:
        raise ValueError(f"Invalid bbox after clamping: {c}")
    return c


def _remap_mask_into(mask: np.ndarray, output_shape: tuple[int, int], x_offset: int, y_offset: int) -> np.ndarray:
    """Paste a crop's mask into a zero frame at (x_offset, y_offset) (multistage_orchestrator._remap_mask_into)."""
    out_h, out_w = output_shape
    out = np.zeros((out_h, out_w), dtype=np.uint8)
    if mask.size == 0 or y_offset >= out_h or x_offset >= out_w:
        return out
    mask_h, mask_w = mask.shape
    y_end, x_end = min(out_h, y_offset + mask_h), min(out_w, x_offset + mask_w)
    src_h, src_w = max(0, y_end - y_offset), max(0, x_end - x_offset)
    if src_h and src_w:
        out[y_offset:y_end, x_offset:x_end] = mask[:src_h, :src_w]
    return out


# --- the stages (multistage_orchestrator.py) -----------------------------------------------------------------------

@dataclass(frozen=True)
class StageConfig:
    """One stage of a text request (multistage_orchestrator.StageConfig)."""

    prompt: str
    confidence_threshold: float
    padding: int = 0                       # pixels added around the box of this stage's mask, for the next crop
    single_detection_only: bool = False    # False: as the service ran (the union); True: the 2026-09-23 fix


@dataclass(frozen=True)
class StageDecision:
    """multistage_orchestrator.StageDecision (with the stage_mask of commit 1238357)."""

    is_positive: bool
    reason: str
    stage_bbox: Optional[BBox]             # in the coordinates of the stage's input image
    stage_mask: Optional[np.ndarray] = field(default=None, compare=False, repr=False)


def _confidence_rank(det: Detection) -> float:
    conf = extract_confidence(det.label)
    return conf if conf is not None else float("-inf")


def evaluate_stage(stage: SinglePass, padding: int, single_detection_only: bool = False) -> StageDecision:
    """Whether a stage succeeded, the mask it keeps and the box of that mask (multistage_orchestrator.
    _evaluate_stage_decision).

    single_detection_only=False reproduces the backend that served the runs (arms_sam3 e5c0bdd to bc48b4f) whatever
    flag the request carried: the flag only narrowed the list for the emptiness check, and the box and the stage mask
    came from the union of every accepted detection (GUI_DESIGN.md section 3.5). single_detection_only=True
    reproduces commit 1238357 (2026-09-23): the detection with the highest two-decimal score is kept (ties to the
    higher full-precision score, the order the backend sorted them in), and the box and the mask come from it alone.
    With one accepted detection the two are the same."""
    if padding < 0:
        raise ValueError("padding must be >= 0")
    annotations = stage.accepted
    stage_mask = stage.union_mask
    if single_detection_only and len(annotations) > 1:
        top = max(annotations, key=_confidence_rank)
        annotations = [top]
        stage_mask = (np.asarray(top.mask) != 0).astype(np.uint8)
    if not annotations:
        return StageDecision(False, "no_detections", None, stage_mask)
    if not np.any(stage_mask):
        return StageDecision(False, "empty_union_mask", None, stage_mask)
    bbox = mask_bbox(stage_mask, padding=padding)
    if bbox is None:
        return StageDecision(False, "invalid_bbox", None, stage_mask)
    return StageDecision(True, "positive", bbox, stage_mask)


@dataclass(frozen=True)
class MultiStageResult:
    """multistage_orchestrator.MultiStageOrchestrationResult, plus the passes themselves (`passes`)."""

    final_mask: np.ndarray                                   # (H, W) uint8 {0, 1}, full frame
    all_stage_masks: list[np.ndarray]                        # each stage's kept mask, full frame
    all_stage_bboxes: list[Optional[BBox]]                   # each stage's box (padded, clamped), full frame
    all_stage_confidence_intervals: list[Optional[tuple[float, float]]]
    all_stage_success_flags: list[bool]
    all_stage_skip_reasons: list[str]
    stage_crop_bboxes: list[Optional[BBox]]                  # the crop each stage ran on, full frame
    message: str
    all_stage_rejected_confidences: list[list[float]] = field(default_factory=list)
    passes: list[Optional[SinglePass]] = field(default_factory=list, compare=False, repr=False)


RunSinglePassFn = Callable[[Image.Image, str, float], SinglePass]


def orchestrate_multistage(image: Image.Image, run_single_pass: RunSinglePassFn, stages: Sequence[StageConfig],
                           initial_static_crop: Optional[BBox] = None) -> MultiStageResult:
    """Run the stages in order, each on the crop to the previous stage's box, and remap every stage's mask into the
    full frame (multistage_orchestrator.orchestrate_multistage).

    The first stage that fails stops the pipeline: the final mask is then all zero and the message names the stage,
    "pipeline stopped at stage <i>: <reason>". The robot read that message to tell a spatula that was not found
    (stage 0, a failed reading) from a clean spatula (stage 1, 0 mg)."""
    if not stages:
        raise ValueError("stages list must be non-empty")
    full_h, full_w = image.height, image.width
    n = len(stages)
    masks: list[Optional[np.ndarray]] = [None] * n
    bboxes: list[Optional[BBox]] = [None] * n
    intervals: list[Optional[tuple[float, float]]] = [None] * n
    flags = [False] * n
    reasons = ["not_reached"] * n
    crops: list[Optional[BBox]] = [None] * n
    rejected: list[list[float]] = [[] for _ in range(n)]
    passes: list[Optional[SinglePass]] = [None] * n
    offsets: list[BBox] = []
    current, x_off, y_off = image, 0, 0
    if initial_static_crop is not None:
        crop = _clamp_crop(initial_static_crop, full_w, full_h)
        x_off, y_off = crop[0], crop[1]
        current = image.crop(crop)
    failed_at: Optional[int] = None
    for i, cfg in enumerate(stages):
        if i == 0 and initial_static_crop is not None:
            crops[i] = _clamp_crop(initial_static_crop, full_w, full_h)
        elif i > 0:
            crops[i] = offsets[-1] if offsets else None
        result = run_single_pass(current, cfg.prompt, cfg.confidence_threshold)
        passes[i] = result
        intervals[i] = result.confidence_interval
        rejected[i] = list(result.rejected_confidences or [])
        decision = evaluate_stage(result, cfg.padding, cfg.single_detection_only)
        local = decision.stage_mask.astype(np.uint8)
        if offsets:                        # _propagate_mask_through_offsets: the deepest crop's top-left corner
            masks[i] = _remap_mask_into(local, (full_h, full_w), offsets[-1][0], offsets[-1][1])
        else:
            masks[i] = _remap_mask_into(local, (full_h, full_w), x_off, y_off)
        flags[i], reasons[i] = decision.is_positive, decision.reason
        if decision.stage_bbox is not None:
            bx1, by1, bx2, by2 = decision.stage_bbox
            bboxes[i] = (x_off + bx1, y_off + by1, x_off + bx2, y_off + by2)
        if not decision.is_positive:
            failed_at = i
            for j in range(i + 1, n):
                masks[j] = np.zeros((full_h, full_w), dtype=np.uint8)
                reasons[j] = f"pipeline stopped at stage {i}"
            break
        if i < n - 1:
            bx1, by1, bx2, by2 = decision.stage_bbox
            nxt = current.crop((bx1, by1, bx2, by2))
            offsets.append((x_off + bx1, y_off + by1, x_off + bx2, y_off + by2))
            current, x_off, y_off = nxt, x_off + bx1, y_off + by1
    if failed_at is None:
        final, message = masks[-1], f"{n}-stage segmentation successful"
    else:
        final = np.zeros((full_h, full_w), dtype=np.uint8)
        message = f"pipeline stopped at stage {failed_at}: {reasons[failed_at]}"
    return MultiStageResult(
        final_mask=final,
        all_stage_masks=[m if m is not None else np.zeros((full_h, full_w), dtype=np.uint8) for m in masks],
        all_stage_bboxes=bboxes, all_stage_confidence_intervals=intervals, all_stage_success_flags=flags,
        all_stage_skip_reasons=reasons, stage_crop_bboxes=crops, message=message,
        all_stage_rejected_confidences=rejected, passes=passes)


def _fmt_conf(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def stage_summary(stages: Sequence[StageConfig], result: MultiStageResult) -> str:
    """The per-stage summary the backend appended to its message (image_segmentation.build_stage_diagnostics, arms_sam3
    bc48b4f), e.g. "s0 'scoop' thr=0.40 ok acc=[0.870,0.870] | s1 ...". The robot stored the whole message in the run
    log for every failed reading. At bc48b4f the post-processor had already dropped every score at or below the
    threshold, so a failed stage always says "(model returned no scored detections)"."""
    parts = []
    for i, cfg in enumerate(stages):
        rej = result.all_stage_rejected_confidences[i] if i < len(result.all_stage_rejected_confidences) else []
        interval, ok = result.all_stage_confidence_intervals[i], result.all_stage_success_flags[i]
        detail = f"s{i} {cfg.prompt!r} thr={cfg.confidence_threshold:.2f}"
        if ok:
            detail += f" ok acc=[{_fmt_conf(interval[0])},{_fmt_conf(interval[1])}]" if interval else " ok"
        else:
            detail += f" {result.all_stage_skip_reasons[i]}"
            detail += (f" best_rejected={_fmt_conf(max(rej))} ({len(rej)} below thr)" if rej
                       else " (model returned no scored detections)")
            if result.stage_crop_bboxes[i] is not None:
                detail += f" crop={tuple(result.stage_crop_bboxes[i])}"
        parts.append(detail)
    return " | ".join(parts)


# --- the requests (image_segmentation.py) --------------------------------------------------------------------------

@dataclass(frozen=True)
class TextResult:
    """What the backend returned for a text request: the mask (full frame for stages), its message, and the details."""

    mask: np.ndarray                      # uint8 {0, 1}
    message: str
    stages: Optional[MultiStageResult]    # None for a one-stage request
    single: Optional[SinglePass]          # the pass of a one-stage request


def segment_one_stage(processor, model, image: Image.Image, prompt: str, threshold: float, device: str,
                      crop: Optional[BBox] = None) -> TextResult:
    """A text request without stages (ImageSegmentation.handle_request, the legacy path): one pass, the union of
    every detection above the threshold. With `crop` (a crop preset of the server) the image is cropped first and the
    mask stays in the crop's coordinates, as the backend returned it."""
    if crop is not None:
        image = image.crop(crop)
    single = run_text_pass(processor, model, image, prompt, threshold, device)
    if not single.is_positive:
        message = f"No detections above threshold ({threshold}) for prompt '{prompt}'"
        rej = single.rejected_confidences
        message += (f" | best_rejected={_fmt_conf(max(rej))} ({len(rej)} below thr)" if rej
                    else " | model returned no scored detections")
    else:
        message = "Segmentation successful"
        if single.confidence_interval:
            message += f" | acc=[{_fmt_conf(single.confidence_interval[0])},{_fmt_conf(single.confidence_interval[1])}]"
    return TextResult(single.union_mask, message, None, single)


def segment_text(processor, model, image: Image.Image, stages: Sequence[StageConfig], device: str,
                 initial_crop: Optional[BBox] = None) -> TextResult:
    """A text request as the robot made it through the ROS node.

    One stage: the ROS node sent the request without stages (sam3_ros2 backend_client.HTTPBackendClient.segment,
    Apache-2.0), so the backend took its one-stage path, which ignores padding and single_detection_only
    (segment_one_stage). Two stages or more: the multistage path (ImageSegmentation.handle_request with "stages",
    orchestrate_multistage), and the message gets the per-stage summary, as in the run logs."""
    stages = list(stages)
    if len(stages) == 1:
        return segment_one_stage(processor, model, image, stages[0].prompt, stages[0].confidence_threshold, device,
                                 crop=initial_crop)

    def run_single_pass(stage_image: Image.Image, prompt: str, threshold: float) -> SinglePass:
        return run_text_pass(processor, model, stage_image, prompt, threshold, device)

    result = orchestrate_multistage(image, run_single_pass, stages, initial_static_crop=initial_crop)
    summary = stage_summary(stages, result)
    return TextResult(result.final_mask, f"{result.message} | {summary}" if summary else result.message, result, None)


def probability_to_mask_logit(probability: float) -> float:
    """The per-pixel threshold of a click request as a logit (image_segmentation.probability_to_mask_logit, arms_sam3
    bc48b4f). 0.0 means unset and gives logit 0, a probability of 0.5, the model's default. The runs never set it."""
    if probability == 0.0:
        return 0.0
    p = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class PointResult:
    """What the backend returned for a click request, with the three candidates' predicted IoUs."""

    mask: np.ndarray                      # uint8 {0, 1}, the input image's size
    predicted_iou: float                  # of the chosen candidate
    presence: float                       # sigmoid of the object-score logit
    index: int                            # the chosen candidate, 0 to 2
    iou_scores: list[float]               # the predicted IoU of every candidate
    message: str
    candidates: Optional[np.ndarray] = field(default=None, compare=False, repr=False)  # (3, H, W) uint8, all three


def segment_points(processor, model, image: Image.Image, points: Sequence[Sequence[float]], labels: Sequence[int],
                   device: str, object_score_threshold: float = 0.0, mask_pixel_threshold: float = 0.0) -> PointResult:
    """A click request (ImageSegmentation.handle_points_request and run_point_segmentation, arms_sam3 bc48b4f; the
    same choice as e5c0bdd): one forward pass of the tracker head with its default multimask_output=True, the
    candidate with the highest predicted IoU (iou_scores[0, 0].argmax()), its logits resized to the image and
    binarised at logit 0 (probability 0.5). Points are pixels (x = column, y = row), rounded to integers as the
    backend did; labels are 1 (positive) and 0 (negative). If the presence probability is below
    `object_score_threshold` the mask is empty; the runs sent 0.0, so they always got the mask."""
    import torch
    inputs = processor(images=image.convert("RGB"), input_points=[[[[int(x), int(y)] for x, y in points]]],
                       input_labels=[[[int(label) for label in labels]]], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    best = int(outputs.iou_scores[0, 0].argmax())
    masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"],
                                         mask_threshold=probability_to_mask_logit(mask_pixel_threshold))[0]
    mask = masks[0][best].to(torch.uint8).cpu().numpy()
    iou = float(outputs.iou_scores[0, 0, best])
    presence = float(torch.sigmoid(outputs.object_score_logits[0, 0, 0]))
    ious = [float(v) for v in outputs.iou_scores[0, 0].detach().float().cpu()]
    candidates = masks[0].to(torch.uint8).cpu().numpy()       # all three, for diagnostics; the backend kept one
    if presence < object_score_threshold:
        return PointResult(np.zeros((image.height, image.width), dtype=np.uint8), iou, presence, best, ious,
                           f"No object at clicks (presence {presence:.2f} < threshold {object_score_threshold:.2f})",
                           candidates)
    message = (f"Segmentation successful ({len(points)} clicks, best IoU {iou:.2f}, presence {presence:.2f}, "
               f"{int(mask.sum())} px" + (f", pixel thr {mask_pixel_threshold:.2f}" if mask_pixel_threshold > 0.0 else "")
               + ")")
    return PointResult(mask, iou, presence, best, ious, message, candidates)
