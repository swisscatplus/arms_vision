"""ssvision.sam3_pipeline without SAM 3: the service's combination of detections, on fake passes with a known answer.

Each fake pass returns detections as SAM 3's post-processor would (masks of the stage's image, scores above the
threshold), so these tests need neither torch nor the weights. The comparison with the service's own code, on real
frames and with SAM 3, is in docs/segmentation_service.md."""
import numpy as np
import pytest

from ssvision import sam3_pipeline as sp

H, W = 80, 100


def blob(shape, x0, y0, x1, y1) -> np.ndarray:
    m = np.zeros(shape, dtype=np.int64)            # the post-processor's masks are int64 {0, 1}
    m[y0:y1, x0:x1] = 1
    return m


def fake_passes(monkeypatch, by_prompt):
    """run_text_pass replaced: by_prompt[prompt](height, width) -> [(mask, score), ...]; records each call."""
    calls = []

    def run_text_pass(processor, model, image, prompt, threshold, device):
        calls.append((prompt, image.size))
        dets = [sp.Detection(m, f"{prompt} ({s:.2f})", s) for m, s in by_prompt[prompt](image.height, image.width)
                if s > threshold]
        dets.sort(key=lambda d: -d.score)
        return sp.build_single_pass((image.height, image.width), prompt, threshold, dets)

    monkeypatch.setattr(sp, "run_text_pass", run_text_pass)
    return calls


def image():
    from PIL import Image
    return Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))


STAGES = [sp.StageConfig("scoop", 0.4), sp.StageConfig("white material on the scoop", 0.4)]


def two_spatulas(h, w):          # stage 1 finds two blobs: the union's box spans both
    return [(blob((h, w), 10, 20, 30, 40), 0.87), (blob((h, w), 60, 50, 70, 60), 0.55)]


def material(h, w):              # stage 2 finds material in the top-left corner of whatever crop it gets
    return [(blob((h, w), 0, 0, 5, 4), 0.9)]


def test_png_round_trip_keeps_the_colours():
    rgb = np.random.default_rng(0).integers(0, 256, (12, 17, 3), dtype=np.uint8)
    back = sp.decode_request_image(sp.png_from_bgr8(rgb[:, :, ::-1]))     # the robot sent BGR, the backend saw RGB
    assert np.array_equal(np.asarray(back), rgb)


def test_mask_bbox():
    m = blob((H, W), 3, 2, 8, 5)
    assert sp.mask_bbox(m) == (3, 2, 8, 5)                                # x2, y2 exclusive
    assert sp.mask_bbox(m, padding=4) == (0, 0, 12, 9)                    # clamped to the image
    assert sp.mask_bbox(np.zeros((4, 4))) is None


def test_union_of_detections_as_the_runs(monkeypatch):
    calls = fake_passes(monkeypatch, {"scoop": two_spatulas, "white material on the scoop": material})
    res = sp.segment_text(None, None, image(), STAGES, "cpu")
    assert res.stages.all_stage_bboxes[0] == (10, 20, 70, 60)             # the box of the union of both blobs
    assert calls[1] == ("white material on the scoop", (60, 40))          # stage 2 ran on that crop
    assert np.array_equal(np.nonzero(res.mask.any(0))[0][[0, -1]], [10, 14])
    assert res.mask.sum() == 20 and res.mask[20:24, 10:15].all()          # pasted back at the crop's corner
    assert res.message == ("2-stage segmentation successful | s0 'scoop' thr=0.40 ok acc=[0.550,0.870] | "
                           "s1 'white material on the scoop' thr=0.40 ok acc=[0.900,0.900]")


def test_single_detection_only_as_since_2026_09_23(monkeypatch):
    fake_passes(monkeypatch, {"scoop": two_spatulas, "white material on the scoop": material})
    stages = [sp.StageConfig(s.prompt, s.confidence_threshold, s.padding, True) for s in STAGES]
    res = sp.segment_text(None, None, image(), stages, "cpu")
    assert res.stages.all_stage_bboxes[0] == (10, 20, 30, 40)             # the top detection alone
    assert res.stages.all_stage_masks[0].sum() == 20 * 20


def test_failed_first_stage_gives_the_message_of_the_run_logs(monkeypatch):
    fake_passes(monkeypatch, {"scoop": lambda h, w: [], "white material on the scoop": material})
    res = sp.segment_text(None, None, image(), STAGES, "cpu")
    assert not res.mask.any()
    # NaCl trial 1, read 0 in the run log, without the robot's prefix and the debug path:
    assert res.message == ("pipeline stopped at stage 0: no_detections | s0 'scoop' thr=0.40 no_detections (model "
                           "returned no scored detections) | s1 'white material on the scoop' thr=0.40 pipeline "
                           "stopped at stage 0 (model returned no scored detections)")


def test_threshold_on_the_two_decimal_label():
    dets = [sp.Detection(blob((H, W), 0, 0, 2, 2), "scoop (0.40)", 0.39996),
            sp.Detection(blob((H, W), 5, 5, 7, 7), "scoop (0.39)", 0.39)]
    single = sp.build_single_pass((H, W), "scoop", 0.4, dets)
    assert [d.label for d in single.accepted] == ["scoop (0.40)"]
    assert single.confidence_interval == (0.40, 0.40)


def test_one_stage_request_takes_the_union_without_cropping(monkeypatch):
    calls = fake_passes(monkeypatch, {"scoop": two_spatulas})
    res = sp.segment_text(None, None, image(), [sp.StageConfig("scoop", 0.4, 0, True)], "cpu")
    assert calls == [("scoop", (W, H))] and res.stages is None
    assert res.mask.sum() == 20 * 20 + 10 * 10                            # both blobs: the flag is not sent
    assert res.message == "Segmentation successful | acc=[0.550,0.870]"


def test_click_threshold_logit():
    assert sp.probability_to_mask_logit(0.0) == 0.0                       # unset: the model's 0.5
    assert sp.probability_to_mask_logit(0.5) == pytest.approx(0.0)
    assert sp.probability_to_mask_logit(0.9) == pytest.approx(np.log(9.0))
