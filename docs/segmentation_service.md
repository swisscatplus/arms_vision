# The segmentation service of the runs

Every SAM 3 mask stored in the run logs, on which the paper's surface and mass estimates rest, came from a
segmentation service on a GPU machine. This page records what the service did with each image, with the source of each statement, and how
[`ssvision/sam3_pipeline.py`](../ssvision/sam3_pipeline.py) reproduces it without ROS, an HTTP server or the GPU
machine.

## Where the code is

| Part | Repository | What it did in the runs |
|---|---|---|
| Robot client | `arms_ws`, the robot's code: `tools/solid_sampling/state/spatula_observer.py`, `state/surface_observer.py`, `hardware/sam3_real.py`, `hardware/sam2_clicks_real.py` | built each request; stored the mask and, for a failed read, the service's message |
| ROS 2 node | `sam3_ros2`, in `swisscatplus/arms_gpu_ws` (`src/sam3_ros2`, Apache-2.0) | turned each ROS request into an HTTP request to the backend, and the reply into a mask image |
| Backend | `swisscatplus/arms_sam3`, a submodule of `arms_gpu_ws` (`tools/arms_sam3`) | decoded the image, ran SAM 3 through Hugging Face `transformers`, and combined the detections into one mask |

`arms_gpu_ws` and `arms_sam3` are private repositories of Swiss Cat+, by Jasper Tan. `ssvision/sam3_pipeline.py`
ports the backend's processing function by function (table below). It leaves out the ROS node, the Flask server, the
browser GUI, the YAML crop presets and the debug dumps. Two of its functions follow `sam3_ros2`, whose Apache License
2.0 is in [`ssvision/LICENSE-sam3_ros2.txt`](../ssvision/LICENSE-sam3_ros2.txt).

## Which version ran

The runs were on 2026-09-10 (sugar from 13:46, sodium chloride from 18:52) and 2026-09-11 (polenta from 09:08).
`arms_sam3` has these commits around them:

| Commit | Date | Change |
|---|---|---|
| `e5c0bdd` | 2026-07-07 | click segmentation (merge of `6066413`) |
| `bc48b4f` | 2026-09-15 | the per-stage summary in each reply's message; debug dumps (`--debug-dir`); a per-pixel threshold for clicks |
| `1238357` | 2026-09-23 | the browser GUI; `single_detection_only` takes effect; failed stages report their near-miss scores |

There is no commit between `e5c0bdd` and `bc48b4f`. The workspace `arms_gpu_ws` moved its submodule pointer to
`e5c0bdd` on 2026-07-07 (`632a0b9`), to `bc48b4f` on 2026-09-15 (`2cfa366`) and to `1238357` on 2026-09-23 (`edcf616`).

**The evidence in the run logs.** For each failed read, the robot stored the backend's reply message in the run log
(`events/spatula_observer/notes`). All 240 failed reads of the three runs are reads 0 and 1 of a trial (the trial start
and before the scoop), where the first stage found nothing. Their messages all have this form, here sodium chloride,
trial 1, read 0:

```
sam3 no detection: pipeline stopped at stage 0: no_detections | s0 'scoop' thr=0.40 no_detections (model returned no
scored detections) | s1 'white material on the scoop' thr=0.40 pipeline stopped at stage 0 (model returned no scored
detections) | debug=debug_dumps/20260910-185415-292_fail-stage0
```

In the published run logs, the `debug=` path is given relative to the service's workspace on the GPU machine.

- `sam3 no detection:` is the robot's prefix (`spatula_observer.py`, `Sam3SpatulaLoadObserver.observe`).
- `pipeline stopped at stage 0: no_detections` is the message of `orchestrate_multistage`
  (`scripts/multistage_orchestrator.py`), the same in every commit.
- The per-stage summary is written by `build_stage_diagnostics` (`scripts/image_segmentation.py`), and the `debug=` path
  by `debug_dump.dump_multistage`, whose folders are named `<date>-<time>-<milliseconds>_<ok|fail-stage<i>>`
  (`scripts/debug_dump.py`, `_new_dump_dir`). Both first appear in `bc48b4f`. At `e5c0bdd` the message stops after
  `no_detections`.
- At `1238357` the summary of a failed stage gives its best near-miss score, `best_rejected=...`, whenever SAM 3 scored
  any candidate at 0.01 or more (GUI_DESIGN.md, section 3.7). Run with `1238357`'s own functions on this Mac, the read
  above gives `s0 'scoop' thr=0.40 no_detections best_rejected=0.105`. No message in the run logs has such a score.
- The stored masks point the same way: with `1238357`, the robot's `single_detection_only` would have kept only the
  top detection of each stage, but the stored masks show the union of several detections on 21 reads (section
  "Detections per stage").

**So the backend that served the runs was `arms_sam3` at `e5c0bdd`, plus uncommitted changes that were committed on
2026-09-15 as `bc48b4f`.** The run logs prove that two of `bc48b4f`'s changes were deployed: the per-stage summary and
the debug dumps. They cannot show the others:

- the per-pixel threshold of click requests (`mask_pixel_threshold`). The robot never sent one
  (`hardware/sam2_clicks_real.py`), and the ROS node of 2026-07-07 has no such field (it was added in `d895db3`,
  2026-09-15). At its default of 0 the code gives the masks of `e5c0bdd`, so this does not matter for the masks.
- the new wording of the click reply (`best IoU ..., presence ..., ... px`), which the robot did not store for
  successful reads.
- the passing of rejected scores through `multistage_orchestrator.py` and `segmentation_filtering.py`, which feeds
  only the message, and which `build_stage_diagnostics` does not need.

**For the masks this is enough.** No change from `e5c0bdd` to `bc48b4f` alters a mask for the runs' requests, and the
equivalence test below gives identical masks with the functions of both commits.

**Not recorded:** the exact state of the backend's working tree, the versions of torch and transformers in its
environment, and the revision of the SAM 3 weights it loaded. The repository's own `environment.yml` (2026-03-26,
transformers 5.3.0, torch 2.7.0+cu126) predates the click head, so it does not record the environment of the runs.

## What the service did with an image

### The image

1. The robot sent each frame as a ROS `bgr8` image. The top camera's frame is RGB in the robot's code and is
   reversed to BGR before the request (`spatula_observer.py`, `observe`: `bgr = image.pixels[..., ::-1]`). The side
   camera's frame comes from `cv_bridge` as BGR and is sent as it is (`surface_observer.py` and
   `sam2_clicks_real.py`, `segment_clicks`).
2. The ROS node wrote the `bgr8` array to a PNG with OpenCV (`sam3_ros2/image_converter.py`, `ros_image_to_bytes`),
   which gives a PNG with true colours, and sent it base64-encoded.
3. The backend decoded it with Pillow and converted it to RGB (`image_segmentation.py`, `handle_request` and
   `handle_points_request`).

So SAM 3 saw the frames in true colour, and PNG is lossless: decoding gives back the frame exactly. The run logs store
top frames as RGB and side frames as BGR; `ssvision.segment` takes RGB arrays. The port's `png_from_bgr8` and
`decode_request_image` reproduce steps 2 and 3.

### Top view: a two-stage text request

**The request.** The run configurations (`config_json` in each run log, `spatula_observer`) give, for all three
materials: prompts `["scoop", "<material prompt>"]`, confidence thresholds `[0.4, 0.4]`, paddings `[0, 0]` and
`single_detection_only` `[true, true]`. With two stages, the ROS node sent a `stages` request
(`sam3_ros2/backend_client.py`, `HTTPBackendClient.segment`), so the backend ran `orchestrate_multistage`.

**One stage, one pass** (`ImageSegmentation.run_image_segmentation`, then `build_single_pass_result` in
`segmentation_filtering.py`):

1. SAM 3's concept head (`Sam3Model`) runs on the stage's image with the stage's prompt.
2. `post_process_instance_segmentation(threshold=0.4, mask_threshold=0.5)` of `transformers` scores each of the
   model's candidate queries as sigmoid(logit) times sigmoid(presence logit), keeps those scoring above 0.4, resizes
   their mask probabilities bilinearly to the image and binarises them at 0.5 (as read in transformers 5.18.0; the
   service's version is not recorded).
3. The backend sorts the detections by score and writes each score into a label with two decimals,
   `"scoop (0.87)"`. From here on it reads the score from the label: a detection is dropped if this two-decimal score
   is below the threshold (which cannot happen at 0.4, since the post-processor kept only scores above 0.4) or if its
   mask is empty.
4. **The stage's mask is the union of the masks of all the detections it kept** (`_union_masks`).

**From stage 1 to stage 2** (`multistage_orchestrator.py`, `_evaluate_stage_decision` and `orchestrate_multistage`):

1. A stage that kept no detection stops the pipeline. The reply is then an all-zero mask with the message
   `pipeline stopped at stage <i>: no_detections`. The robot read the stage number from it
   (`spatula_observer.py`, `_empty_mask_is_failure`): at stage 0 the spatula was not found and the read is a failure;
   at stage 1 the spatula is clean, a reading of 0 mg.
2. `single_detection_only` was read, but it narrowed the list only for the check above. **The box and the stage's
   mask both came from the union of every accepted detection, so the flag had no effect** (arms_gpu_ws,
   `design_docs/GUI_DESIGN.md`, section 3.5).
3. The box is the tight box of the union mask's pixels, with exclusive right and bottom edges, plus the padding
   (0 in the runs), clamped to the image (`mask_geometry.derive_mask_bounding_boxes`). It is not SAM 3's predicted
   box.
4. Stage 2 runs on the image cropped to that box (Pillow's `crop`). Its union mask is pasted back into an all-zero
   frame at the box's top-left corner (`_propagate_mask_through_offsets`, `_remap_mask_into`). That full-frame mask is
   the reply.

### Side view: a click request

`ImageSegmentation.handle_points_request` and `run_point_segmentation` (the same at `e5c0bdd` and `bc48b4f`; at
`bc48b4f` the forward pass moved into `_tracker_forward`):

1. The request carries the positive point (638, 718) and the negative point (1198, 272) of the run configuration
   (`surface.material_click` and `negative_click`), rounded to integers, labels `[1, 0]`, and
   `object_score_threshold = 0.0` (`surface_observer.py`, `sam2_clicks_real.py`).
2. SAM 3's tracker head (`Sam3TrackerModel`, from the same checkpoint) runs once with its default
   `multimask_output=True`, which gives three candidate masks and a predicted IoU for each.
3. **The candidate with the highest predicted IoU is kept** (`outputs.iou_scores[0, 0].argmax()`). Its logits are
   resized bilinearly to the image and binarised at logit 0, a probability of 0.5 (`post_process_masks` with its
   default `mask_threshold=0.0`; arms_gpu_ws, `design_docs/CLICK_SEGMENTATION_DESIGN.md`, section 1.1).
4. A presence probability below `object_score_threshold` would give an empty mask; at 0.0 the mask was always
   returned.

### Precision and device

- `ImageSegmentation.__init__` loaded `facebook/sam3` without a revision, on `cuda`, in FP32 (the checkpoint's dtype),
  and set `torch.backends.cuda.matmul.allow_tf32` and `torch.backends.cudnn.allow_tf32` to True.
- It also entered `torch.autocast(bfloat16)`, but only in the thread that built the object. Autocast state is per
  thread, and Flask served each request on a worker thread, so **every served request, including all requests of the
  runs, ran in FP32 with TF32 matrix products** (GUI_DESIGN.md, section 3.6).

### Debug dumps

The server ran with `--debug-dir` (the run logs name the dump of each failed read). With the dump
code of `bc48b4f`, deployed before its commit (above), every text request wrote a folder
(`debug_dump.dump_multistage`) with:

- `input.jpg`, and for each stage `stage<i>_input_crop.jpg` (JPEG, quality 88 in `bc48b4f`, so not the exact pixels)
  and `stage<i>_mask_full.png` (the stage's union mask, lossless);
- `summary.json`: per stage the input crop box, the detection box, the accepted confidence interval (the lowest and
  highest two-decimal score kept), the rejected scores and the mask's pixel count.

Click requests were not dumped. The run logs name the folders of the 240 failed reads; the 360 folders of the
successful reads (`*_ok`) should sit next to them, from `20260910-134804` (sugar, trial 1) to the end of the polenta
run on 2026-09-11. A stage whose accepted
confidence interval has a lowest score below its highest kept more than one detection; equal ends are consistent with
one. The stage masks and detection boxes can be compared with the port's directly.

## What changed on 2026-09-23

`1238357` made `single_detection_only` take effect (GUI_DESIGN.md, section 3.5): a stage with the flag keeps only its
top detection (the highest two-decimal score; ties to the earlier one in the sorted list), and both the next crop and
the stage's mask come from it. Since the robot's configuration sets the flag on both stages, the same requests give a
different mask after 2026-09-23 wherever a stage had more than one detection above 0.4. With the flag off, the result
is bit-identical to before. One-stage requests from ROS are not affected: the ROS node sends them without stages, and
the one-stage path always takes the union. `1238357` also added the near-miss scores to the message of a failed stage
(section 3.7), which changes no mask.

## The port

`ssvision/sam3_pipeline.py`, with `ssvision/segment.py` calling it:

| Port | Reproduces (arms_sam3 `bc48b4f` unless stated) |
|---|---|
| `decode_request_image` | the image decoding of `ImageSegmentation.handle_request` and `handle_points_request` |
| `png_from_bgr8` | `sam3_ros2/image_converter.py`, `ros_image_to_bytes` (Apache-2.0) |
| `load_text_model`, `load_tracker_model` | the model loading of `ImageSegmentation.__init__` and `_load_tracker`: FP32, TF32 allowed on CUDA, no autocast |
| `run_text_pass`, `build_single_pass`, `extract_confidence`, `binarize_mask` | `ImageSegmentation.run_image_segmentation`; `segmentation_filtering.py`: `build_single_pass_result`, `filter_annotations`, `_union_masks`, `extract_confidence`, `binarize_mask` |
| `mask_bbox` | `mask_geometry.derive_mask_bounding_boxes` (the clamped box) |
| `evaluate_stage` | `multistage_orchestrator._evaluate_stage_decision`: as at `bc48b4f` with `single_detection_only=False`, as at `1238357` with True |
| `orchestrate_multistage` | `multistage_orchestrator.orchestrate_multistage`, with `_clamp_bbox_to_image_bounds`, `_remap_mask_into` and `_propagate_mask_through_offsets` |
| `stage_summary` | `image_segmentation.build_stage_diagnostics` (the message only) |
| `segment_text` | `ImageSegmentation.handle_request`, and the dispatch by number of stages of `sam3_ros2/backend_client.py`, `HTTPBackendClient.segment` (Apache-2.0) |
| `segment_one_stage` | the path of `handle_request` for a request without stages |
| `segment_points`, `probability_to_mask_logit` | `ImageSegmentation.handle_points_request`, `run_point_segmentation`, `_tracker_forward`, `probability_to_mask_logit` |

`ssvision.segment.top_view_mask` and `side_view_mask` call `segment_text` and `segment_points` with the run's prompts.
Their defaults are the service as it ran; `top_view_mask(..., single_detection_only=True)` gives the behaviour since
2026-09-23. As on the robot, a first stage that finds nothing raises `RuntimeError` (a failed read), and a second
stage that finds nothing gives an empty mask (0 mg).

## Equivalence with the service's own code

The test is not in this repository, because it imports `arms_sam3`, which is private. It ran in a separate
environment with torch 2.11.0, torchvision 0.26.0 and transformers 5.18.0 and the weights `3c879f39`, on an Apple M1
Max, once on the CPU and once with MPS, both in FP32. For each frame it gives `arms_sam3`'s own
`ImageSegmentation.handle_request` or `handle_points_request` the request the ROS node built (the robot's BGR array,
written to PNG by OpenCV, in base64), gives the port the RGB frame, and compares the results. Both sides use the same
loaded models. The backend object is built without its constructor, as `arms_sam3`'s own tests do, so that it runs on
the chosen device without autocast, as a served request did.

The frames are 16 reads of the runs:

- the after-scoop top view and the side view of the three demo trials;
- read 0 of sodium chloride trial 1, where the first stage found nothing;
- six after-scoop top views where the second stage kept several detections or sat at the threshold: sodium chloride
  trials 1, 28 and 30 (12, 15 and 21 detections), sugar trial 14 (14, the top two tied at 0.85), polenta trial 16
  (4) and polenta trial 11 (one, at 0.4015);
- the side views of sodium chloride trials 1, 14 and 24, whose two best candidates are within 0.003 in predicted IoU;
- the request without stages (the one-stage path), on two of the top views.

Identical masks, on the CPU and with MPS alike:

| `arms_sam3` | Two-stage text requests (10 top views) | Click requests (6 side views) | One-stage requests (2) |
|---|---|---|---|
| `e5c0bdd` | the runs' request (`single_detection_only` true), against the port as the service ran: 10 of 10 | 6 of 6 | 2 of 2 |
| `bc48b4f` | the same: 10 of 10 | 6 of 6 | 2 of 2 |
| `1238357` | the runs' request, against the port with `single_detection_only=True`: 10 of 10; the flag false, against the port as the service ran: 10 of 10 | 6 of 6 | 2 of 2 |

The stage boxes are identical too, wherever the backend reports them (`bc48b4f` and `1238357`). The messages are
identical at `bc48b4f`, including the failed read, where both equal the message in the run log without the robot's
prefix and the debug path. At `e5c0bdd` only the wording of the click and one-stage replies differs (that commit has no
per-pixel threshold and no summary), and at `1238357` only the failed read differs (`best_rejected=0.105`).

## Recomputed and stored masks

The port reproduces the service's processing, but not its hardware and software: the service ran on a CUDA GPU with
TF32 matrix products, with versions of torch and transformers and a revision of the weights that were not recorded
(above). Scores therefore differ slightly, and a detection near the threshold or two candidates near a tie can come
out differently. `paper/analysis/tools/sam3_recompute.py` measures the effect on every after-scoop top view and every
side view of the three runs, here on Apple silicon (MPS, FP32; record:
[`paper/expected/sam3_recompute.txt`](../paper/expected/sam3_recompute.txt), with one row per read in
`sam3_recompute_top.csv` and `sam3_recompute_side.csv`):

| 40 reads per material | Top view: IoU with the stored mask, median (lowest) | change in the mass estimate, median (largest) | Side view: IoU, median (lowest) | change in depth, median |
|---|---|---|---|---|
| Sodium chloride | 0.996 (0.828) | 0.018 mg (0.55 mg) | 0.956 (0.561) | 0.012 mm |
| Sugar | 0.997 (0.946) | 0.008 mg (0.34 mg) | 0.985 (0.935) | 0.049 mm |
| Polenta | 0.994 (0.000) | 0.018 mg (0.63 mg) | 0.996 (0.990) | 0.009 mm |

- **Top view.** The masks agree closely. The lowest IoU, polenta trial 11, is a threshold case: the stored mask is
  empty, and here the second stage finds one detection at 0.4015.
- **Side view.** For sugar and polenta the stored mask is almost always SAM 3's third candidate, and the service and
  the port choose it alike. For sodium chloride the first and third candidates are often near a tie (predicted IoUs
  within 0.01 on 8 of 40 reads), and on 6 reads the port chooses another candidate than the service did, which gives
  IoUs of 0.56 to 0.70.
- **The surface estimate is sensitive to small changes of the mask.** Its depth changes by more than 1 mm on 9 of
  the 120 side views (8 of sodium chloride, 1 of sugar), some with IoUs above 0.93. Among them are the paper's two
  surface failures, sodium chloride trials 14 and 36: on the recomputed masks they give 32.8 and 33.2 mm below the lip,
  in line with the other trials, against the stored 21.6 and 11.7 mm. On trial 14 the service kept the candidate that
  ranks second here (predicted IoUs 0.816 and 0.813); on trial 36 both kept the same candidate (IoU 0.968), and the
  small difference between the two masks moved the fitted line.

**Simpler choices match less well.** Taking only the top detection of each stage, with SAM 3's predicted box as the
crop, and the tracker's single-mask output (`multimask_output=False`), as an earlier draft of this repository did,
gives top-view IoUs with medians of 0.985, 0.987 and 0.979 and minima of 0.04, 0.07 and 0, and changes the mass
estimate by more than 1 mg on 13 of the 120 reads (up to 86 mg). The service's processing raises the top-view IoU
on 97 of the 120 reads and lowers it on one, and raises the side-view IoU on 58 and lowers it on 2 (two sodium
chloride near ties).

## Detections per stage

`paper/analysis/tools/sam3_recompute.py` recomputed every after-scoop top view of the three runs as the service ran
(record: [`paper/expected/sam3_recompute.txt`](../paper/expected/sam3_recompute.txt), on Apple silicon). Number of
reads, of 40 per material:

| | Stage 1 (`scoop`): more than one detection above 0.4 | Stage 2 (material): none | one | more than one | of these, the stored mask shows the union |
|---|---|---|---|---|---|
| Sodium chloride | 0 | 2 | 23 | 15 | 13 |
| Sugar | 0 | 10 | 21 | 9 | 7 |
| Polenta | 0 | 5 | 29 | 6 | 1 |

- **Stage 1 always found exactly one spatula,** so the crop of stage 2 was the box of one detection's mask.
- **Stage 2 often found the material as several detections,** on 30 of the 120 reads, up to 21 at once.
- **On 21 of these 30 the stored mask itself shows that the service took the union:** it matches the union of the
  detections and not the top detection alone (IoU higher by more than 0.01). For example, on sodium chloride trial 1
  the union has IoU 0.990 with the stored mask and the top detection alone 0.115. On the other 9 the further
  detections lie almost inside the top one, and the two masks are nearly the same.
- **One count is at the edge:** on sodium chloride trial 21 the second detection scored 0.409 here, close enough to
  0.4 to have fallen below it on the GPU.
- **With `single_detection_only=True`,** the behaviour since 2026-09-23 and what the robot's configuration asked for,
  these reads would lose every detection but the top one: the mass estimate drops by up to 8.7 mg for sodium chloride
  (trial 30), 3.2 mg for sugar (trial 14) and 85.8 mg for polenta (trial 16), with the material's a and b of Table 4.
  Whoever reruns the robot with the current service should therefore set the flag to false to get the runs'
  behaviour.

The debug dumps on the GPU machine would confirm the counts: in a dump's `summary.json`, a stage whose
`accepted_confidence_interval` has two different ends kept more than one detection.
