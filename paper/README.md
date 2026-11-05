# Reproducing the paper

> J. Hermus, J. Tan, S. Calinon, P. Miéville and E. Senft, "Quantitative Vision for Solid Sampling at the Milligram
> Scale", *Advanced Robotics Research*, submitted, 2026.

This folder reproduces the numbers, tables and data figures of the paper from the per-trial data in `data/`. Install
the `ssvision` package as in the top-level README, then run:

```bash
python paper/reproduce.py
```

It runs `analysis/make_all.py`, which prints every number of the Results and Tables 1 to 4, and checks the output line
by line against `expected/make_all.txt`. It then draws Figures 6 to 8 into `output/` and compares their pixels with the
figures in the paper. It takes about 20 s and ends with "the numbers match the expected output". Figure 5 also needs
the run logs:

```bash
python scripts/download_data.py runs
```

Each script also runs on its own from `analysis/`, for example `python table3_vision_fit.py`.

## Scripts

| Script | Gives |
|---|---|
| `table1_depth_mass.py` | carried mass per offset (Table 3), surface-estimation failures, transfer loss and the deposit |
| `table3_vision_fit.py` | power-law fit per material, leave-one-out errors, mis-segmentations (Table 4, Detected column of Table 1) |
| `table2_merged.py` | the LaTeX rows of Table 3 |
| `table5_trials.py` | how every trial enters each statistic (Table 2) |
| `table4_cycletime.py` | times between logged events and the trial period (Timing) |
| `prompt_sweep_table.py` | the prompt screening (Screened column of Table 1) |
| `surface_compare.py` | vision against the force probe, and the time per estimate |
| `make_all.py` | all of the above in one printout |

| Figure | Script | Needs beyond `data/` |
|---|---|---|
| 1 | `fig_materials.py --faded` | photographs |
| 2 | `fig_cameras.py --variant compact` | photographs |
| 3 | `fig_cycle.py` | photographs, run logs |
| 4 | `fig_tp_scoop.py` | scooping model code, NaCl run log |
| 5 | `fig3_examples.py` | run logs |
| 6 | `fig2.py --linear --name=fig2_benchmark_linear_floor` | |
| 7 | `fig_surface_height.py` | |
| 8 | `fig_deposit.py` | |

The thresholds quoted in Statistical Analysis (empty scoop, surface-estimation failure, mis-segmentation) are set in
`analysis/common.py`. The power-law fit and the surface line are the same `ssvision` code the demo runs. The two
percentages of the Discussion (time spent weighing and probing) are computed by hand from the NaCl block of
`table4_cycletime.py` and from `surface_compare.py`.

## Data

| File | Contents |
|---|---|
| `<material>_trials.csv` | one row per trial, 40 per material (`salt` = NaCl, `sugar`, `polenta`) |
| `<material>_run_meta.json` | the settings of the run, from its log |
| `surface_compare_2026-09-15-11-22.csv` | vision against force probe on NaCl, five fill levels; `rotate` rows mark the reset before each block |
| `sam3_prompt_sweep_new.csv` | the prompt screening (the glass-bead rows are not in the paper) |
| `fig_tp_scoop.json`, `fig_cycle/` | inputs of Figures 3 and 4 |

Columns of `<material>_trials.csv`:

| Column | Meaning |
|---|---|
| `iteration` | trial number, in the order run |
| `sz_mm` | commanded offset s_z (mm), positive above the estimated surface |
| `source_before_mg`, `source_after_mg` | source-vial readings before the scoop and after the deposit |
| `carried_mg` | `source_before_mg - source_after_mg` |
| `delivered_mg` | net mass in the goal vial after the deposit |
| `lost_mg` | `carried_mg - delivered_mg` |
| `surface_z_lip_mm` | estimated surface from the vial lip (mm, negative below) |
| `px_obs0` ... `px_obs4` | mask pixel count of the top-camera images: 0 trial start, 1 before the scoop, 2 after the scoop (the mass estimate), 3 before and 4 after the deposit; 0 = no mask |
| `t_*_ns` | Unix time (ns) of each logged event |

In `surface_compare_*.csv`, use `surface_z_mm` (from the lip), as `surface_compare.py` does. The column
`height_above_vial_base_mm` uses a different lip-to-base distance for each method, so Figure 7 converts both with
54.0 mm.

### Run logs

The CSVs were extracted from the run logs, one zarr store per material (about 0.35 GB each) in the Zenodo record.
`python scripts/download_data.py runs` puts them where the scripts read them:

| Record file | Unzipped into |
|---|---|
| `sodium_chloride_run.zarr.zip` | `data/HS1_vision/benchmark_2026-09-10-18-52.zarr` |
| `sugar_run.zarr.zip` | `data/sugar_vision/benchmark_2026-09-10-13-46.zarr` |
| `polenta_run.zarr.zip` | `data/polenta_vision/benchmark_2026-09-11-09-08.zarr` |

`HS1` is the robot's name for NaCl, and each run is named after its start time. For zips downloaded by hand, add
`--from-dir ~/Downloads`. With the logs in place, `python paper/analysis/extract_runs.py` rewrites the CSVs and run
settings byte for byte; it also reads a zip directly (`extract_runs.py salt=sodium_chloride_run.zarr.zip`).
`python scripts/make_demo_data.py --trial salt:23` writes any other trial for the demo scripts.

The published logs keep what the scripts read: the side-view and after-scoop top-view images with their masks, and
the pixel counts and times of all five top-camera reads. The top-view images of the other four reads, the fitted surface
plane (read only by Figure 3), the mass estimate logged online (placeholder calibration) and three arrays the robot
never filled are left out; the rest is as the robot stored it. Side-camera images are in BGR, top-camera images in RGB.

## Figures 1 to 4

These are illustrations, and no number of the paper depends on them. Figures 1 to 3 are built from photographs in
`data/photos/`, which are not published, so only the authors can redraw them. Figure 4 needs the training code and the
deployed scooping model, which are not public; set `EXPERIMENT_2` and `ARMS_SOLID_SAMPLING` to their folders (the
package versions it ran with are in the docstring of `fig_tp_scoop.py`). The method is described in Tan et al. (ChemRxiv, 2026,
doi:10.26434/chemrxiv.15009421), and the NaCl demonstrations that trained it are at
[github.com/swisscatplus/arms_tplqt](https://github.com/swisscatplus/arms_tplqt).

`analysis/tools/` holds the helpers that made inputs of Figures 2 and 3 (their outputs are in `data/`), and
`sam3_recompute.py`, which reruns SAM 3 on the stored frames and compares each mask with the stored one (output in
`expected/sam3_recompute*`). The analysis uses the stored masks, so no number depends on it.
