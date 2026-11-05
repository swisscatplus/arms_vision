"""Surface estimate from one side-view image: the material in the vial -> surface height and tilt (paper: Methods,
Vision-Based Surface Estimation; Figure 5, top row).

From the stored SAM 3 mask of a trial (no SAM 3 needed):

    python demo/estimate_surface.py                       # the three demo trials
    python demo/estimate_surface.py --material polenta    # one of them (salt, sugar, polenta)
    python demo/estimate_surface.py --plot                # also draw the fit to demo/out/
    python demo/estimate_surface.py --sample salt_t23     # another trial, after scripts/make_demo_data.py --trial salt:23

From the image, with SAM 3 installed (optional, see the README): the mask is first recomputed on the full frame with
the run's two point prompts, processed as the service of the runs did (the best of SAM 3's three candidate masks by
predicted IoU, ssvision.sam3_pipeline), then the same geometry runs on it. Needs the full frames
(scripts/download_data.py demo).

    python demo/estimate_surface.py --sam3 [--device cpu|mps|cuda] [--sam3-model facebook/sam3]

The geometry is ssvision.surface (the robot's surface observer): the largest contour of the mask, its upper boundary
in 40 bins, a RANSAC line (two-point samples, 600 iterations, seed 0, inliers within 0.75 mm, total-least-squares
refit), and the point p where the line crosses the vial axis. Two fixed reference pixels on the axis and the vial
length they span (57.0 mm) give the scale. The depth of p below the lip and the tilt of the line are the estimate.

The robot stored the result as T_vialLip_surface (a rotation by the tilt about x_lip, origin [0, 0, depth]), in
float32. The script prints its own result next to the stored one; on the stored mask they agree to float32 rounding
for every trial of the three runs.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from demo_data import LABEL, OUT, TRIALS, VIAL_FLOOR_BELOW_LIP_MM, load_trial
from ssvision.surface import N_BINS, estimate_surface

TOL_Z_MM, TOL_TILT_DEG = 1e-4, 1e-4   # "to rounding": the stored pose is float32


def plot(rgb, mask, origin, res, rec, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    over = rgb.astype(float)
    over[mask] = 0.6 * over[mask] + 0.4 * np.array([60, 120, 255])
    o = np.asarray(origin, float)
    vc = np.asarray(rec["surface_stored"]["vial_clicks_image_px"], float) - o
    fig, ax = plt.subplots(figsize=(7, 7 * rgb.shape[0] / rgb.shape[1]), constrained_layout=True)
    ax.imshow(over.astype(np.uint8))
    c, v = res["ends"].mean(axis=0) - o, np.diff(res["ends"], axis=0)[0]
    v = v / np.linalg.norm(v)
    tt = np.array([-2000.0, 2000.0])
    ax.plot(c[0] + tt * v[0], c[1] + tt * v[1], color="#ff3030", lw=1.4, label="fitted surface line")
    b, inl = res["boundary"] - o, res["inliers"]
    ax.scatter(*b[~inl].T, s=8, color="white", label="upper boundary, outliers")
    ax.scatter(*b[inl].T, s=8, color="#ff3030", label="upper boundary, inliers")
    ax.plot(vc[:, 0], vc[:, 1], color="white", ls="--", lw=1.0, label="vial axis (towards the lip)")
    ax.scatter(*(res["p_img"] - o), marker="o", s=40, facecolor="none", edgecolor="yellow", lw=1.5, label="p")
    ax.set_xlim(0, rgb.shape[1]); ax.set_ylim(rgb.shape[0], 0); ax.axis("off")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.85)
    ax.set_title(f"{LABEL[rec['material']]} trial {rec['trial']}: {res['depth_mm']:.2f} mm below the lip, "
                 f"tilt {res['tilt_deg']:.2f} deg", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(stem: str, args) -> bool:
    t = load_trial(stem, full=args.sam3)
    rec = t["rec"]
    st = rec["surface_stored"]
    stored_mask, origin = t["side_mask"], t["side_mask_origin"]
    if args.sam3:
        from ssvision.segment import side_view_mask
        mask, info = side_view_mask(t["side"], st["material_click_px"][0], st["negative_click_px"][0],
                                    args.sam3_model, args.device)
        inter, union = int((mask & stored_mask).sum()), int((mask | stored_mask).sum())
        source = info["note"] + (f"; {int(mask.sum())} px, IoU {inter / union:.4f} with the stored mask "
                                 f"({int(stored_mask.sum())} px)")
    else:
        mask, source = stored_mask, f"stored SAM 3 mask ({int(stored_mask.sum())} px)"
    res = estimate_surface(mask, st["vial_clicks_image_px"], st["vial_length_mm"], st["ransac_tol_mm"], origin=origin)
    dz = res["depth_mm"] - st["surface_z_lip_mm"]
    dt = res["tilt_deg"] - st["tilt_about_x_lip_deg"]
    pc = np.asarray(st["plane_center_image_px"])
    vc = np.asarray(st["vial_clicks_image_px"])
    print(f"\n{LABEL[rec['material']]}, trial {rec['trial']} (run {rec['run']['zarr']}, s_z = "
          f"{rec['commanded_offset_sz_mm']:+.1f} mm), {source}")
    print(f"  scale             {st['vial_length_mm']:.1f} mm between the reference pixels ({vc[0, 0]:.0f}, {vc[0, 1]:.0f}) and "
          f"({vc[1, 0]:.0f}, {vc[1, 1]:.0f}): {res['mm_per_px']:.6f} mm/px; inlier band +-{st['ransac_tol_mm']} mm = "
          f"+-{res['tol_px']:.2f} px")
    print(f"  upper boundary    {len(res['boundary'])} of {N_BINS} bins kept; RANSAC inliers {int(res['inliers'].sum())} "
          f"of {len(res['boundary'])}")
    print(f"  surface point p   ({res['p_img'][0]:.3f}, {res['p_img'][1]:.3f}) px     stored ({pc[0]:.3f}, {pc[1]:.3f}) px")
    print(f"  depth below lip   {res['depth_mm']:+.5f} mm        stored {st['surface_z_lip_mm']:+.5f} mm    "
          f"difference {dz:+.1e} mm")
    print(f"  tilt about x_lip  {res['tilt_deg']:+.5f} deg       stored {st['tilt_about_x_lip_deg']:+.5f} deg   "
          f"difference {dt:+.1e} deg")
    print(f"  height above the vial floor ({VIAL_FLOOR_BELOW_LIP_MM} mm below the lip, as in Figure 7): "
          f"{VIAL_FLOOR_BELOW_LIP_MM + res['depth_mm']:.2f} mm")
    ok = abs(dz) <= TOL_Z_MM and abs(dt) <= TOL_TILT_DEG
    if args.sam3:
        print("  (SAM 3 mask recomputed here, so the result is compared with the stored one, not required to match it)")
    else:
        print(f"  -> {'reproduces the stored estimate to float32 rounding' if ok else 'DIFFERS from the stored estimate'}")
    if args.plot:
        out = OUT / f"{stem}_surface{'_sam3' if args.sam3 else ''}.png"
        plot(t["side"], mask, origin, res, rec, out)
        print(f"  wrote demo/out/{out.name}")
    return ok or args.sam3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--material", choices=sorted(TRIALS), help="one demo trial (default: all three)")
    ap.add_argument("--sample", help="another trial in demo/data by its file stem, e.g. salt_t23 "
                                     "(written by scripts/make_demo_data.py --trial salt:23)")
    ap.add_argument("--sam3", action="store_true", help="recompute the mask from the full frame with SAM 3 first")
    ap.add_argument("--sam3-model", default="facebook/sam3", help="Hugging Face id or local folder of the SAM 3 weights")
    ap.add_argument("--device", default="auto", help="auto, cpu, mps or cuda (with --sam3)")
    ap.add_argument("--plot", action="store_true", help="write demo/out/<trial>_surface.png")
    args = ap.parse_args()
    stems = [args.sample] if args.sample else ([TRIALS[args.material]] if args.material else list(TRIALS.values()))
    ok = [run(s, args) for s in stems]
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
