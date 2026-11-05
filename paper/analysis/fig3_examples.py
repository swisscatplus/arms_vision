"""Fig. 3 (fig:examples): stored SAM 3 masks from the runs.

Top row: side view per material with the surface mask (blue), the fitted surface line
(red, recomputed from the stored mask with the run's own RANSAC procedure), the vial
axis between the two reference pixels (white dashed), and the two prompt clicks
(+ green, - red).  Bottom row: top views of a small, a medium and a large load with
the material mask (red), one per material.

Writes ../output/fig3_examples.{png,pdf}.  Run after extract_runs.py.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

from common import IMAGES, LABEL, load
from extract_runs import RUNS
from ssvision.surface import surface_line   # the run's surface-line procedure (robot code, surface_observer.py)

# (material, target carried mass in mg) for the three top views, small -> large
TOP_VIEWS = [("salt", 1.0), ("sugar", 10.0), ("polenta", 70.0)]
SIDE_VIEWS = ["salt", "sugar", "polenta"]  # one side view each, ~10 mg trial


def overlay(rgb, mask, color, alpha=0.4):
    out = rgb.astype(float)
    mm = mask.astype(bool)
    out[mm] = (1 - alpha) * out[mm] + alpha * np.array(color)
    return out.astype(np.uint8)


def crop(img, mask, pad=140, min_box=600, aspect=1.6):
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    h = max(y1 - y0 + 2 * pad, min_box)
    w = max(x1 - x0 + 2 * pad, int(h * aspect))
    h = int(w / aspect)
    ya, yb = max(cy - h // 2, 0), min(cy + h // 2, img.shape[0])
    xa, xb = max(cx - w // 2, 0), min(cx + w // 2, img.shape[1])
    return (ya, xa), img[ya:yb, xa:xb]


def pick_trial(d, target_mg):
    cand = np.where((d["px_obs2"] > 0) & ~d["surface_fail"])[0]
    k = cand[np.argmin(np.abs(d["carried_mg"][cand] - target_mg))]
    return int(d["iteration"][k]), float(d["carried_mg"][k]), int(d["px_obs2"][k])


def side_panel(ax, mat, trial):
    g = zarr.open_group(str(RUNS[mat]), mode="r")
    cfg = json.loads(dict(g.attrs)["config_json"])["surface"]
    su = g["events"]["surface_observer"]
    i = int(np.where(np.array(su["iteration"]) == trial)[0][0])
    img = np.array(su["image"][i])[:, :, ::-1]
    mask = np.array(su["mask"][i])
    vc = np.array(su["vial_clicks_image"][i])
    c, v, pts, inl = surface_line(mask, vc)
    (oy, ox), view = crop(overlay(img, mask, (60, 120, 255)), mask, pad=250, min_box=900, aspect=1.5)
    ax.imshow(view)
    # fitted surface line, drawn across the crop
    t = np.linspace(-1500, 1500, 2)
    ax.plot(c[0] - ox + t * v[0], c[1] - oy + t * v[1], color="#ff3030", lw=1.6)
    ax.scatter(pts[inl, 0] - ox, pts[inl, 1] - oy, s=3, color="#ff3030", alpha=0.7)
    ax.plot(vc[:, 0] - ox, vc[:, 1] - oy, color="white", lw=1.0, ls="--")
    pos = np.array(cfg["material_click"], float)[0]
    neg = np.array(cfg["negative_click"], float)[0]
    ax.scatter([pos[0] - ox], [pos[1] - oy], marker="+", s=70, color="#20e060", lw=1.8)
    ax.scatter([neg[0] - ox], [neg[1] - oy], marker="_", s=70, color="#ff3030", lw=1.8)
    ax.set_xlim(0, view.shape[1]); ax.set_ylim(view.shape[0], 0)
    ax.axis("off")
    ax.set_title(f"{LABEL[mat]}: surface estimate", fontsize=7)


def top_panel(ax, mat, trial, mass, px):
    g = zarr.open_group(str(RUNS[mat]), mode="r")
    sp = g["events"]["spatula_observer"]
    it = np.array(sp["iteration"]); oi = np.array(sp["observation_index"])
    j = int(np.where((it == trial) & (oi == 2))[0][0])
    img = np.array(sp["image_camera44"][j])                       # stored RGB (side frames are BGR)
    mask = np.array(sp["mask_camera44"][j])
    _, view = crop(overlay(img, mask, (255, 60, 60), 0.45), mask, pad=160, min_box=520, aspect=1.5)
    ax.imshow(view); ax.axis("off")
    ax.set_title(f"{LABEL[mat]}: {mass:.1f} mg, {px} px", fontsize=7)


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.3), constrained_layout=True)
    for ax, mat in zip(axes[0], SIDE_VIEWS):
        d = load(mat)
        tr, _, _ = pick_trial(d, 10.0)
        side_panel(ax, mat, tr)
    for ax, (mat, target) in zip(axes[1], TOP_VIEWS):
        d = load(mat)
        tr, mass, px = pick_trial(d, target)
        top_panel(ax, mat, tr, mass, px)
    IMAGES.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(IMAGES / f"fig3_examples.{ext}", dpi=220, bbox_inches="tight")
    print("wrote", IMAGES / "fig3_examples.png")


if __name__ == "__main__":
    main()
