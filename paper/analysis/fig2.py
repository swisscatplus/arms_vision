"""Fig. 2 (fig:benchmark): (A) carried mass vs commanded depth, (B) mask pixels vs carried
mass with the power-law fits, (C) example side-view and top-view frames with their stored
SAM 3 masks. Writes ../output/fig2_benchmark.png (and .pdf), or ../output/NAME.png with --name=NAME.
Panel A shows every trial; panel B the scoops that were not empty (common.loaded_scoops), with the
non-detections among them marked at zero pixels.

Run: python3 fig2.py [--linear] [--name=NAME]   (after extract_runs.py)
"""
from __future__ import annotations

import sys

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

from common import EMPTY_SCOOP_MG, IMAGES, LABEL, load, loaded_scoops
from extract_runs import RUNS
from ssvision.mass import power_law_fit

COLORS = {"salt": "#1f77b4", "sugar": "#d62728", "polenta": "#ff9f1c"}
MARK = {"salt": "o", "sugar": "s", "polenta": "^"}
JITTER = {"salt": -0.15, "sugar": 0.0, "polenta": 0.15}  # mm, horizontal offset so per-offset points do not overprint
LINEAR = "--linear" in sys.argv  # linear axes in A and B; writes fig2_benchmark_linear.*
OFFSET_AXIS = LINEAR  # panel A x-axis is the commanded offset s_z (positive above the surface), inverted so depth increases to the right
PREVIEW = "--preview" in sys.argv  # write fig2_preview_offset.* instead of the file the paper includes
NAME = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--name=")), None)  # output stem, overrides the above

EXAMPLE_TRIAL = {"salt": 10, "sugar": 10, "polenta": None}  # ~13 mg and ~9 mg; polenta picked at run time (~10 mg, detected)


def _draw_a(ax, mats, small=False):
    for mat in mats:
        d = load(mat)
        ok = ~d["surface_fail"]
        c = COLORS[mat]
        szs = sorted(set(d["sz_mm"]), reverse=True)
        mean = np.array([d["carried_mg"][ok & (d["sz_mm"] == s)].mean() for s in szs])
        sd = np.array([d["carried_mg"][ok & (d["sz_mm"] == s)].std(ddof=1) if (ok & (d["sz_mm"] == s)).sum() > 1 else 0 for s in szs])
        clip = (lambda v: np.asarray(v)) if LINEAR else (lambda v: np.clip(v, 0.05, None))
        jx = 0.0 if small else JITTER[mat]
        xs = d["sz_mm"] if OFFSET_AXIS else -d["sz_mm"]
        ax.scatter(xs[ok] + jx, clip(d["carried_mg"][ok]), s=4 if small else 12, color=c, alpha=0.45, marker=MARK[mat])
        ax.errorbar((np.array(szs) if OFFSET_AXIS else -np.array(szs)) + jx, clip(mean), yerr=sd, color=c, marker=MARK[mat], ms=2.5 if small else 5, lw=0.8 if small else 1.4, ls="none", capsize=0 if small else 2, label=LABEL[mat])
        if (~ok).any():
            ax.scatter(xs[~ok], clip(d["carried_mg"][~ok]), s=20 if small else 60, marker="x", color="k", lw=1.0,
                       label=None if (small or mat != "salt") else "surface failed", zorder=5)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)


def _draw_b(ax, mats, small=False):
    for mat in mats:
        d = load(mat)
        ok = loaded_scoops(d)  # valid surface estimate, not an empty scoop
        x, y = d["px_obs2"], d["carried_mg"]
        fit = ok & (x > 0) & (y > 0)
        a, b, r2, *_ = power_law_fit(x[fit], y[fit])
        c = COLORS[mat]
        ax.scatter(x[fit], y[fit], s=5 if small else 14, color=c, marker=MARK[mat], alpha=0.8,
                   label=None if small else f"{LABEL[mat]}: $b$={b:.2f}, $R^2$={r2:.3f}, n={fit.sum()}")
        xx = np.linspace(x[fit].min(), x[fit].max(), 100) if LINEAR else np.logspace(np.log10(x[fit].min()), np.log10(x[fit].max()), 50)
        ax.plot(xx, a * xx ** b, color=c, lw=0.8 if small else 1.2)
        miss = ok & (x == 0)  # non-detections
        if miss.any():
            ax.scatter(np.full(miss.sum(), 0 if LINEAR else 60), y[miss], marker="x", color=c, s=15 if small else 40, lw=1.0, zorder=5)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)


def panel_a(ax, mats):
    _draw_a(ax, mats)
    ax.set_xlabel("Commanded offset $s_z$ (mm), positive above the surface" if OFFSET_AXIS else "Depth below estimated surface $-s_z$ (mm)")
    ax.set_ylabel("Carried mass (mg)")
    if LINEAR:
        # main view: the 0-15 mg working range; inset: the whole ladder. Reversed limits invert the axis.
        ax.set_xlim(8.7, 2.3); ax.set_ylim(-0.5, 16)
        ax.legend(fontsize=6.5, loc="lower right", framealpha=0.9)
        ins = ax.inset_axes([0.07, 0.55, 0.42, 0.40])
        _draw_b if False else _draw_a(ins, mats, small=True)
        ins.set_xlim(9, -16); ins.set_ylim(-4, 100); ins.tick_params(labelsize=5.5, length=2)
        ins.set_title("full range", fontsize=6, pad=2)
        ins.axvspan(8.7, 2.3, color="k", alpha=0.06)
    else:
        ax.set_yscale("log"); ax.set_ylim(0.04, 150)
        ax.axhline(EMPTY_SCOOP_MG, color="gray", lw=0.6, ls=":")
        ax.text(6.0, 1.1 * EMPTY_SCOOP_MG, f"empty below {EMPTY_SCOOP_MG} mg", fontsize=7, color="gray", va="bottom")
        ax.legend(fontsize=6.5, loc="center right", framealpha=0.9)
    ax.set_title("A. Offset $\\to$ mass", fontsize=9, loc="left")


def panel_b(ax, mats):
    _draw_b(ax, mats)
    ax.set_ylabel("Carried mass (mg)")
    if LINEAR:
        kpx = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}")  # ticks in thousands of pixels
        ax.set_xlabel("Mask pixel count $n_{\\mathrm{px}}$ ($\\times10^3$)")
        ax.xaxis.set_major_formatter(kpx)
        ax.set_xlim(-150, 6500); ax.set_ylim(-0.5, 16)
        ax.legend(fontsize=6.5, loc="lower right", framealpha=0.9)
        ins = ax.inset_axes([0.07, 0.55, 0.42, 0.40])
        _draw_b(ins, mats, small=True)
        ins.xaxis.set_major_formatter(kpx)
        ins.set_xlim(-500, 28000); ins.set_ylim(-4, 100); ins.tick_params(labelsize=5.5, length=2)
        ins.set_title("full range", fontsize=6, pad=2)
        ins.axvspan(-150, 6500, color="k", alpha=0.06)
    else:
        ax.set_xlabel("Mask pixel count $n_{\\mathrm{px}}$")
        ax.axvline(80, color="gray", lw=0.6, ls=":")
        ax.text(85, 0.6, "0 px", fontsize=7, color="gray", rotation=90, va="bottom")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(50, 4e4); ax.set_ylim(0.4, 150)
        ax.legend(fontsize=6.5, loc="upper left", framealpha=0.9)
    ax.set_title("B. Segmented area $\\to$ mass", fontsize=9, loc="left")


def overlay(rgb, mask, color=(255, 60, 60), alpha=0.45):
    out = rgb.astype(float)
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color)
    return out.astype(np.uint8)


def crop_to_mask(img, mask, pad=140, min_box=600):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return img
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    h = max(y1 - y0 + 2 * pad, min_box); w = max(x1 - x0 + 2 * pad, int(min_box * 1.6))
    ya, yb = max(cy - h // 2, 0), min(cy + h // 2, img.shape[0]); xa, xb = max(cx - w // 2, 0), min(cx + w // 2, img.shape[1])
    return img[ya:yb, xa:xb]


def panel_c(axes, mats):
    k = 0
    for mat in mats:
        g = zarr.open_group(str(RUNS[mat]), mode="r")
        E = g["events"]
        d = load(mat)
        tr = EXAMPLE_TRIAL[mat]
        if tr is None:
            cand = np.where((d["px_obs2"] > 0) & ~d["surface_fail"])[0]
            tr = int(d["iteration"][cand[np.argmin(np.abs(d["carried_mg"][cand] - 10.0))]])
        row = int(np.where(d["iteration"] == tr)[0][0])
        # side view + surface mask
        su = E["surface_observer"]
        i = int(np.where(np.array(su["iteration"]) == tr)[0][0])
        side = np.array(su["image"][i])[:, :, ::-1]
        smask = np.array(su["mask"][i])
        ax = axes[k]; k += 1
        ax.imshow(crop_to_mask(overlay(side, smask, (60, 120, 255)), smask, pad=200, min_box=900))
        ax.set_title(f"{LABEL[mat]}\nside view", fontsize=6.5); ax.axis("off")
        # top view + material mask (after-scoop read, obs 2)
        sp = E["spatula_observer"]
        it = np.array(sp["iteration"]); oi = np.array(sp["observation_index"])
        j = int(np.where((it == tr) & (oi == 2))[0][0])
        top = np.array(sp["image_camera44"][j])                       # stored RGB (side frames are BGR)
        tmask = np.array(sp["mask_camera44"][j])
        ax = axes[k]; k += 1
        ax.imshow(crop_to_mask(overlay(top, tmask), tmask))
        ax.set_title(f"{LABEL[mat]} top view\n{int(tmask.sum())} px, {d['carried_mg'][row]:.1f} mg", fontsize=6.5); ax.axis("off")


def main() -> None:
    mats = ["salt", "sugar", "polenta"]
    if LINEAR:
        fig = plt.figure(figsize=(7.2, 3.1), constrained_layout=True)
        gs = fig.add_gridspec(1, 2)
        panel_a(fig.add_subplot(gs[0, 0]), mats)
        panel_b(fig.add_subplot(gs[0, 1]), mats)
    else:
        fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
        gs = fig.add_gridspec(2, 6, height_ratios=[1.7, 1.0])
        panel_a(fig.add_subplot(gs[0, :3]), mats)
        panel_b(fig.add_subplot(gs[0, 3:]), mats)
        axes = [fig.add_subplot(gs[1, i]) for i in range(6)]
        panel_c(axes, mats)
        t0 = axes[0].get_title(); axes[0].set_title("C. " + t0, fontsize=6.5)
    IMAGES.mkdir(exist_ok=True)
    name = NAME or ("fig2_preview_offset" if PREVIEW else ("fig2_benchmark_linear" if LINEAR else "fig2_benchmark"))
    for ext in ("png", "pdf"):
        fig.savefig(IMAGES / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    print("wrote", IMAGES / f"{name}.png")


if __name__ == "__main__":
    main()
