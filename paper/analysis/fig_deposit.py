"""Deposit figure (fig:deposit): mass lost between spatula and goal vial against carried mass, per material,
as measured, with no success criterion. Linear axes over the 0-ZOOM_MG range, with the full range in an inset
whose shaded band is the main view (the layout of fig2.py --linear). Writes ../output/fig_deposit_linear.png
(and .pdf), Figure 8 of the paper, or ../output/NAME.png with --name=NAME.
--log draws a variant instead: logarithmic carried mass and symlog loss, ../output/fig_deposit_loss.*.

Same trials as the deposit block of table1_depth_mass.py: the scoops that were not empty
(common.loaded_scoops: surface-estimation failures and empty scoops excluded).

Run: python3 fig_deposit.py [--log] [--name=NAME]   (after extract_runs.py)
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
plt.rcParams["font.size"] = 8  # single-column figure, included at \linewidth

from common import IMAGES, LABEL, load, loaded_scoops

COLORS = {"salt": "#1f77b4", "sugar": "#d62728", "polenta": "#ff9f1c"}  # same as fig2.py
MARK = {"salt": "o", "sugar": "s", "polenta": "^"}
LOG = "--log" in sys.argv  # the variant on logarithmic axes, ../output/fig_deposit_loss.*
ZOOM_MG = 15.5      # main view: carried mass up to this, covering the 0.5-15 mg range the method is meant for
LINTHRESH_MG = 0.1  # --log, symlog: linear between -0.1 and +0.1 mg
X_MIN_MG = 0.2      # --log: left edge of the carried-mass axis, below the smallest scoop that was not empty
NAME = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--name=")),
            "fig_deposit_loss" if LOG else "fig_deposit_linear")


def scoops():
    """Per material: carried mass and mass lost, over the scoops that were not empty."""
    for mat in ["salt", "sugar", "polenta"]:
        d = load(mat)
        loaded = loaded_scoops(d)
        c, lost = d["carried_mg"][loaded], d["lost_mg"][loaded]
        print(f"{LABEL[mat]}: {len(c)} scoops that were not empty; lost {lost.min():+.2f} to {lost.max():.2f} mg")
        yield mat, c, lost


def draw_linear(ax) -> None:
    ins = ax.inset_axes([0.07, 0.55, 0.42, 0.40])
    for mat, c, lost in scoops():
        ax.scatter(c, lost, s=14, color=COLORS[mat], marker=MARK[mat], alpha=0.8, label=f"{LABEL[mat]}: n={len(c)}")
        ins.scatter(c, lost, s=4, color=COLORS[mat], marker=MARK[mat], alpha=0.8)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlim(0, ZOOM_MG); ax.set_ylim(-0.25, 3.5)
    ax.set_xlabel("Carried mass (mg)")
    ax.set_ylabel("Mass lost in the deposit (mg)")
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.9)
    ins.axhline(0, color="gray", lw=0.4)
    ins.axvspan(0, ZOOM_MG, color="k", alpha=0.06)
    ins.set_xlim(0, 95); ins.set_ylim(-0.5, 11)
    ins.tick_params(labelsize=5.5, length=2)
    ins.grid(True, lw=0.3, alpha=0.5)
    ins.set_title("full range", fontsize=6, pad=2)


def draw_log(ax) -> None:
    ax.axhline(0, color="gray", lw=0.5)
    for mat, c, lost in scoops():
        ax.scatter(c, lost, s=14, color=COLORS[mat], marker=MARK[mat], alpha=0.8, label=f"{LABEL[mat]}: n={len(c)}")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=LINTHRESH_MG, linscale=0.6)
    ax.set_xlim(X_MIN_MG, 150); ax.set_ylim(-0.2, 30)
    ax.set_xlabel("Carried mass (mg)")
    ax.set_ylabel("Mass lost in the deposit (mg)")
    plain = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    ax.xaxis.set_major_formatter(plain); ax.yaxis.set_major_formatter(plain)
    ax.grid(True, which="major", lw=0.3, alpha=0.5)
    ax.legend(fontsize=6.5, loc="upper left", framealpha=0.9)


def main() -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.9), constrained_layout=True)
    (draw_log if LOG else draw_linear)(ax)
    IMAGES.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(IMAGES / f"{NAME}.{ext}", dpi=300, bbox_inches="tight")
    print("wrote", IMAGES / f"{NAME}.png")


if __name__ == "__main__":
    main()
