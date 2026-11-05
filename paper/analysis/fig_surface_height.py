"""Surface figure, single panel (fig:surface_height): vision against force-probe surface height, NaCl.
Data: paper/data/surface_compare_2026-09-15-11-22.csv.

One point per block = the mean of its three vision reads, against the probe of that block.
Default: both methods converted from depth below the lip (surface_z_mm, the column surface_compare.py compares
in) to height above the floor with the same VIAL_FLOOR_BELOW_LIP_MM. Writes ../output/fig_surface_height.*
--raw: the CSV's own height_above_vial_base_mm column, as in the check plot; the two methods use different
lip-to-base distances there (56.70 vs 54.00 mm). Writes ../output/fig_surface_height_raw.*

Run: python3 fig_surface_height.py [--raw]
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 8  # single-column figure, included at \linewidth
import numpy as np

from common import IMAGES, VIAL_FLOOR_BELOW_LIP_MM
from surface_compare import load

RAW = "--raw" in sys.argv
# Block markers avoid the material shapes (o, s, ^) and colours of fig2/fig_deposit: all data here are NaCl.
BLOCK_STYLE = {B: dict(marker=m, s=sz, color=c, edgecolors="0.1", linewidths=0.4)  # sizes equalise the look
               for B, m, sz, c in ((1, "D", 22, "0.15"), (2, "P", 36, "0.5"), (3, "*", 60, "0.8"))}


def main() -> None:
    rows, vis, probe, t_vis, t_probe = load()
    keys = sorted(vis)
    blk = np.array([k[1] for k in keys])
    if RAW:
        p_raw = {(int(r["level"]), int(r["block"])): float(r["height_above_vial_base_mm"]) for r in rows if r["method"] == "probe"}
        hv = np.array([np.mean([x[2] for x in vis[k]]) for k in keys])
        hp = np.array([p_raw[k] for k in keys])
    else:
        hv = VIAL_FLOOR_BELOW_LIP_MM + np.array([np.mean([x[0] for x in vis[k]]) for k in keys])  # surface_z_mm < 0
        hp = VIAL_FLOOR_BELOW_LIP_MM + np.array([probe[k] for k in keys])
    diff = hv - hp
    print(f"{'raw height columns' if RAW else f'both from surface_z_mm, floor {VIAL_FLOOR_BELOW_LIP_MM} mm below the lip'}: "
          f"vision - probe mean {diff.mean():+.2f} mm (SD {diff.std(ddof=1):.2f}, range {diff.min():+.2f} to {diff.max():+.2f}), n = {len(keys)}")

    fig, ax = plt.subplots(figsize=(3.4, 2.9), constrained_layout=True)
    lim = (5, 35)
    ax.plot(lim, lim, color="k", lw=0.8, ls=":", label="identity")
    ax.plot(lim, np.array(lim) + diff.mean(), color="0.6", lw=1.0, label=f"identity {diff.mean():+.1f} mm")
    for B, st in BLOCK_STYLE.items():
        m = blk == B
        ax.scatter(hp[m], hv[m], **st, label=f"block {B}", zorder=3)
    ax.set_xlim(lim); ax.set_ylim(lim)  # same size as fig_deposit, so not square
    ref = "base" if RAW else "the floor"
    ax.set_xlabel(f"Force probe, height above {ref} (mm)")
    ax.set_ylabel(f"Vision, height above {ref} (mm)")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.grid(True, lw=0.3, alpha=0.5)
    name = "fig_surface_height_raw" if RAW else "fig_surface_height"
    IMAGES.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(IMAGES / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    print("wrote", IMAGES / f"{name}.png")


if __name__ == "__main__":
    main()
