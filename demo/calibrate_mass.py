"""Calibrate the mass estimate for a material: fit a and b of m = a * n_px ** b from weighed scoops (paper: Methods,
Vision-Based Mass Estimation and Statistical Analysis; Table 4). No images are needed, only one row per scoop with the
pixel count of its top-view mask after the scoop and the mass the balance measured.

    python demo/calibrate_mass.py                            # the paper's three materials (demo/data/calibration.csv)
    python demo/calibrate_mass.py my_scoops.csv              # your own material
    python demo/calibrate_mass.py my_scoops.csv --plot       # also draw the fit to demo/out/calibration.png
    python demo/calibrate_mass.py my_scoops.csv --out my_material.json   # save a and b

The CSV needs the columns pixel_count and mass_mg (mg), and may have material (one fit per material) and trial.
Use the scoops that were not empty and returned a mask: as in the paper, a scoop the balance cannot tell from an empty
spatula, or a scoop with an empty mask (a non-detection), is left out of the fit. Rows with a zero pixel count or a
mass of zero or less cannot enter a fit in log-log coordinates; they are counted and skipped.

The fit is least squares in log-log coordinates, and R^2 is computed in the same coordinates. The errors are
leave-one-out: each scoop is predicted by a fit that did not see it, as for every vision error in the paper. On
demo/data/calibration.csv the script prints the numbers of Table 4.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from demo_data import DATA, LABEL, OUT, calibration_scoops
from ssvision.mass import power_law_fit


def plot(fits: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.2, 3.4), constrained_layout=True)
    for name, f in fits.items():
        pts = ax.scatter(f["x"], f["y"], s=12, alpha=0.8, label=f"{name}: b = {f['b']:.2f}, R$^2$ = {f['r2']:.3f}")
        xx = np.logspace(np.log10(f["x"].min()), np.log10(f["x"].max()), 100)
        ax.plot(xx, f["a"] * xx ** f["b"], color=pts.get_facecolor()[0], lw=1.0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Mask pixel count n_px"); ax.set_ylabel("Balance mass (mg)")
    ax.legend(fontsize=7)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", nargs="?", default=str(DATA / "calibration.csv"),
                    help="one row per scoop: pixel_count, mass_mg (and optionally material, trial)")
    ap.add_argument("--plot", action="store_true", help="write demo/out/calibration.png")
    ap.add_argument("--out", help="write a and b per material to this JSON file")
    args = ap.parse_args()
    fits = {}
    print(f"m = a n_px^b, fitted by least squares in log-log coordinates; leave-one-out errors ({Path(args.csv).name})")
    for material, s in calibration_scoops(Path(args.csv)).items():
        name = LABEL.get(material, material)
        x, y = s["pixel_count"], s["mass_mg"]
        use = (x > 0) & (y > 0)
        if use.sum() < 3:
            print(f"{name}: only {int(use.sum())} usable scoops, at least 3 are needed")
            continue
        x, y = x[use], y[use]
        a, b, r2, _, loo, _ = power_law_fit(x, y)
        rel = 100 * loo / y
        fits[name] = {"a": a, "b": b, "r2": r2, "n": int(use.sum()), "x": x, "y": y}
        skipped = f"; {int((~use).sum())} rows skipped (zero pixels or mass)" if (~use).any() else ""
        print(f"{name:<8} n = {len(x)}, {y.min():.2f}-{y.max():.1f} mg{skipped}")
        print(f"  a = {a:.4e}, b = {b:.4f}, R^2 = {r2:.3f}")
        print(f"  leave-one-out error: median {np.median(loo):.2f} mg, mean {loo.mean():.2f} mg, max {loo.max():.1f} mg; "
              f"relative median {np.median(rel):.0f} %, 90th percentile {np.percentile(rel, 90):.0f} %")
    if args.out:
        Path(args.out).write_text(json.dumps({k: {"a": f["a"], "b": f["b"], "r2": f["r2"], "n": f["n"]}
                                              for k, f in fits.items()}, indent=1) + "\n")
        print(f"wrote {args.out}")
    if args.plot and fits:
        plot(fits, OUT / "calibration.png")
        print("wrote demo/out/calibration.png")
    return 0 if fits else 1


if __name__ == "__main__":
    sys.exit(main())
