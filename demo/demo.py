"""The one-minute demo: the surface estimate and the mass estimate on one stored trial per material.

    python demo/demo.py

For each demo trial (sodium chloride, sugar and polenta; the ~10 mg scoop of Figure 5 of the paper) it takes the
stored SAM 3 masks of the two camera images and prints:
  - the surface: the depth of the material surface below the vial lip and its tilt, from the side-view mask
    (ssvision.surface)
  - the mass: the pixel count of the top-view mask after the scoop, and the mass from the material's power law
    (ssvision.mass, a and b fitted from demo/data/calibration.csv), next to the balance reading
and draws both to demo/out/demo.png. estimate_surface.py and estimate_mass.py show each step and compare the results
with the values the robot stored; calibrate_mass.py fits the power law.
"""
from __future__ import annotations

import sys

import numpy as np

from demo_data import LABEL, OUT, TRIALS, VIAL_FLOOR_BELOW_LIP_MM, calibration_scoops, load_trial
from ssvision.mass import mass_from_pixels, pixel_count, power_law_fit
from ssvision.surface import estimate_surface


def overlay(rgb, mask, colour, alpha):
    out = rgb.astype(float)
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(colour)
    return out.astype(np.uint8)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scoops = calibration_scoops()
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.0), constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.45, 1.0]})
    for k, (material, stem) in enumerate(TRIALS.items()):
        t = load_trial(stem)
        rec, st = t["rec"], t["rec"]["surface_stored"]
        o = np.asarray(t["side_mask_origin"], float)
        surf = estimate_surface(t["side_mask"], st["vial_clicks_image_px"], st["vial_length_mm"], st["ransac_tol_mm"],
                                origin=t["side_mask_origin"])
        s = scoops[material]
        a, b, *_ = power_law_fit(s["pixel_count"], s["mass_mg"])
        n_px = pixel_count(t["top_mask"])
        m_hat = mass_from_pixels(n_px, a, b)
        carried = rec["balance"]["carried_mg"]
        print(f"{LABEL[material]}, trial {rec['trial']} (s_z = {rec['commanded_offset_sz_mm']:+.1f} mm)")
        print(f"  surface  {-surf['depth_mm']:.2f} mm below the lip ({VIAL_FLOOR_BELOW_LIP_MM + surf['depth_mm']:.2f} mm "
              f"above the vial floor), tilt {surf['tilt_deg']:.2f} deg")
        print(f"  mass     {n_px} px -> {m_hat:.2f} mg (a = {a:.4e}, b = {b:.4f}); the balance measured {carried:.2f} mg")

        ax = axes[0, k]
        ax.imshow(overlay(t["side"], t["side_mask"], (60, 120, 255), 0.4))
        c, v = surf["ends"].mean(axis=0) - o, np.diff(surf["ends"], axis=0)[0]
        v = v / np.linalg.norm(v)
        ax.plot(c[0] + np.array([-2000, 2000]) * v[0], c[1] + np.array([-2000, 2000]) * v[1], color="#ff3030", lw=1.4)
        bnd, inl = surf["boundary"] - o, surf["inliers"]
        ax.scatter(*bnd[inl].T, s=4, color="#ff3030")
        ax.scatter(*bnd[~inl].T, s=4, color="white")
        vc = np.asarray(st["vial_clicks_image_px"], float) - o
        ax.plot(vc[:, 0], vc[:, 1], color="white", ls="--", lw=0.9)
        ax.scatter(*(surf["p_img"] - o), marker="o", s=36, facecolor="none", edgecolor="yellow", lw=1.4)
        ax.set_xlim(0, t["side"].shape[1]); ax.set_ylim(t["side"].shape[0], 0); ax.axis("off")
        ax.set_title(f"{LABEL[material]}: surface {-surf['depth_mm']:.1f} mm below the lip,\n"
                     f"tilt {surf['tilt_deg']:.1f}$^\\circ$", fontsize=9)
        ax = axes[1, k]
        ax.imshow(overlay(t["top"], t["top_mask"], (255, 60, 60), 0.45))
        ax.axis("off")
        ax.set_title(f"{n_px} px $\\rightarrow$ {m_hat:.2f} mg (balance {carried:.2f} mg)", fontsize=9)
    OUT.mkdir(exist_ok=True)
    path = OUT / "demo.png"
    fig.savefig(path, dpi=110, metadata={"Software": None})
    plt.close(fig)
    print(f"wrote demo/out/{path.name}: top row, the side view with the mask (blue), its upper boundary (red: RANSAC "
          f"inliers), the fitted surface line, the vial axis (dashed) and the point p where they cross; bottom row, "
          f"the top view with the mask (red)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
