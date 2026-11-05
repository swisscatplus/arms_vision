"""Mass estimate from one top-view image: the material on the spatula -> mask pixel count -> mass (paper: Methods,
Vision-Based Mass Estimation, Eq. 1 and Table 4; Figure 5, bottom row).

From the stored SAM 3 mask of a trial (no SAM 3 needed):

    python demo/estimate_mass.py                       # the three demo trials
    python demo/estimate_mass.py --material sugar      # one of them (salt, sugar, polenta)
    python demo/estimate_mass.py --plot                # also draw the mask to demo/out/
    python demo/estimate_mass.py --sample salt_t23     # another trial, after scripts/make_demo_data.py --trial salt:23

From the image, with SAM 3 installed (optional, see the README): the mask is first recomputed on the full frame with
the run's two-stage text prompt, processed as the service of the runs did (ssvision.sam3_pipeline), then counted the
same way. Needs the full frames (scripts/download_data.py demo).

    python demo/estimate_mass.py --sam3 [--device cpu|mps|cuda] [--sam3-model facebook/sam3]

Pipeline:
  1. mask of the material on the spatula, from the image taken at the photograph pose after the scoop. SAM 3 ran in
     two stages: the spatula with the prompt "scoop", then the material, with the material's prompt (Table 1), on the
     crop to the box of the first stage's mask. Both stages at confidence 0.4; each kept every detection above it
     (docs/segmentation_service.md).
  2. n_px = the number of mask pixels in the full 1936 x 1216 frame (the crop holds the whole mask).
  3. m = a * n_px ** b (Eq. 1) with the material's a and b (Table 4), fitted here by ssvision.mass.power_law_fit from
     demo/data/calibration.csv: the pixel counts and balance masses of the scoops the paper's fit uses (least squares
     in log-log, over the scoops that were not empty and returned a mask). calibrate_mass.py prints the same fit.

The script also gives the leave-one-out estimate, from a and b fitted without this trial, which is how every vision
error in the paper is computed, and the mass the balance measured.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from demo_data import LABEL, OUT, TRIALS, calibration_scoops, load_trial
from ssvision.mass import fit_power_law, mass_from_pixels, pixel_count, power_law_fit


def power_law(material: str, trial: int) -> dict:
    """a, b of Eq. 1 for the material (Table 4), and the leave-one-out a, b without `trial`."""
    s = calibration_scoops()[material]
    x, y = s["pixel_count"], s["mass_mg"]
    a, b, r2, *_ = power_law_fit(x, y)
    out = {"a": a, "b": b, "r2": r2, "n": len(x)}
    if trial in s["trial"]:
        keep = s["trial"] != trial
        out["a_loo"], out["b_loo"] = fit_power_law(x[keep], y[keep])
    return out


def plot(rgb, mask, rec, n_px, m_hat, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    over = rgb.astype(float)
    over[mask] = 0.55 * over[mask] + 0.45 * np.array([255, 60, 60])
    ys, xs = np.nonzero(mask) if mask.any() else (np.array([rgb.shape[0] // 2]), np.array([rgb.shape[1] // 2]))
    cy, cx = int(ys.mean()), int(xs.mean())
    fig, ax = plt.subplots(figsize=(6, 3.4), constrained_layout=True)
    ax.imshow(over.astype(np.uint8)[max(cy - 260, 0):cy + 260, max(cx - 420, 0):cx + 420])
    ax.axis("off")
    ax.set_title(f"{LABEL[rec['material']]} trial {rec['trial']}: {n_px} px -> {m_hat:.2f} mg "
                 f"(balance {rec['balance']['carried_mg']:.2f} mg)", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(stem: str, args) -> bool:
    t = load_trial(stem, full=args.sam3)
    rec = t["rec"]
    material = rec["material"]
    ms = rec["mass_stored"]
    stored_mask = t["top_mask"]
    if args.sam3:
        from ssvision.segment import top_view_mask
        mask, info = top_view_mask(t["top"], ms["prompts"], ms["confidence_thresholds"], ms["paddings_px"],
                                   args.sam3_model, args.device)
        union = int((mask | stored_mask).sum())
        source = info["note"] + (f"; IoU {int((mask & stored_mask).sum()) / union:.4f} with the stored mask" if union else "")
    else:
        mask, source = stored_mask, "stored SAM 3 mask"
    n_px = pixel_count(mask)
    pl = power_law(material, rec["trial"])
    m_hat = mass_from_pixels(n_px, pl["a"], pl["b"])
    carried = rec["balance"]["carried_mg"]
    print(f"\n{LABEL[material]}, trial {rec['trial']} (run {rec['run']['zarr']}, s_z = "
          f"{rec['commanded_offset_sz_mm']:+.1f} mm), {source}")
    print(f"  prompts           {ms['prompts'][0]!r}, then {ms['prompts'][1]!r}, confidence {ms['confidence_thresholds']}")
    print(f"  mask pixels n_px  {n_px} px          stored pixel count {ms['pixel_count']} px"
          + ("" if args.sam3 else f"   {'identical' if n_px == ms['pixel_count'] else 'DIFFERENT'}"))
    print(f"  power law (Eq. 1) m = a n_px^b, a = {pl['a']:.4e}, b = {pl['b']:.4f}, R^2 = {pl['r2']:.3f} "
          f"(Table 4; fitted over the {pl['n']} {LABEL[material]} scoops in data/calibration.csv)")
    print(f"  vision estimate   {m_hat:.2f} mg")
    if "a_loo" in pl and n_px > 0:
        m_loo = mass_from_pixels(n_px, pl["a_loo"], pl["b_loo"])
        print(f"  leave-one-out     {m_loo:.2f} mg (a = {pl['a_loo']:.4e}, b = {pl['b_loo']:.4f}, fitted without this trial)")
    print(f"  balance           {carried:.2f} mg carried (source vial before - after); "
          f"vision - balance {m_hat - carried:+.2f} mg")
    ok = n_px == ms["pixel_count"]
    if args.plot:
        out = OUT / f"{stem}_mass{'_sam3' if args.sam3 else ''}.png"
        plot(t["top"], mask, rec, n_px, m_hat, out)
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
    ap.add_argument("--plot", action="store_true", help="write demo/out/<trial>_mass.png")
    args = ap.parse_args()
    stems = [args.sample] if args.sample else ([TRIALS[args.material]] if args.material else list(TRIALS.values()))
    ok = [run(s, args) for s in stems]
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
