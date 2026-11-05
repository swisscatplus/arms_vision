"""Materials panel: spatula photographs and SAM 3 masks, nine materials.

Adapted from the nine-material panel of the companion paper (Fig. 5,
`figures/benchmarking_mats.pdf`), with soda-lime glass beads dropped and
polenta added.  Row 1 is the bench photograph of the loaded spatula, row 2
the SAM 3 overlay for the same photograph.  A box marks the three materials
this paper calibrates against the balance.

Source images are the bench set Jasper shot and segmented (2026-09-23):
one photograph and one `*_overlay_view.png` per material, plus `prompts.txt`
giving the prompt used for each.  These are NOT the camera-44 frames of the
autonomous runs and the prompts are NOT the run prompts of Table 1 (tab:prompts).

Writes ../output/fig_materials.{png,pdf}.

    python paper/analysis/fig_materials.py [SRC_DIR]

With --faded, row 2 shows the photograph at PHOTO_KEEP opacity over white,
with the mask drawn on top as a light fill and a vector outline in the
overlay's own pink, so the outline stays visible on the dark background and
the bright blade.  Writes ../output/fig_materials_faded.{png,pdf} instead.

    python paper/analysis/fig_materials.py [SRC_DIR] --faded
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
FADED = "--faded" in sys.argv[1:]
SRC = Path(ARGS[0] if ARGS
           else Path(__file__).resolve().parent.parent / "data" / "photos" / "vision_paper_segmented_materials")
OUT = Path(__file__).resolve().parent.parent / "output"

# --faded only: how row 2 is drawn.
PHOTO_KEEP = 0.5               # opacity of the photograph over white
MASK_RGB = (255, 64, 129)      # #FF4081, the outline colour in Jasper's overlays
FILL_ALPHA = 0.30              # opacity of the mask fill
LINE_W = 0.9                   # outline width in points

# (file stem, label).  Order follows the companion panel with glass beads
# removed and polenta inserted third, so the three calibrated materials are
# contiguous and can be boxed.
MATERIALS = [
    ("salt", "NaCl"),
    ("sugar", "Sugar"),
    ("polenta", "Polenta"),
    ("honey", "Honey"),
    ("grease", "Grease"),
    ("glucose", "Glucose\nsyrup"),
    ("coconut_wax", "Coconut\noil"),
    ("hobt", "HOBt"),
    ("kh2po4", "KH$_2$PO$_4$"),
]
N_BOXED = 3          # leading columns enclosed by the box
DIFF_THRESHOLD = 30  # per-pixel RGB distance that counts as overlay paint
CROP_ASPECT = 4 / 3   # height / width of each panel
MASK_FRACTION = 0.16  # target share of the panel area covered by the material
MASK_TOP = 0.16       # where the top of the material sits, as a share of panel height

# Per-material framing tweaks, applied on top of the area rule above.  Larger
# `zoom` shows more background; `dx` shifts the window across the blade.  These
# are framing only and were set by eye - the photographs were taken freehand on
# two different cameras, so no single rule frames all nine well.
TWEAK = {
    # stem: (zoom, dx in mask widths)
    "salt": (1.15, 0.0),
    "sugar": (1.15, 0.0),
    "polenta": (1.00, 0.0),
    "honey": (1.30, 0.0),
    "grease": (1.30, 0.0),
    "glucose": (0.90, 0.0),
    "coconut_wax": (0.80, 0.0),
    "hobt": (0.90, 0.0),
    "kh2po4": (0.85, 0.0),
}


def mask_from_overlay(photo: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """The painted region, recovered by differencing the overlay and the photo."""
    if photo.shape != overlay.shape:
        raise ValueError("photo and overlay differ in size")
    return np.abs(photo.astype(int) - overlay.astype(int)).sum(2) > DIFF_THRESHOLD


def crop_box(mask: np.ndarray, shape: tuple[int, int],
             zoom: float = 1.0, dx: float = 0.0) -> tuple[int, int, int, int]:
    """A panel window around the mask, tip up, in rotated image coordinates.

    Scale is set so the material covers a fixed share of the panel area.  Area
    rather than width, because the materials range from a strip lying in the
    blade to a droplet bulging past it, and the photographs were taken freehand
    on two cameras, so neither the mask width nor the image width is a usable
    reference.
    """
    ys, xs = np.where(mask)
    cx = (xs.min() + xs.max()) / 2 + dx * (xs.max() - xs.min())
    w = zoom * np.sqrt(mask.sum() / (MASK_FRACTION * CROP_ASPECT))
    w = float(np.clip(w, 60.0, min(shape[1], shape[0] / CROP_ASPECT)))
    h = w * CROP_ASPECT
    # Slide the window back inside the frame rather than clipping it, so every
    # panel comes out at the same aspect and no panel is silently stretched.
    x0 = int(round(np.clip(cx - w / 2, 0, max(shape[1] - w, 0))))
    y0 = int(round(np.clip(ys.min() - MASK_TOP * h, 0, max(shape[0] - h, 0))))
    return x0, x0 + int(round(w)), y0, y0 + int(round(h))


def panels(stem: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The photograph, its overlay and the mask, rotated tip-up and cropped identically."""
    photo = np.asarray(Image.open(SRC / f"{stem}.jpg").convert("RGB"))
    overlay = np.asarray(Image.open(SRC / f"{stem}_overlay_view.png").convert("RGB"))
    mask = mask_from_overlay(photo, overlay)
    if not mask.any():
        raise ValueError(f"{stem}: overlay paints no pixels")
    zoom, dx = TWEAK.get(stem, (1.0, 0.0))
    # The spatula lies horizontally with its tip to the right; one
    # counter-clockwise quarter turn stands it up with the tip at the top.
    photo, overlay, mask = (np.rot90(a) for a in (photo, overlay, mask))
    # One window, computed from the mask and applied to both images, so the
    # photograph and its overlay are always the same crop at the same zoom.
    x0, x1, y0, y1 = crop_box(mask, photo.shape[:2], zoom, dx)
    top, bottom = photo[y0:y1, x0:x1], overlay[y0:y1, x0:x1]
    if top.shape != bottom.shape:
        raise AssertionError(f"{stem}: rows cropped differently, {top.shape} vs {bottom.shape}")
    return top, bottom, mask[y0:y1, x0:x1]


def faded_mask_panel(ax, photo: np.ndarray, mask: np.ndarray) -> None:
    """Row 2 for --faded: the photograph at PHOTO_KEEP over white, the mask as a
    light fill, and its outline as a vector contour on top.

    The mask is the region the overlay paints (mask_from_overlay), so it
    includes the overlay's own outline, a few source pixels wide; at panel size
    that is well under a printed pixel.
    """
    faded = PHOTO_KEEP * photo.astype(float) + (1.0 - PHOTO_KEEP) * 255.0
    ax.imshow(faded.astype(np.uint8))
    fill = np.zeros(mask.shape + (4,))
    fill[..., :3] = np.array(MASK_RGB) / 255.0
    fill[..., 3] = FILL_ALPHA * mask
    ax.imshow(fill)
    h, w = mask.shape
    ax.contour(mask.astype(float), levels=[0.5], colors=[tuple(np.array(MASK_RGB) / 255.0)],
               linewidths=LINE_W)
    ax.set_xlim(-0.5, w - 0.5)   # contour must not change the window imshow set
    ax.set_ylim(h - 0.5, -0.5)


def read_prompts() -> dict[str, str]:
    out: dict[str, str] = {}
    path = SRC / "prompts.txt"
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def main() -> None:
    n = len(MATERIALS)
    fig_w, fig_h = 7.2, 2.24
    fig = plt.figure(figsize=(fig_w, fig_h))

    left, right, gap = 0.052, 0.004, 0.0045
    col_w = (1.0 - left - right - (n - 1) * gap) / n
    row_h = (col_w * fig_w * CROP_ASPECT) / fig_h
    y_lower, row_gap = 0.012, 0.007
    y_upper = y_lower + row_h + row_gap
    top = y_upper + row_h

    for i, (stem, label) in enumerate(MATERIALS):
        photo, overlay, mask = panels(stem)
        x = left + i * (col_w + gap)
        for y, img in ((y_upper, photo), (y_lower, overlay)):
            ax = fig.add_axes((x, y, col_w, row_h))
            if FADED and y == y_lower:
                faded_mask_panel(ax, photo, mask)
            else:
                ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
                s.set_color("0.55")
        fig.text(x + col_w / 2, top + 0.018, label, ha="center", va="bottom",
                 fontsize=6.4, linespacing=1.05)

    fig.text(left - 0.010, y_upper + row_h / 2, "Photograph", rotation=90,
             ha="right", va="center", fontsize=6.4)
    fig.text(left - 0.010, y_lower + row_h / 2, "SAM 3 mask", rotation=90,
             ha="right", va="center", fontsize=6.4)

    # The box: the three free-flowing materials this paper investigates.
    pad = 0.006
    box_w = N_BOXED * col_w + (N_BOXED - 1) * gap + 2 * pad
    box = Rectangle((left - pad, y_lower - pad), box_w,
                    top - y_lower + 2 * pad,
                    transform=fig.transFigure, facecolor="none",
                    edgecolor="#c0392b", linewidth=2.0, zorder=5)
    fig.add_artist(box)

    OUT.mkdir(exist_ok=True)
    stem_out = "fig_materials_faded" if FADED else "fig_materials"
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem_out}.{ext}", dpi=300)
    print("wrote", OUT / f"{stem_out}.png")

    prompts = read_prompts()
    if prompts:
        print("\nprompt used for each panel (bench set, NOT the run prompts):")
        for stem, label in MATERIALS:
            print(f"  {label.replace(chr(10), ' '):16s} {prompts.get(stem, '?')}")


if __name__ == "__main__":
    main()
