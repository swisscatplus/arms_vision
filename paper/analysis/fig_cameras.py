"""Figure C (fig:cameras): the workbench, the two cameras, and what each camera sees.

Layout after the companion paper's workbench figure (arms_components.pdf): a central photo of
the bench with lettered orange boxes, and inset panels around it.

    A  both cameras on their mounts, source vial at the presentation pose, spatula   (photo)
    B  side camera, raw frame of the source vial before the scoop                     (run zarr)
    C  top camera, raw frame of the loaded spatula after the scoop                    (run zarr, read 2)
    D  balance with the cover                                                         (photo)
    E  sandblasted spatula with the force-torque sensor on the FR3                    (photo)
    F  vial storage                                                                   (photo)

B and C are full camera frames as logged (the driver's rectified stream, image_rect), changed
here only by the BGR->RGB swap and a uniform resampling to the print resolution; no mask is
drawn (Figure fig:examples shows the masks).  Both come from the same trial, chosen by rule
(TRIAL_RULE below), and the script prints the trial, its offset and its carried mass for the
caption.

Photos are inputs.  To drop in a new photo, set its path in PHOTOS, set its crop (pixel box in
that photo, x0, y0, x1, y1; the crop is widened to the panel's aspect around its centre),
remove `placeholder`, and re-measure any LABELS / BENCH_BOXES given in that photo's pixels.
An RGBA photo (background removed) is composited on white.  A PHOTOS entry with path=None is
drawn as a labelled placeholder box.  The printout gives each photo panel's printed size and
the nominal resolution of its crop (crop pixels / printed inches).

Writes ../output/fig_cameras.{png,pdf}.  Nothing is written into any input file, and the
outputs carry no EXIF/GPS (photos are read into arrays; savefig metadata is cleared).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import sys

import numpy as np
import zarr
from matplotlib.patches import Rectangle
from PIL import Image

from common import IMAGES, LABEL, PHOTOS, load
from extract_runs import RUNS

SRC = PHOTOS

# ---- inputs: photos ---------------------------------------------------------------------
# crop = (x0, y0, x1, y1) in pixels of the source photo; widened to the panel aspect.
PHOTOS = {
    # central photo: the companion paper's bench photo with the background removed
    # (companion figures/workbench_nobg.png).  The top camera
    # hangs on the vertical post beside a white panel, not on the horizontal arm of the
    # 2026-10-02 photos (A); the vial in storage is in a white holder; the cover is open.
    "bench": dict(path=SRC / "workbench_nobg.png", crop=(0, 0, 4080, 2960),
                  placeholder="placeholder: earlier\ntop-camera mount,\ncover open (shot 1)",
                  note_at="top"),
    # A: James's phone photo of 2026-10-02 (1 of 3; the sharpest, and the only one that also
    # shows the top camera's vertical post).  Both cameras, the presented source vial, spatula.
    "A": dict(path=SRC / "2026-10-02_cameras_1.jpg", crop=(0, 567, 2268, 3087)),
    # D: crop of the bench photo.  Cover beside the balance, NOT in place.  Shot 4.
    "D": dict(path=SRC / "workbench_nobg.png", crop=(1840, 2030, 3380, 2940),
              placeholder="placeholder (cover open), shot 4"),
    # E: James's phone photo of 2026-10-02 (3 of 3), lower part: spatula, F/T sensor, markers.
    "E": dict(path=SRC / "2026-10-02_cameras_3.jpg", crop=(0, 2350, 1915, 3459)),
    # F: crop of the bench photo; the vial sits in the white holder, not the run's black one.
    "F": dict(path=SRC / "workbench_nobg.png", crop=(3500, 2370, 4080, 2820),
              placeholder="placeholder (white holder), shot 6", note_at="bottom-left"),
}

# Labels inside panel A, in pixels of its photo: text, text centre (x, y), arrow tip (x, y) or None.
# The arrow starts at the edge of the text.
A_LABELS = [
    ("top camera", (1000, 1745), (1528, 1720)),
    ("side camera", (830, 2085), None),
    ("source vial", (1545, 2330), (1545, 2480)),
    ("spatula", (1540, 2930), (1215, 2770)),
]

# Boxes on the bench photo, in its pixels (x0, y0, x1, y1, letter side), and plain labels.
# letter side: "above" puts the letter over the top-left corner, "left" beside it.
BENCH_BOXES = {
    "A": (1740, 40, 2160, 1290, "left"),     # top camera on its post; side camera (partly hidden
                                             # behind the vial) and the presented vial
    "D": (1960, 2040, 3260, 2935, "above"),  # balance and cover
    "E": (1250, 1000, 1700, 1510, "above"),  # F/T sensor, flange adapter, spatula
    "F": (3750, 2380, 4030, 2810, "above"),  # vial storage
}
BENCH_LABELS = {"FR3": (430, 1500), "UR3e": (3520, 1250)}

# ---- inputs: camera frames ---------------------------------------------------------------
# B and C come from one trial: the sodium chloride trial at s_z = +3 mm with the median
# carried mass (an offset inside the 0.5-15 mg range the paper targets).
TRIAL_RULE = dict(material="salt", sz_mm=3.0)
READ_AFTER_SCOOP = 2  # spatula read index: 0 start, 1 before scoop, 2 after scoop, 3, 4 (README.md, Data)

# ---- style --------------------------------------------------------------------------------
ORANGE = "#C73F00"           # box orange of the companion figure (sampled from arms_components.pdf)
PLACEHOLDER_FC = "#EDEDED"
PLACEHOLDER_EC = "#8A8A8A"
FS = 7.0                     # smallest text, pt, at print size
FS_LETTER = 8.5
DPI = 600                    # print resolution of the PNG and of every embedded photo

# ---- geometry (mm) -------------------------------------------------------------------------
TEXTWIDTH_MM = 178.0         # USG.cls: 210 mm paper, 16 mm margins (measured: 506.459 pt)
W = 0.92 * TEXTWIDTH_MM      # figure* at 0.92\textwidth, as the other full-width figures
GAP = 1.6
FRAME_ASPECT = 1936 / 1216   # camera frame; checked against the zarr below
A_ASPECT = 0.90              # panel A (portrait photo, full width of the phone frame)
BENCH_ASPECT = 1.38          # central photo
BOTTOM_H = 31.0              # bottom row D, E, F, equal widths


def layout():
    """Return {panel: (x, y, w, h)} in mm, y from the top, and the figure height.

    Top block: A | bench | B over C, all of one height h = 2 camera frames + gap."""
    r = A_ASPECT + BENCH_ASPECT
    s = (W - 2 * GAP - r * GAP) / (1 + r * 2 / FRAME_ASPECT)
    fh = s / FRAME_ASPECT
    h = 2 * fh + GAP
    a, c = A_ASPECT * h, BENCH_ASPECT * h
    P = {
        "A": (0, 0, a, h),
        "bench": (a + GAP, 0, c, h),
        "B": (a + GAP + c + GAP, 0, s, fh),
        "C": (a + GAP + c + GAP, fh + GAP, s, fh),
    }
    y = h + GAP
    w = (W - 2 * GAP) / 3
    for i, k in enumerate("DEF"):
        P[k] = (i * (w + GAP), y, w, BOTTOM_H)
    return P, y + BOTTOM_H


# ---- helpers --------------------------------------------------------------------------------
def fit_crop(size, box, aspect):
    """Widen box (x0, y0, x1, y1) about its centre to width/height = aspect, inside size."""
    W_, H_ = size
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    w, h = min(w, W_), min(h, H_)
    if w / h > aspect:
        w = h * aspect
    else:
        h = w / aspect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0 = min(max(cx - w / 2, 0), W_ - w)
    y0 = min(max(cy - h / 2, 0), H_ - h)
    return (int(round(x0)), int(round(y0)), int(round(x0 + w)), int(round(y0 + h)))


def open_rgb(path):
    """Open a photo as RGB; an RGBA photo (background removed) is composited on white."""
    im = Image.open(path)
    im.load()
    if im.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.getchannel("A"))
        return bg
    return im.convert("RGB")


def to_print(arr_or_img, w_mm, h_mm):
    """Uniform resample to the panel's pixel size at DPI (no other change)."""
    im = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    px = (int(round(w_mm / 25.4 * DPI)), int(round(h_mm / 25.4 * DPI)))
    return np.asarray(im.convert("RGB").resize(px, Image.LANCZOS)), px


def add_axes(fig, rect, H):
    x, y, w, h = rect
    ax = fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def tag(ax, letter, title=None):
    """Orange letter in a white box at the top-left corner, then a short title."""
    t = ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(2.2, -2.2),
                    textcoords="offset points", ha="left", va="top", fontsize=FS_LETTER,
                    fontweight="bold", color=ORANGE,
                    bbox=dict(boxstyle="square,pad=0.22", fc="white", ec="none"))
    if title:
        # pads chosen so the two white boxes have the same height and touch
        ax.annotate(title, xy=(1, 0.5), xycoords=t, xytext=(0.22 * FS_LETTER + 0.39 * FS, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=FS, color="black",
                    bbox=dict(boxstyle="square,pad=0.39", fc="white", ec="none"))


def placeholder_note(ax, text, where="bottom"):
    """Grey italic tag marking a photo that is to be replaced."""
    xy, off, ha, va = {"top": ((1, 1), (-2, -2), "right", "top"),
                       "bottom": ((1, 0), (-2, 2), "right", "bottom"),
                       "bottom-left": ((0, 0), (2, 2), "left", "bottom"),
                       "right": ((1, 0.5), (-2, 0), "right", "center")}[where]
    ax.annotate(text, xy=xy, xycoords="axes fraction", xytext=off, textcoords="offset points",
                ha=ha, va=va, ma=ha, fontsize=FS, color="#444444", style="italic",
                bbox=dict(boxstyle="square,pad=0.2", fc="white", ec=PLACEHOLDER_EC, lw=0.5, alpha=0.9))


def label(ax, x, y, s, dark=False, **kw):
    """Annotation text: white with a black halo on dark ground, black with a white halo on light."""
    kw.setdefault("fontsize", FS)
    fg, halo = ("black", "white") if dark else ("white", "black")
    return ax.text(x, y, s, color=fg, path_effects=[pe.withStroke(linewidth=1.2 if not dark else 1.6,
                                                                  foreground=halo)], **kw)


def callout(ax, text, xy_text, xy_tip=None, dark=False, **kw):
    """Label with an optional arrow; the arrow starts at the edge of the text (invisible bbox)."""
    kw.setdefault("fontsize", FS)
    kw.setdefault("ha", "center"); kw.setdefault("va", "center")
    fg, halo = ("black", "white") if dark else ("white", "black")
    props = None
    if xy_tip is not None:
        props = dict(arrowstyle="-|>", color=fg, lw=0.8, mutation_scale=7, shrinkA=1.5, shrinkB=0,
                     path_effects=[pe.withStroke(linewidth=1.6, foreground=halo)])
    ax.annotate(text, xy=xy_tip if xy_tip is not None else xy_text, xytext=xy_text, color=fg,
                arrowprops=props, bbox=dict(boxstyle="square,pad=0.15", fc="none", ec="none"),
                path_effects=[pe.withStroke(linewidth=1.6 if dark else 1.2, foreground=halo)], **kw)


def photo_panel(ax, spec, rect):
    _, _, w, h = rect
    im = open_rgb(spec["path"])
    box = fit_crop(im.size, spec["crop"], w / h)
    arr, _ = to_print(im.crop(box), w, h)
    ax.imshow(arr, extent=(box[0], box[2], box[3], box[1]), interpolation="none")
    ax.set_xlim(box[0], box[2]); ax.set_ylim(box[3], box[1])
    if spec.get("placeholder"):
        placeholder_note(ax, spec["placeholder"], spec.get("note_at", "bottom"))
    return box


def empty_panel(ax, spec):
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc=PLACEHOLDER_FC,
                           ec=PLACEHOLDER_EC, lw=0.8, ls=(0, (4, 2.5))))
    ax.text(0.5, 0.44, spec["text"], transform=ax.transAxes, ha="center", va="center",
            fontsize=FS, color="#444444", linespacing=1.25)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


def panel(ax, key, rect):
    spec = PHOTOS[key]
    if spec.get("path") is None:
        empty_panel(ax, spec)
        return None
    return photo_panel(ax, spec, rect)


# ---- camera frames --------------------------------------------------------------------------
def pick_trial():
    mat, sz = TRIAL_RULE["material"], TRIAL_RULE["sz_mm"]
    d = load(mat)
    ok = (d["sz_mm"] == sz) & ~d["surface_fail"] & (d["px_obs2"] > 0)
    idx = np.where(ok)[0]
    order = idx[np.argsort(d["carried_mg"][idx])]
    k = order[len(order) // 2]
    return mat, int(d["iteration"][k]), float(d["sz_mm"][k]), float(d["carried_mg"][k]), int(d["px_obs2"][k])


def camera_frames(mat, trial):
    g = zarr.open_group(str(RUNS[mat]), mode="r")
    su = g["events"]["surface_observer"]
    i = int(np.where(np.array(su["iteration"]) == trial)[0][0])
    side = np.array(su["image"][i])[:, :, ::-1]                  # stored BGR
    centre = np.array(su["plane_center_image"][i], dtype=float).ravel()[:2]
    sp = g["events"]["spatula_observer"]
    it, oi = np.array(sp["iteration"]), np.array(sp["observation_index"])
    j = int(np.where((it == trial) & (oi == READ_AFTER_SCOOP))[0][0])
    top = np.array(sp["image_camera44"][j])                     # stored RGB (side frames are BGR)
    mask = np.array(sp["mask_camera44"][j]) > 0                   # used only to place a label
    px = int(np.array(sp["pixel_count"][j]))
    return side, centre, top, mask, px


def frame_panel(ax, frame, rect):
    _, _, w, h = rect
    H_, W_ = frame.shape[:2]
    arr, _ = to_print(frame, w, h)
    ax.imshow(arr, extent=(0, W_, H_, 0), interpolation="none")
    ax.set_xlim(0, W_); ax.set_ylim(H_, 0)
    for s in ax.spines.values():          # thin frame edge: the panel is the full field of view
        s.set_visible(True); s.set_linewidth(0.4); s.set_color("#555555")
    label(ax, W_ - 25, H_ - 25, f"full frame, {W_} $\\times$ {H_} px", ha="right", va="bottom")


# ---- figure ---------------------------------------------------------------------------------
def main() -> None:
    P, H = layout()
    mat, trial, sz, carried, px_csv = pick_trial()
    side, centre, top, mask, px = camera_frames(mat, trial)
    assert side.shape[:2] == top.shape[:2] == (1216, 1936), side.shape
    assert abs(side.shape[1] / side.shape[0] - FRAME_ASPECT) < 1e-9
    assert px == px_csv, (px, px_csv)

    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    plt.rcParams["font.size"] = FS
    crops = {}

    # central bench photo with boxes
    ax = add_axes(fig, P["bench"], H)
    crops["bench"] = panel(ax, "bench", P["bench"])
    for letter, (x0, y0, x1, y1, anchor) in BENCH_BOXES.items():
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=ORANGE, lw=1.3))
        xy, ha, va = ((x0, y0), "left", "bottom") if anchor == "above" else ((x0, y0), "right", "top")
        ax.annotate(letter, xy=xy, xytext=(0 if anchor == "above" else -1.5, 1.2 if anchor == "above" else 0),
                    textcoords="offset points", ha=ha, va=va, fontsize=FS_LETTER, fontweight="bold",
                    color=ORANGE, bbox=dict(boxstyle="square,pad=0.18", fc="white", ec="none"))
    for name, (x, y) in BENCH_LABELS.items():
        label(ax, x, y, name, dark=True, ha="center", va="center", fontsize=FS + 1, fontweight="bold")

    # A: both cameras and the presented vial
    ax = add_axes(fig, P["A"], H)
    crops["A"] = panel(ax, "A", P["A"])
    tag(ax, "A", "Cameras and presented vial")
    for text, xy, tip in A_LABELS:
        callout(ax, text, xy, tip, dark=True)

    # B: side camera raw frame (before the scoop)
    ax = add_axes(fig, P["B"], H)
    frame_panel(ax, side, P["B"])
    tag(ax, "B", "Side-camera image")
    callout(ax, "material surface", (1900, 880), (centre[0] + 20, centre[1] + 30), ha="right")

    # C: top camera raw frame, after the scoop
    ax = add_axes(fig, P["C"], H)
    frame_panel(ax, top, P["C"])
    tag(ax, "C", "Top-camera image")
    ys, xs = np.where(mask)
    mx, my = float(np.median(xs)), float(ys.max())
    callout(ax, "loaded spatula", (mx, my + 370), (mx, my + 40))
    callout(ax, "source vial", (1500, 300))

    # D, E, F
    for letter, title in (("D", "Balance and cover"), ("E", "Spatula and force–torque sensor"),
                          ("F", "Vial storage")):
        ax = add_axes(fig, P[letter], H)
        crops[letter] = panel(ax, letter, P[letter])
        tag(ax, letter, title)

    IMAGES.mkdir(exist_ok=True)
    fig.savefig(IMAGES / "fig_cameras.png", dpi=DPI, metadata={"Software": None})
    fig.savefig(IMAGES / "fig_cameras.pdf", dpi=DPI,
                metadata={"Creator": None, "Producer": None, "CreationDate": None})

    # ---- printout: caption numbers, crop resolution, minimum photo sizes for the shot list --
    print(f"figure: {W:.1f} x {H:.1f} mm ({W / 25.4:.3f} x {H / 25.4:.3f} in), PNG at {DPI} dpi")
    print(f"B, C: {LABEL[mat]} run {RUNS[mat].name}, trial {trial}, s_z = {sz:+.1f} mm, "
          f"carried {carried:.2f} mg (balance), read {READ_AFTER_SCOOP} mask {px} px; "
          f"frames {side.shape[1]} x {side.shape[0]} px")
    print("panel     printed (mm)   min px at 300 dpi   source crop (px)          nominal ppi of crop")
    for k in ("bench", "A", "B", "C", "D", "E", "F"):
        _, _, w, h = P[k]
        need = f"{w / 25.4 * 300:4.0f} x {h / 25.4 * 300:4.0f}"
        box = crops.get(k)
        if box is None:
            src = f"{side.shape[1]} x {side.shape[0]} (camera frame)" if k in "BC" else "none (placeholder)"
            cw = side.shape[1] if k in "BC" else None
        else:
            cw = box[2] - box[0]
            src = f"{cw} x {box[3] - box[1]} at {box[:2]}"
        ppi = f"{cw / (w / 25.4):5.0f}" if cw else "    -"
        print(f"  {k:6s} {w:5.1f} x {h:4.1f}   {need}         {src:28s} {ppi}")


# ---- compact variants ------------------------------------------------------------------------
# The cycle figure (fig_cycle.py) shows the weighing close up, so B, C, D and F are not needed here.
#   compact   : the bench photo and one photo of the cameras, the presented vial, the spatula and the
#               force-torque sensor (James's photo 1 of 2026-10-02, cropped down to the sensor).
#   compact_E : the same plus the spatula/sensor close-up (old panel E) as panel B, for comparison.
# Run:  $PY fig_cameras.py --variant compact | compact_E      (no argument: the full figure, unchanged)
COMPACT_A = dict(path=SRC / "2026-10-02_cameras_1_nobg.png", crop=(0, 950, 2268, 3550))   # bracket .. sensor base;
# background (the wall) removed with SAM 3 by tools/clean_photo_background_cameras.py, as for the bench photo
COMPACT_A_ASPECT = 2268 / 2600
COMPACT_BENCH = dict(path=SRC / "workbench_nobg.png", crop=(0, 0, 4080, 2960))        # no placeholder tag
COMPACT_E = dict(path=SRC / "2026-10-02_cameras_3.jpg", crop=(0, 2350, 1915, 3459))
COMPACT_E_ASPECT = 1.0
# text, text centre (x, y), arrow tip (x, y) or None, dark ground under the text?  (pixels of photo 1)
COMPACT_A_LABELS = [
    ("top camera", (1000, 1745), (1528, 1720), False),
    ("side camera", (830, 2085), None, False),
    ("source vial", (1545, 2330), (1545, 2480), False),
    ("spatula", (1540, 2930), (1215, 2770), False),
    ("force–torque\nsensor", (1800, 3360), (1095, 3140), True),   # two lines: stays on the dark mat in both widths
]
# Station labels on the bench photo, in its pixels: text, centre (x, y), dark ground?
COMPACT_BENCH_LABELS = [
    ("balance", (2320, 2140), True),          # on the mat above the balance
    ("cover", (2970, 2560), True),            # on the cover's dark face
    ("vial\nstorage", (3890, 2235), False),  # on the white ground above the stored vial
]
COMPACT_BOXES = {
    # compact: one box for everything panel A shows (cameras, presented vial, spatula, sensor)
    "compact": {"A": (1250, 40, 2160, 1510, "left")},
    # compact_E: the cameras and vial (A) and the sensor/spatula (B), as in the full figure
    "compact_E": {"A": (1740, 40, 2160, 1290, "left"), "B": (1250, 1000, 1700, 1510, "above")},
}


# ---- style of the compact figure ----------------------------------------------------------------
#   - annotation text sits on a grey, partly transparent box instead of carrying a stroke, so the text
#     reads without a halo drawn over the photo;
#   - every arrow is grey, which reads on the white background and on the black mat;
#   - the bench photo is on the left and the close-up A on the right.
# compact_E (a comparison variant) keeps the earlier style: stroked labels, A | bench | B.
BOX_FC, BOX_ALPHA = "#BFBFBF", 0.70   # label box: light grey at 70 % opacity, black text
ARROW_GREY = "#7F7F7F"


def _grey_box():
    return dict(boxstyle="round,pad=0.28,rounding_size=0.25", fc=BOX_FC, ec="none", alpha=BOX_ALPHA)


def label_boxed(ax, x, y, s, **kw):
    """Annotation text in black on a grey, partly transparent box (no stroke)."""
    kw.setdefault("fontsize", FS)
    return ax.text(x, y, s, color="black", bbox=_grey_box(), **kw)


def callout_boxed(ax, text, xy_text, xy_tip=None, **kw):
    """label_boxed with an optional grey arrow; the arrow starts at the edge of the box."""
    kw.setdefault("fontsize", FS)
    kw.setdefault("ha", "center"); kw.setdefault("va", "center")
    props = None
    if xy_tip is not None:
        props = dict(arrowstyle="-|>", color=ARROW_GREY, lw=1.0, mutation_scale=7.5, shrinkA=0.5, shrinkB=0)
    ax.annotate(text, xy=xy_tip if xy_tip is not None else xy_text, xytext=xy_text, color="black",
                arrowprops=props, bbox=_grey_box(), **kw)


def layout_compact(variant):
    """One row: bench | A for compact, A | bench | B for compact_E. Equal heights; the row fills W."""
    aspects = [("A", COMPACT_A_ASPECT), ("bench", BENCH_ASPECT)]
    if variant == "compact":
        aspects.reverse()
    if variant == "compact_E":
        aspects.append(("B", COMPACT_E_ASPECT))
    h = (W - GAP * (len(aspects) - 1)) / sum(a for _, a in aspects)
    P, x = {}, 0.0
    for k, a in aspects:
        P[k] = (x, 0.0, a * h, h)
        x += a * h + GAP
    return P, h


def main_compact(variant) -> None:
    P, H = layout_compact(variant)
    boxed = variant == "compact"          # grey label boxes and grey arrows
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    plt.rcParams["font.size"] = FS
    crops = {}
    ax = add_axes(fig, P["bench"], H)
    crops["bench"] = photo_panel(ax, COMPACT_BENCH, P["bench"])
    for letter, (x0, y0, x1, y1, anchor) in COMPACT_BOXES[variant].items():
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=ORANGE, lw=1.3))
        xy, ha, va = ((x0, y0), "left", "bottom") if anchor == "above" else ((x0, y0), "right", "top")
        ax.annotate(letter, xy=xy, xytext=(0 if anchor == "above" else -1.5, 1.2 if anchor == "above" else 0),
                    textcoords="offset points", ha=ha, va=va, fontsize=FS_LETTER, fontweight="bold",
                    color=ORANGE, bbox=dict(boxstyle="square,pad=0.18", fc="white", ec="none"))
    for name, (x, y) in BENCH_LABELS.items():
        if boxed:
            label_boxed(ax, x, y, name, ha="center", va="center", fontsize=FS + 1, fontweight="bold")
        else:
            label(ax, x, y, name, dark=True, ha="center", va="center", fontsize=FS + 1, fontweight="bold")
    for text, (x, y), dark_ground in COMPACT_BENCH_LABELS:
        if boxed:
            label_boxed(ax, x, y, text, ha="center", va="center", ma="center")
        else:
            label(ax, x, y, text, dark=not dark_ground, ha="center", va="center", ma="center")
    ax = add_axes(fig, P["A"], H)
    crops["A"] = photo_panel(ax, COMPACT_A, P["A"])
    tag(ax, "A", "Cameras, vial and spatula" if variant == "compact" else "Cameras and presented vial")
    for text, xy, tip, dark_ground in COMPACT_A_LABELS:
        if boxed:
            callout_boxed(ax, text, xy, tip)
        else:
            callout(ax, text, xy, tip, dark=not dark_ground)
    if variant == "compact_E":
        ax = add_axes(fig, P["B"], H)
        crops["B"] = photo_panel(ax, COMPACT_E, P["B"])
        tag(ax, "B", "Spatula and force–torque sensor")
    out = IMAGES / f"fig_cameras_{variant}"
    fig.savefig(out.with_suffix(".png"), dpi=DPI, metadata={"Software": None})
    fig.savefig(out.with_suffix(".pdf"), dpi=DPI, metadata={"Creator": None, "Producer": None, "CreationDate": None})
    print(f"{out.name}: {W:.1f} x {H:.1f} mm, PNG at {DPI} dpi")
    print("panel     printed (mm)   source crop (px)          nominal ppi of crop")
    for k in P:
        _, _, w, h = P[k]
        box = crops[k]
        cw = box[2] - box[0]
        print(f"  {k:6s} {w:5.1f} x {h:4.1f}   {cw} x {box[3] - box[1]} at {box[:2]}   {cw / (w / 25.4):5.0f}")


if __name__ == "__main__":
    if "--variant" in sys.argv:
        v = sys.argv[sys.argv.index("--variant") + 1]
        if v == "full":
            main()
        elif v in COMPACT_BOXES:
            main_compact(v)
        else:
            raise SystemExit(f"unknown variant {v!r}: full, compact, compact_E")
    else:
        main()
