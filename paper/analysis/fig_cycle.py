"""Figure 3 (fig:cycle): one trial as run, panels A to G, in the order of the Autonomous Experiment section
(sec:benchmarking).

The side-camera captures of B, C and E were re-staged on 2026-10-02 at the presentation pose of the runs; B, C and
E share one crop window, and the spatula of the scoop capture is measured against the drawn path. Two two-row
layouts are built (LAYOUTS); MAIN_LAYOUT, with the cleaning step G, is the one in the paper.

    A  weigh the vials           text boxes "Tare goal vial", "Weigh source vial" (the order of sec:benchmarking);
                                 SOURCE_WEIGH_PHOTO, when set, replaces them with a phone photo on the balance,
                                 tagged SOURCE_WEIGH_PHOTO_VIAL; here it is weighing.jpg, as in F
    B  estimate the surface      side capture vialSideView.png (2026-10-02, NaCl) with its SAM 3 mask (local run of
                                 tools/segment_side_views.py with the runs' two point prompts), the surface line
                                 fitted by the paper's code path (fig3_examples.surface_line), p and n
    C  scoop                     side capture vial_sideVewWithSpatula.png (same pose) with F_vial lip at the lip,
                                 F_contact point at p + s_z n (s_z = SZ_MM, along the surface normal) with the axes
                                 of F_vial lip (as the scoop code builds it), the estimated
                                 surface, and the spatula-tip path generated offline by tools/scoop_path_tpscoop.py
                                 (a visualization: the executed path was not recorded with these captures)
    D  material on the spatula   stored top frame, read 2 (after the scoop), NaCl run of 2026-09-10, the
                                 median-mass trial at s_z = SZ_MM, with its stored SAM 3 mask and its balance mass
    (change vials)               unlettered text box "Change vials / source -> goal" (robot motion) between D and E;
                                 CHANGE_PHOTO, when set, replaces it with a photo
    E  deposit                   side capture vial_sideView_deposite.png (goal vial, same pose) with a curved arrow
                                 from the blade tip for material leaving the blade (forward, towards the vial base,
                                 and down) and the tap direction of the deposit (run config)
    F  weigh both vials          phone photo photos/weighing.jpg (2026-10-02): the GOAL vial (a few grains on its
                                 floor), the last weighing of a trial, tagged (WEIGH_PHOTO_VIAL)
    G  clean the spatula         photo photos/companion_hs1_spatula_cleaning.jpg: snapshot 6 ("Spatula
                                 cleaning") of the companion paper's Fig. 4

Geometry (printed with its source on every run):
  * Registration of the 2026-10-02 vial against the runs: outer glass edges against the click axis, the lip
    rim along the axis, the inner lower wall from the material masks; and the three captures against each
    other (ECC). Frames are drawn only if the vial sits on the clicks to REG_GATE_PX.
  * {V} on the image: origin at click 0 (the lip, "top of vial"), z along the click axis, y the in-plane
    perpendicular (its sign from the calibration: up), x into the page; scale 57 mm over the clicks, the
    observer's own (compute_plane_pose). Points off that plane are projected with a pinhole at the depth
    fx / scale. Cross-check: the same frame through world_to_camera_43_sensor and the vial_lip pose the
    polenta run's observer used at the presentation pose (projection_check, PROJ_GATE_PX).
  * The surface: plane_pose() reproduces the observer's compute_plane_pose (arms_ws state/surface_observer.py,
    the -pi/2 convention of 2026-09-07) and is checked here against every stored T_vialLip_surface.
  * {C} = T_vialLip_surface @ [0, 0, s_z] (arms_ws tp_gmm_action.surface_frame_contact_point), axes parallel
    to {V} (the paper's Task-Parameterized Sampling section; tp_scoop synthesize.frames_from_vial_pose).
  * The path: data/fig_cycle/scoop_path_tpscoop_sz-4.json from tools/scoop_path_tpscoop.py; its contact point must
    equal the one computed here, or the script stops and prints the command that regenerates it.
  * The spatula of the scoop capture: background subtraction against vialSideView.png (spatula_silhouette); its
    tip and shaft are measured against the drawn path, and the retake waypoint of tools/scoop_waypoint_pose_tpscoop.py
    is printed in vial_lip and in fr3_link0.

A loader for a recorded trial is kept (INPUTS: pose bag, image bag, stills or CSV, behind the TRAJ_GATE_OK gate):
a recorded trajectory would replace the generated path. No trajectory was recorded for the paper's figure.

Projection (validated in projection_check, printed on every run):
    p_cam = inv(T_world_cam43) @ T_world_base @ p_fr3          base == fr3_link0 (fr3.urdf, fixed identity joint)
    uv    = K43 @ p_cam / z                                     camera_43_sensor is an optical frame (z forward,
                                                                x right, y down); image_rect has no distortion
The stored `plane_origin_world` / `plane_normal_world` are in fr3_link0, not in `world`: the click-based observer
looks up its vial frames from fr3_link0 (surface_observer.py, Sam2DMaterialSurfaceObserver.observe).

Channel order: side frames (surface_observer/image) are stored BGR; top frames (spatula_observer/image_camera44)
are stored RGB. The 2026-10-02 PNGs are RGB (checked against a stored frame).

Run from paper/analysis/ after extract_runs.py, tools/segment_side_views.py, tools/scoop_path_tpscoop.py (and, for the
retake waypoint, tools/scoop_waypoint_pose_tpscoop.py):
    python fig_cycle.py [OUT_DIR] [--check CHECK_DIR]
Writes OUT_DIR/fig_cycle_2row.{pdf,png} and fig_cycle.{pdf,png} (the same figure); OUT_DIR defaults to ../output.
--check writes the check images (registration, frames, entry angle, spatula against the path, deposit) to
CHECK_DIR; nothing else is written there.
"""
from __future__ import annotations

import io
import json
import re
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib import patheffects
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as MplPath
from PIL import Image, ImageCms
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation, Slerp

from common import DATA, IMAGES, PHOTOS
from extract_runs import RUNS, extract
from ssvision.surface import compute_plane_pose, surface_line

# ============================================================================= inputs
FIGC = DATA / "fig_cycle"                     # capture trial, bags, stills, the generated path (git-ignored data)
CAPTURES = dict(                              # 2026-10-02, side camera, 1936 x 1216, RGB PNG (copied, metadata stripped)
    surface=PHOTOS / "vialSideView.png",              # source vial (NaCl) at the presentation pose
    scoop=PHOTOS / "vial_sideVewWithSpatula.png",     # same pose, spatula in the vial, tip near the surface
    deposit=PHOTOS / "vial_sideView_deposite.png",    # goal vial at the same pose, a small load on the blade
)
WEIGH_PHOTO = PHOTOS / "weighing.jpg"         # phone, 3024 x 4032, Display P3 (converted to sRGB on read); panel F
WEIGH_PHOTO_VIAL = "goal vial"                # what the photo shows (a few grains on the floor); tag in panel F
SOURCE_WEIGH_PHOTO = WEIGH_PHOTO              # panel A: the same photo as F. Swap in a
                                              # phone photo of the SOURCE vial on the balance from the viewpoint of
                                              # weighing.jpg when one exists (e.g. PHOTOS / "weighing_source.jpg");
                                              # None -> the two text boxes.  Cropped with the box of F.
SOURCE_WEIGH_PHOTO_VIAL = WEIGH_PHOTO_VIAL    # tag in panel A: what that photo shows ("source vial" for a source photo)
SHOW_PHOTO_TAGS = False                       # no "goal vial" tag on the photos in A and F
CHANGE_PHOTO = None                           # the change-vials step: an optional photo (None -> text box)
CHANGE_PHOTO_BOX = None                       # its crop (x0, y0, x1, y1) in photo pixels; None -> the whole photo
CLEAN_PHOTO = PHOTOS / "companion_hs1_spatula_cleaning.jpg"   # G: snapshot 6 "Spatula cleaning" of the
                                              # companion paper's Fig. 4 (figures/HS1_full.pdf, embedded image 0)
CLEAN_PHOTO_BOX = (780, 540, 2880, 4040)      # G: the FR3 flange, the spatula and the first sponge beaker, kept whole
MAIN_LAYOUT = "2row_G"                        # the layout copied to output/fig_cycle.* (the file the paper
                                              # includes): two rows, with the cleaning step G
MASK_MODELS = ("sam3", "sam2.1")              # tools/segment_side_views.py outputs, first found wins
SCOOP_PATH = FIGC / "scoop_path_tpscoop_sz-4.json"  # the path of the runs' TP-scoop model (tools/scoop_path_tpscoop.py)
WAYPOINT = FIGC / "scoop_waypoint_tpscoop.json"  # tools/scoop_waypoint_pose_tpscoop.py: the pose to re-take C at
                                                 # (optional, skipped if missing)
SCOOP_PATH_CMD = ("PYTHONDONTWRITEBYTECODE=1 "
                  "python paper/analysis/tools/scoop_path_tpscoop.py "   # in the tp-scoop environment of fig_tp_scoop.py
                  "--contact-mm {x:.3f} {y:.3f} {z:.3f} --sz-mm {sz:g}")
INPUTS = dict(                                # a recorded trial, when there is one (all None for the paper)
    trial_zarr=None,                          # e.g. FIGC / "benchmark_2026-10-XX-XX-XX.zarr"
    pose_bag=None,                            # /tf /tf_static /planned_waypoints joint_states camera_info
    image_bag=None,                           # /camera_43/pylon_ros2_camera_node/image_raw (+ camera_info)
    scoop_still=None, deposit_still=None,     # e.g. FIGC / "cam43_rect_<time_ns>.png"
    scoop_csv=None, deposit_csv=None,         # t,x,y,z,qx,qy,qz,qw of `spatula` in fr3_link0, t in s (ROS time)
    second_pose_bag=None, second_trial_zarr=None,   # its /planned_waypoints in its own scoop window, drawn dashed
)
# set True only after the capture check: the projected `spatula` frame lands on the visible blade in the
# capture frame (written by --check).  Until then no recorded trajectory is drawn even if the data are present.
TRAJ_GATE_OK = False
SCOOP_FRAME_T_NS = None                       # optional explicit capture times (ns, ROS time)
DEPOSIT_FRAME_T_NS = None

# ============================================================================= calibration
# arms_ws/src/arms_data_collection/config/static_tfs.yaml (unchanged in git since 2026-06-01: d5c627c, 1d38a35,
# 16d3879).  When a pose bag is given, its /tf_static values are used instead and the difference is printed.
WORLD_TO_BASE = dict(t=[-0.154826244, 0.795879495, -0.001069670],          # calibrated_on 2026-06-01
                     q=[-0.001543266, -0.006411778, 0.002305206, 0.999975596])
WORLD_TO_CAM43 = dict(t=[0.21917980429957762, 0.37006992469701405, 0.34090477006071773],   # 2026-05-26, ad hoc
                      q=[-0.5040699839109564, -0.5132240126130757, 0.48167330975590256, 0.5005051317080934])
# camera 43 intrinsics, tools/solid_sampling/config_benchmark_chem_real.yaml surface.intrinsics
# (= camera_43_intrinsic.yaml, captured 2026-05-26); P == K on image_rect, the topic of all runs
K43 = np.array([[2412.21103, 0.0, 959.638097], [0.0, 2409.38664, 586.63599], [0.0, 0.0, 1.0]])
IMG_W, IMG_H = 1936, 1216
PROJ_GATE_PX = 50.0       # a run's stored vial pose is used for the cross-check only if its max projection error
                          # is below this (about 2.4 mm at the vial).  Chosen after seeing the data.
PAPER_RUNS = ("salt", "sugar", "polenta")
VIAL_LIP_TO_BASE_M = 0.0567    # arms_data_collection/vial_pose.py lip_z (vial_base -> vial_lip)
EXTRA_RUNS = ()                # release: the 10-trial NaCl-prompt polenta run is not part of the paper
CALIB_RUN = "polenta"          # the run whose observer saw the presentation pose
VIAL_CLICK_LENGTH_MM = 57.0    # vial.outer_clickable_length_mm of the runs
REG_GATE_PX = 5.0              # the 2026-10-02 vial must sit on the runs' clicks to this, or no frames are drawn

# ============================================================================= what is shown
TOP_TRIAL = dict(material="salt", sz_mm=-4.0)    # D: the median-mass trial of this run and offset (pick_trial)
SZ_MM = -4.0                                     # C: the offset of {C} and of the generated path
SIDE_CROP = (330, 50, 1936, 905)                 # B, C, E: one window, the whole vial from base to lip (checked
                                                 # and printed) and the shaft out to the right image edge
TAP = dict(direction_lip=(0.0, -1.0, 0.0), n_taps=10)   # config_benchmark_HS1_vision.yaml tapped_deposit.tap
TAP_MARK_X = 1820                                # E: where the tap direction is drawn on the shaft (image x, px)
SCALE_BAR_XY = (1560, 700)                       # B: left end of the 5 mm bar (image px), free backdrop
SPATULA_DIFF_THR = 20.0     # grey levels: the blade is |capture - vialSideView.png| above this; the tip moves by
                            # at most 2 px for thresholds 12-25.  Chosen after seeing the data.
TIP_ON_PATH_MM = 1.0        # the tip is "on the path" if this close; then the spot is marked
TOPICS = dict(tf="/tf", tf_static="/tf_static", waypoints="/planned_waypoints",
              image_raw="/camera_43/pylon_ros2_camera_node/image_raw",
              image_rect="/camera_43/pylon_ros2_camera_node/image_rect",
              camera_info="/camera_43/pylon_ros2_camera_node/camera_info")
TIP = "spatula"

# ============================================================================= style
FIG_W_IN = 0.92 * 178.0 / 25.4          # 0.92 \textwidth; USG.cls 210 mm paper, 16 mm side margins (178 mm)
FS = 7.0
PHOTO_PPI = 600
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": FS, "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
})
KIND = {"balance": "#6A51A3", "vision": "#2C8C6E", "motion": "#E7A33E"}     # the bar under each title
INK, INK2, MUTED = "#1a1a1a", "#4d4d4d", "#8a8a8a"
C_VIAL = "#0072B2"                        # F_vial (Okabe-Ito blue)
C_CON = "#CC79A7"                         # F_contact (Okabe-Ito reddish purple): not red, which would read as
                                          # the surface line next to it
PATH_C = "#E7A33E"                        # the tip path: the robot-motion colour of KIND
SIDE_MASK = (60, 120, 255)                # fig3_examples.py
SURF_RED = "#ff3030"                      # fig3_examples.py
TOP_MASK = (255, 60, 60)                  # fig3_examples.py
HALO = [patheffects.withStroke(linewidth=2.2, foreground="black", alpha=0.55)]
MASK_OUTLINE_LW = 1.3      # a visible outline of the segmentation in B and D
MASK_OUTLINE_MIN_FRAC = 0.05    # outline only components of at least this fraction of the mask: in B the material body is 96 % of
                                # the mask and the next part 0.9 %, so only the body is outlined; specks stay in the fill
SURF_LINE_LW_B = 1.8       # the fitted surface line in B
HALO_W = [patheffects.withStroke(linewidth=1.8, foreground="white")]
LABEL_VIAL = r"$F_{\mathrm{vial\ lip}}$"           # frame labels: F, with upright subscripts
LABEL_CON = r"$F_{\mathrm{contact\ point}}$"


# ============================================================================= geometry
def T_from(t, q) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = Rotation.from_quat(q).as_matrix()
    M[:3, 3] = t
    return M


def project(T_cam_X: np.ndarray, P_X, K: np.ndarray = K43):
    """Pixels (N, 2) and optical-frame depth (N,) of points P_X (N, 3) given T_cam<-X."""
    P = np.atleast_2d(np.asarray(P_X, float))
    pc = np.c_[P, np.ones(len(P))] @ T_cam_X.T
    uvw = pc[:, :3] @ K.T
    return uvw[:, :2] / uvw[:, 2:3], pc[:, 2]


T_WORLD_BASE = T_from(**WORLD_TO_BASE)
T_WORLD_CAM = T_from(**WORLD_TO_CAM43)
T_CAM_WORLD = np.linalg.inv(T_WORLD_CAM)
T_CAM_FR3 = T_CAM_WORLD @ T_WORLD_BASE


def implied_vial_lip(su) -> dict:
    """The vial_lip frame the observer used, recovered from its outputs: origin_i = p_lip + z_i z_lip and
    normal_i = cos(a_i) z_lip - sin(a_i) y_lip, with z_i, a_i from T_vialLip_surface (rotation about x_lip)."""
    o = np.array(su["plane_origin_world"], float)
    n = np.array(su["plane_normal_world"], float)
    L = np.array(su["T_vialLip_surface_pose7"], float)
    z, a = L[:, 2], Rotation.from_quat(L[:, 3:7]).as_rotvec()[:, 0]
    A = np.zeros((3 * len(n), 6))
    for i in range(len(n)):
        A[3 * i:3 * i + 3, :3] = np.cos(a[i]) * np.eye(3)
        A[3 * i:3 * i + 3, 3:] = -np.sin(a[i]) * np.eye(3)
    sol = np.linalg.lstsq(A, n.ravel(), rcond=None)[0]
    zl, yl = sol[:3] / np.linalg.norm(sol[:3]), sol[3:] / np.linalg.norm(sol[3:])
    lips = o - np.outer(z, zl)
    p_lip = lips.mean(0)
    return {"p_lip": p_lip, "z": zl, "y": yl, "spread_um": float(np.ptp(lips, axis=0).max() * 1e6),
            "p_lip_world": (T_WORLD_BASE @ np.r_[p_lip, 1])[:3], "z_world": T_WORLD_BASE[:3, :3] @ zl}


def projection_check(runs=PAPER_RUNS + EXTRA_RUNS) -> dict:
    """Project every stored plane_origin_world (fr3_link0) into its own side frame; compare with the stored
    plane_center_image.  Also recover the vial_lip frame each run used (implied_vial_lip)."""
    out = {}
    for mat in runs:
        g = zarr.open_group(str(RUNS[mat]), mode="r")
        su = g["events"]["surface_observer"]
        o = np.array(su["plane_origin_world"], float)
        pc = np.array(su["plane_center_image"], float)
        uv, depth = project(T_CAM_FR3, o)
        d = uv - pc
        e = np.linalg.norm(d, axis=1)
        lip = implied_vial_lip(su)
        ends, _ = project(T_CAM_FR3, [lip["p_lip"], lip["p_lip"] - VIAL_LIP_TO_BASE_M * lip["z"]])
        clicks = np.array(su["vial_clicks_image"][0], float)
        out[mat] = dict(n=len(e), err=e, median=float(np.median(e)), max=float(e.max()), min=float(e.min()),
                        du=float(d[:, 0].mean()), dv=float(d[:, 1].mean()), depth=(float(depth.min()), float(depth.max())),
                        mm_per_px=float(np.median(depth)) * 1000.0 / K43[0, 0], lip=lip, ends=ends, clicks=clicks,
                        created=dict(g.attrs)["created_at"],
                        span_h=float(np.ptp(np.array(su["time_ns"], np.int64))) / 3.6e12)
    return out


def print_projection_check(pc: dict) -> None:
    print("\nProjection check: stored plane_origin_world (fr3_link0) -> camera_43_sensor -> K43, against the stored "
          "plane_center_image of the same frame")
    print(f"  {'run':20s} {'created':20s} {'n':>3s} {'median':>7s} {'max':>7s} {'min':>7s} {'mean du':>8s} {'mean dv':>8s}"
          f" {'depth [m]':>13s}  px")
    for mat, r in pc.items():
        print(f"  {mat:20s} {r['created']:20s} {r['n']:3d} {r['median']:7.1f} {r['max']:7.1f} {r['min']:7.1f} "
              f"{r['du']:8.1f} {r['dv']:8.1f} {r['depth'][0]:6.3f}-{r['depth'][1]:.3f}")
    allp = np.concatenate([pc[m]["err"] for m in PAPER_RUNS if m in pc])
    print(f"  {'ALL three paper runs':41s} {len(allp):3d} {np.median(allp):7.1f} {allp.max():7.1f} {allp.min():7.1f}")
    for mat, r in pc.items():
        if r["max"] < PROJ_GATE_PX:
            print(f"  {mat}: passes (max {r['max']:.1f} px < {PROJ_GATE_PX:.0f} px); 1 px = {r['mm_per_px']:.4f} mm "
                  f"at the stored point (depth / fx), so the median is {r['median'] * r['mm_per_px']:.2f} mm")
    print("  Vial_lip frame implied by each run's stored surface (lstsq over all trials; spread = max-min of the "
          "per-trial lip origin):")
    for mat, r in pc.items():
        L = r["lip"]
        el = np.degrees(np.arcsin(np.clip(L["z_world"][2], -1, 1)))
        print(f"    {mat:20s} lip in world [{L['p_lip_world'][0]:+.4f}, {L['p_lip_world'][1]:+.4f}, "
              f"{L['p_lip_world'][2]:+.4f}] m, axis elevation {el:5.1f} deg, per-trial spread {L['spread_um']:.3f} um over "
              f"{r['span_h']:.1f} h; "
              f"lip/base project to ({r['ends'][0, 0]:.0f}, {r['ends'][0, 1]:.0f}) / ({r['ends'][1, 0]:.0f}, "
              f"{r['ends'][1, 1]:.0f}) against clicks ({r['clicks'][0, 0]:.0f}, {r['clicks'][0, 1]:.0f}) / "
              f"({r['clicks'][1, 0]:.0f}, {r['clicks'][1, 1]:.0f})")


# ============================================================================= the stored trial of panel D
def rows_to_arrays(rows: list[dict]) -> dict:
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def pick_trial(d: dict, sz_mm: float) -> int:
    """Median-mass trial (lower median for an even count) among the trials at sz_mm with a mask after the scoop."""
    ok = np.where(np.isclose(d["sz_mm"], sz_mm) & (d["px_obs2"] > 0) & np.isfinite(d["carried_mg"]))[0]
    k = ok[np.argsort(d["carried_mg"][ok])][(len(ok) - 1) // 2]
    return int(d["iteration"][k])


def load_trial() -> dict:
    zpath = Path(INPUTS["trial_zarr"]) if INPUTS["trial_zarr"] else RUNS[TOP_TRIAL["material"]]
    rows, meta = extract(zpath)
    d = rows_to_arrays(rows)
    trial = int(d["iteration"][0]) if INPUTS["trial_zarr"] else pick_trial(d, TOP_TRIAL["sz_mm"])
    k = int(np.where(d["iteration"] == trial)[0][0])
    g = zarr.open_group(str(zpath), mode="r")
    su, sp = g["events"]["surface_observer"], g["events"]["spatula_observer"]
    i = int(np.where(np.array(su["iteration"]) == trial)[0][0])
    it, oi = np.array(sp["iteration"]), np.array(sp["observation_index"])
    j = int(np.where((it == trial) & (oi == 2))[0][0])
    level = (d["sz_mm"] == d["sz_mm"][k]) & (d["px_obs2"] > 0) & np.isfinite(d["carried_mg"])
    return {
        "zarr": zpath, "is_stand_in": INPUTS["trial_zarr"] is None, "trial": trial, "meta": meta,
        "sz_mm": float(d["sz_mm"][k]), "carried_mg": float(d["carried_mg"][k]), "px": int(d["px_obs2"][k]),
        "z_lip_mm": float(d["surface_z_lip_mm"][k]),
        "level_mass": (float(np.min(d["carried_mg"][level])), float(np.max(d["carried_mg"][level])), int(level.sum()),
                       int((d["sz_mm"] == d["sz_mm"][k]).sum())),
        "t": {key: int(d[key][k]) for key in d if key.startswith("t_")},
        "side": np.array(su["image"][i])[:, :, ::-1],              # stored BGR -> RGB
        "side_mask": np.array(su["mask"][i]) > 0,
        "clicks": np.array(su["vial_clicks_image"][i], float),
        "pc_img": np.array(su["plane_center_image"][i], float),
        "T_lip_surf": T_from(np.array(su["T_vialLip_surface_pose7"][i][:3], float),
                             np.array(su["T_vialLip_surface_pose7"][i][3:7], float)),
        "top": np.array(sp["image_camera44"][j]),                  # stored RGB -> no flip
        "top_mask": np.array(sp["mask_camera44"][j]) > 0,
    }


# ============================================================================= bags (rosbag2, MCAP)
def bag_messages(folder, topics, t0=None, t1=None):
    from mcap_ros2.reader import read_ros2_messages
    files = sorted(Path(folder).glob("*.mcap"))
    if not files:
        raise FileNotFoundError(f"no .mcap in {folder}; record with `ros2 bag record -s mcap`")
    for f in files:
        for m in read_ros2_messages(str(f), topics=list(topics), start_time=t0, end_time=t1):
            yield m.channel.topic, m.log_time_ns, m.ros_msg


def stamp_ns(header) -> int:
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


class TfBuffer:
    """/tf and /tf_static from a bag, looked up by time (linear translation, slerp rotation)."""

    def __init__(self):
        self.static: dict[str, tuple[str, np.ndarray]] = {}
        self.dyn: dict[str, dict] = {}

    def add(self, msg, is_static: bool) -> None:
        for tr in msg.transforms:
            parent, child = tr.header.frame_id.lstrip("/"), tr.child_frame_id.lstrip("/")
            t, q = tr.transform.translation, tr.transform.rotation
            if is_static:
                self.static[child] = (parent, T_from([t.x, t.y, t.z], [q.x, q.y, q.z, q.w]))
            else:
                d = self.dyn.setdefault(child, {"parent": parent, "t": [], "p": [], "q": []})
                d["t"].append(stamp_ns(tr.header)); d["p"].append([t.x, t.y, t.z]); d["q"].append([q.x, q.y, q.z, q.w])

    def finalize(self) -> "TfBuffer":
        for d in self.dyn.values():
            t = np.asarray(d["t"], np.int64)
            o = np.argsort(t, kind="stable")
            t, keep = np.unique(t[o], return_index=True)
            d["t"], d["p"], d["q"] = t, np.asarray(d["p"])[o][keep], np.asarray(d["q"])[o][keep]
            d["t0"] = int(t[0])                                    # slerp on times relative to the first sample
            d["slerp"] = Slerp((t - t[0]).astype(float), Rotation.from_quat(d["q"])) if len(t) > 1 else None
        return self

    @classmethod
    def from_bag(cls, folder) -> "TfBuffer":
        buf = cls()
        for topic, _, msg in bag_messages(folder, [TOPICS["tf"], TOPICS["tf_static"]]):
            buf.add(msg, topic == TOPICS["tf_static"])
        return buf.finalize()

    def edge(self, child: str, t_ns: int, tol_ns: int = 200_000_000):
        if child in self.static:
            return self.static[child]
        if child not in self.dyn:
            return None
        d = self.dyn[child]
        t = d["t"]
        if len(t) == 1 or t_ns <= t[0] or t_ns >= t[-1]:
            k = 0 if t_ns <= t[0] else len(t) - 1
            if abs(int(t[k]) - t_ns) > tol_ns:
                raise ValueError(f"tf {d['parent']}->{child}: no sample within {tol_ns / 1e9:.1f} s of {t_ns}")
            return d["parent"], T_from(d["p"][k], d["q"][k])
        k = int(np.searchsorted(t, t_ns))
        w = (t_ns - t[k - 1]) / (t[k] - t[k - 1])
        M = np.eye(4)
        M[:3, :3] = d["slerp"](float(t_ns - d["t0"])).as_matrix()
        M[:3, 3] = (1 - w) * d["p"][k - 1] + w * d["p"][k]
        return d["parent"], M

    def to_root(self, frame: str, t_ns: int):
        M, f = np.eye(4), frame
        while True:
            e = self.edge(f, t_ns)
            if e is None:
                return f, M
            f, M = e[0], e[1] @ M

    def lookup(self, target: str, source: str, t_ns: int) -> np.ndarray:
        """T_target<-source at t_ns."""
        rt, Mt = self.to_root(target, t_ns)
        rs, Ms = self.to_root(source, t_ns)
        if rt != rs:
            raise ValueError(f"tf: {target} (root {rt}) and {source} (root {rs}) are not connected")
        return np.linalg.inv(Mt) @ Ms

    def sample_times(self, child: str, t0: int, t1: int) -> np.ndarray:
        t = self.dyn[child]["t"]
        return t[(t >= t0) & (t <= t1)]


def planned_scoop(folder, t0: int, t1: int, T_world_base: np.ndarray) -> dict | None:
    """The generated scoop from /planned_waypoints: each track_trajectory publishes every streamed waypoint with
    world coordinates (motion_primitives.py l.409, header says fr3_link0) and then every 5th waypoint in fr3_link0
    (l.410).  Returns the full set in world, checked against the decimated set through world_to_base."""
    strips = []
    for _, t, msg in bag_messages(folder, [TOPICS["waypoints"]], t0, t1):
        for m in msg.markers:
            if m.ns == "planned_waypoints" and len(m.points):
                strips.append((t, np.array([[p.x, p.y, p.z] for p in m.points])))
    for (ta, A), (tb, B) in zip(strips, strips[1:]):
        if len(B) == len(range(0, len(A), 5)) and tb - ta < 2_000_000_000:
            Bw = (np.c_[B, np.ones(len(B))] @ T_world_base.T)[:, :3]
            err = float(np.abs(Bw - A[::5]).max())
            return {"world": A, "fr3": B, "t_ns": ta, "check_m": err}
    return None


def camera_info(folder) -> dict | None:
    for _, _, msg in bag_messages(folder, [TOPICS["camera_info"]]):
        return {"K": np.array(msg.k, float).reshape(3, 3), "D": np.array(msg.d, float),
                "R": np.array(msg.r, float).reshape(3, 3), "P": np.array(msg.p, float).reshape(3, 4),
                "model": msg.distortion_model, "size": (int(msg.width), int(msg.height))}
    return None


def to_rgb(msg, cinfo: dict | None) -> np.ndarray:
    """sensor_msgs/Image -> RGB.  image_raw (bayer_rggb8) is debayered as cv_bridge does for bgr8
    (COLOR_BayerBG2BGR) and rectified as image_geometry does (initUndistortRectifyMap(K, D, R, P), INTER_LINEAR)."""
    h, w, enc = int(msg.height), int(msg.width), msg.encoding
    buf = np.frombuffer(bytes(msg.data), np.uint8).reshape(h, int(msg.step))
    if enc == "bayer_rggb8":
        bgr = cv2.cvtColor(np.ascontiguousarray(buf[:, :w]), cv2.COLOR_BayerBG2BGR)
        if cinfo is None:
            raise ValueError("image_raw needs camera_info in the bag to be rectified")
        m1, m2 = cv2.initUndistortRectifyMap(cinfo["K"], cinfo["D"], cinfo["R"], cinfo["P"][:, :3], (w, h), cv2.CV_16SC2)
        bgr = cv2.remap(bgr, m1, m2, cv2.INTER_LINEAR)
    elif enc in ("bgr8", "rgb8"):
        bgr = buf[:, :3 * w].reshape(h, w, 3)
        if enc == "rgb8":
            return bgr.copy()
    else:
        raise ValueError(f"image encoding {enc!r} not handled")
    return bgr[:, :, ::-1].copy()


def nearest_image(folder, t_ns: int, cinfo, window_ns: int = 1_000_000_000):
    best = None
    for topic, _, msg in bag_messages(folder, [TOPICS["image_raw"], TOPICS["image_rect"]], t_ns - window_ns,
                                      t_ns + window_ns):
        dt = abs(stamp_ns(msg.header) - t_ns)
        if best is None or dt < best[0]:
            best = (dt, topic, stamp_ns(msg.header), msg)
    if best is None:
        return None
    return {"rgb": to_rgb(best[3], cinfo), "t_ns": best[2], "dt_ms": best[0] / 1e6, "topic": best[1]}


def read_csv_traj(path) -> dict:
    a = np.genfromtxt(path, delimiter=",", names=True)
    P = np.c_[a["x"], a["y"], a["z"]]
    return {"t": np.round(a["t"] * 1e9).astype(np.int64), "fr3": P,
            "world": (np.c_[P, np.ones(len(P))] @ T_WORLD_BASE.T)[:, :3], "z_lip": None}


def measured_path(tf: TfBuffer, t0: int, t1: int) -> dict:
    """World positions of the `spatula` frame at every FR3 joint-TF sample in [t0, t1]."""
    child = next(c for c in ("fr3_link7", "fr3_link6", "fr3_link1") if c in tf.dyn)
    ts = tf.sample_times(child, t0, t1)
    P = np.array([tf.lookup("world", TIP, int(t))[:3, 3] for t in ts])
    Z = np.array([tf.lookup("vial_lip", TIP, int(t))[2, 3] for t in ts]) if "vial_lip" in tf.static else None
    return {"t": ts, "world": P, "z_lip": Z}


# ============================================================================= capture data
def capture_data(tr: dict) -> dict:
    """Everything that comes from a recorded capture; empty when nothing is in INPUTS."""
    cap = {"have": False}
    t = tr["t"]
    if INPUTS["pose_bag"]:
        tf = TfBuffer.from_bag(INPUTS["pose_bag"])
        cap["tf"] = tf
        Twb = tf.lookup("world", "base", t["t_surface_ns"])
        Twc = tf.lookup("world", "camera_43_sensor", t["t_surface_ns"])
        cap["calib_diff_mm"] = (np.abs(Twb[:3, 3] - T_WORLD_BASE[:3, 3]).max() * 1e3,
                                np.abs(Twc[:3, 3] - T_WORLD_CAM[:3, 3]).max() * 1e3)
        cap["T_cam_world"] = np.linalg.inv(Twc)
        cap["T_world_base"] = Twb
        ci = camera_info(INPUTS["pose_bag"])
        cap["K"] = ci["P"][:, :3] if ci is not None else K43
        cap["cinfo"] = ci
        cap["T_world_lip"] = tf.lookup("world", "vial_lip", t["t_surface_ns"])
        cap["scoop_planned"] = planned_scoop(INPUTS["pose_bag"], t["t_obs1_ns"], t["t_obs2_ns"], Twb)
        cap["scoop_meas"] = measured_path(tf, t["t_obs1_ns"], t["t_obs2_ns"])
        cap["deposit_meas"] = measured_path(tf, t["t_obs3_ns"], t["t_obs4_ns"])
        cap["have"] = True
        if INPUTS["second_pose_bag"] and INPUTS["second_trial_zarr"]:
            r2 = rows_to_arrays(extract(Path(INPUTS["second_trial_zarr"]))[0])
            tf2 = TfBuffer.from_bag(INPUTS["second_pose_bag"])
            Twb2 = tf2.lookup("world", "base", int(r2["t_surface_ns"][0]))
            sp2 = planned_scoop(INPUTS["second_pose_bag"], int(r2["t_obs1_ns"][0]), int(r2["t_obs2_ns"][0]), Twb2)
            if sp2 is not None:
                sp2["sz_mm"] = float(r2["sz_mm"][0])
                cap["scoop_planned_2"] = sp2
    for key in ("scoop", "deposit"):
        if INPUTS[f"{key}_csv"]:
            cap[f"{key}_meas"] = read_csv_traj(INPUTS[f"{key}_csv"])
            cap.setdefault("T_cam_world", T_CAM_WORLD); cap.setdefault("K", K43)
            cap["have"] = True
    # capture frames: bag (deepest spatula frame in the window) or stills
    for key, win, t_fix in (("scoop", ("t_obs1_ns", "t_obs2_ns"), SCOOP_FRAME_T_NS),
                            ("deposit", ("t_obs3_ns", "t_obs4_ns"), DEPOSIT_FRAME_T_NS)):
        if INPUTS[f"{key}_still"]:
            path = Path(INPUTS[f"{key}_still"])
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            m_ = re.fullmatch(r"cam43_rect_(\d{18,20})\.png", path.name)      # time.time_ns() in the name
            t_still = t_fix if t_fix else (int(m_.group(1)) if m_ else None)
            cap[f"{key}_frame"] = {"rgb": bgr[:, :, ::-1].copy(), "t_ns": t_still, "topic": "still"}
        elif INPUTS["image_bag"] and f"{key}_meas" in cap:
            m = cap[f"{key}_meas"]
            if t_fix:
                t_frame = t_fix
            elif m.get("z_lip") is not None and len(m["z_lip"]):
                t_frame = int(m["t"][int(np.argmin(m["z_lip"]))])            # spatula deepest along the vial axis
            else:
                t_frame = (tr["t"][win[0]] + tr["t"][win[1]]) // 2           # no vial pose: middle of the window
            ci = cap.get("cinfo") or camera_info(INPUTS["image_bag"])
            cap[f"{key}_frame"] = nearest_image(INPUTS["image_bag"], t_frame, ci)
    if INPUTS["image_bag"] and "tf" in cap:      # rectification check: the bag frame at the surface image
        fr = nearest_image(INPUTS["image_bag"], t["t_surface_ns"], cap.get("cinfo"))
        if fr is not None:
            cap["rect_check"] = (fr["dt_ms"], float(np.abs(fr["rgb"].astype(float) - tr["side"].astype(float)).mean()))
    return cap



# ============================================================================= the 2026-10-02 captures
def load_captures() -> dict:
    caps = {}
    for key, path in CAPTURES.items():
        rgb = np.asarray(Image.open(path).convert("RGB"))
        if rgb.shape[:2] != (IMG_H, IMG_W):
            raise ValueError(f"{path.name}: expected {IMG_W} x {IMG_H}, got {rgb.shape[1]} x {rgb.shape[0]}")
        caps[key] = rgb
    caps["masks"], caps["mask_model"] = {}, None
    for model in MASK_MODELS:                                  # SAM 3 first; a SAM 2.1 mask is marked provisional
        files = {k: p.with_name(f"{p.stem}.mask_{model}.png") for k, p in CAPTURES.items()}
        if files["surface"].exists():
            caps["mask_model"] = model
            caps["masks"] = {k: np.asarray(Image.open(f)) > 0 for k, f in files.items() if f.exists()}
            log = PHOTOS / f"segment_side_views_{model}.json"
            caps["mask_log"] = json.loads(log.read_text()) if log.exists() else {}
            break
    if caps["mask_model"] is None:
        raise SystemExit("no mask for vialSideView.png: run tools/segment_side_views.py first")
    caps["provisional"] = caps["mask_model"] != "sam3"
    return caps


def load_weigh_photo(path=WEIGH_PHOTO) -> np.ndarray:
    """A phone photo in sRGB (weighing.jpg carries a Display P3 profile, the only metadata kept in the copy)."""
    im = Image.open(path)
    icc = im.info.get("icc_profile")
    if icc:
        im = ImageCms.profileToProfile(im, ImageCms.ImageCmsProfile(io.BytesIO(icc)), ImageCms.createProfile("sRGB"),
                                       outputMode="RGB")
    return np.asarray(im.convert("RGB"))


# ============================================================================= surface estimate (observer's model)
def plane_pose(mask: np.ndarray, clicks: np.ndarray) -> dict:
    """compute_surface_plane / compute_plane_pose of arms_ws state/surface_observer.py on the paper's edge fit:
    the inlier span of the RANSAC line, sorted across the axis; theta = -pi/2 - angle(edge -> axis) about x_lip
    (the 2026-09-07 convention), z of the surface centre along the axis from click 0 (the lip), 57 mm over the clicks.
    The line and the pose are ssvision.surface.surface_line and compute_plane_pose, as in the demo."""
    c, v, pts, inl = surface_line(mask.astype(np.uint8), clicks)
    t = (pts[inl] - c) @ v
    pair = np.vstack([c + v * t.min(), c + v * t.max()])
    top, bot = clicks[0], clicks[1]
    dv = (top - bot) / np.linalg.norm(top - bot)
    pair = pair[np.argsort(pair @ np.array([-dv[1], dv[0]]))]
    theta, depth_m, p_img = compute_plane_pose(pair[0], pair[1], top, bot, VIAL_CLICK_LENGTH_MM / 1000.0)
    z_mm = depth_m * 1000.0
    n_img = np.array([v[1], -v[0]])                      # image normal of the line, away from the material
    ys, xs = np.nonzero(mask)
    if np.dot([xs.mean() - p_img[0], ys.mean() - p_img[1]], n_img) > 0:
        n_img = -n_img
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("x", theta).as_matrix()
    T[2, 3] = z_mm
    return {"theta": float(theta), "z_mm": float(z_mm), "p_img": p_img, "c": c, "v": v, "n_img": n_img,
            "T_lip_surf_mm": T, "pts": pts, "inl": inl}


def contact_point_mm(ps: dict, sz_mm: float) -> np.ndarray:
    """{C} origin in vial_lip, mm: T_vialLip_surface @ [0, 0, s_z] (tp_gmm_action.surface_frame_contact_point)."""
    return (ps["T_lip_surf_mm"] @ np.array([0.0, 0.0, sz_mm, 1.0]))[:3]


def check_plane_pose(runs=PAPER_RUNS) -> dict:
    """plane_pose() against the stored T_vialLip_surface and plane_center_image of every trial."""
    out = {}
    for mat in runs:
        su = zarr.open_group(str(RUNS[mat]), mode="r")["events"]["surface_observer"]
        P7, PC = np.array(su["T_vialLip_surface_pose7"]), np.array(su["plane_center_image"])
        e = []
        for i in range(len(P7)):
            ps = plane_pose(np.array(su["mask"][i]) > 0, np.array(su["vial_clicks_image"][i], float))
            e.append((np.degrees(ps["theta"] - Rotation.from_quat(P7[i, 3:7]).as_rotvec()[0]),
                      ps["z_mm"] - 1000.0 * P7[i, 2], np.linalg.norm(ps["p_img"] - PC[i])))
        out[mat] = np.abs(np.array(e)).max(0)
    return out


# ============================================================================= registration (2026-10-02 vs the runs)
def gray(rgb: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return cv2.GaussianBlur(cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32),
                            (0, 0), sigma)


def axis_basis(clicks):
    c0, c1 = np.asarray(clicks, float)
    L = float(np.linalg.norm(c0 - c1))
    ez = (c0 - c1) / L
    return c0, c1, L, ez, np.array([ez[1], -ez[0]])           # ey: in-plane perpendicular, points up


def outline_midline(rgb: np.ndarray, clicks) -> dict:
    """The outer glass edges of the vial body, sampled perpendicular to the click axis (t = 300-790 px from the
    base click): pass 1 takes the strongest edge 230-360 px from the axis on each side, pass 2 the strongest
    within 30 px of the pass-1 lines; robust line fits (3 rounds, 2.5 sigma)."""
    c0, c1, L, ez, ey = axis_basis(clicks)
    G = gray(rgb)
    ts, s = np.arange(300, 800, 10.0), np.arange(-380, 381, 0.25)
    prof = []
    for t in ts:
        P = c1[None] + t * ez[None] + s[:, None] * ey[None]
        prof.append(np.abs(np.gradient(map_coordinates(G, [P[:, 1], P[:, 0]], order=1), s)))
    prof = np.array(prof)

    def pick(lo_fn, hi_fn):
        e = np.zeros((len(ts), 2))
        for i, t in enumerate(ts):
            for j, (lo, hi) in enumerate((lo_fn(t, 0), hi_fn(t, 1))):
                w = (s > lo) & (s < hi)
                e[i, j] = s[np.argmax(np.where(w, prof[i], -1))]
        return e

    def fit(e):
        ok = np.ones(len(ts), bool)
        for _ in range(3):
            pu, pl = np.polyfit(ts[ok], e[ok, 0], 1), np.polyfit(ts[ok], e[ok, 1], 1)
            ru, rl = e[:, 0] - np.polyval(pu, ts), e[:, 1] - np.polyval(pl, ts)
            sd = max(1.0, float(np.std(np.r_[ru[ok], rl[ok]])))
            ok = (np.abs(ru) < 2.5 * sd) & (np.abs(rl) < 2.5 * sd)
        return pu, pl, ok, float(np.std(np.r_[ru[ok], rl[ok]]))

    e1 = pick(lambda t, j: (230, 360), lambda t, j: (-360, -230))
    pu, pl, _, _ = fit(e1)
    e2 = pick(lambda t, j: (np.polyval(pu, t) - 30, np.polyval(pu, t) + 30),
              lambda t, j: (np.polyval(pl, t) - 30, np.polyval(pl, t) + 30))
    pu, pl, ok, rms = fit(e2)
    mid = (pu + pl) / 2
    return {"base_px": float(np.polyval(mid, 0.0)), "lip_px": float(np.polyval(mid, L)),
            "angle_deg": float(np.degrees(np.arctan(mid[0]))), "rms_px": rms, "kept": (int(ok.sum()), len(ts)),
            "dia_px": float(np.polyval(pu - pl, 550.0)), "edges": (ts, e2, ok), "lines": (pu, pl)}


def rim_along_axis(rgb: np.ndarray, clicks, offsets=(-20.0, 0.0, 20.0)) -> dict:
    """Where the glass ends beyond click 0 along the axis: per profile parallel to the axis (offsets across it),
    the last edge above 35 % of the profile's strongest; the median over the offsets."""
    c0, c1, L, ez, ey = axis_basis(clicks)
    G = gray(rgb)
    s = np.arange(L - 150, L + 160, 0.5)
    ends = []
    for off in offsets:
        P = c1[None] + s[:, None] * ez[None] + off * ey[None]
        dv = np.abs(np.gradient(map_coordinates(G, [P[:, 1], P[:, 0]], order=1), s))
        k = np.where(dv > 0.35 * dv.max())[0]
        ends.append(float(s[k.max()] - L))
    return {"median_px": float(np.median(ends)), "ends_px": ends}


def inner_wall(mask: np.ndarray, clicks) -> dict:
    """The material's lower boundary where it lies on the glass (t = 150-700 px from the base click): the inner
    lower wall, as a line across-axis offset s(t); and the inner base end (0.2 percentile of t)."""
    c0, c1, L, ez, ey = axis_basis(clicks)
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float)
    t, s = (c - c1) @ ez, (c - c1) @ ey
    R = np.array([(t0 + 5, s[(t >= t0) & (t < t0 + 10)].min()) for t0 in np.arange(150, 700, 10)
                  if ((t >= t0) & (t < t0 + 10)).sum() >= 2])
    p = np.polyfit(R[:, 0], R[:, 1], 1)
    res = R[:, 1] - np.polyval(p, R[:, 0])
    ok = np.abs(res) < 3 * max(res.std(), 0.5)
    p = np.polyfit(R[ok, 0], R[ok, 1], 1)
    return {"base_px": float(p[1]), "lip_px": float(np.polyval(p, L)), "angle_deg": float(np.degrees(np.arctan(p[0]))),
            "base_end_px": float(np.percentile(t, 0.2)), "line": p}


def ecc_shift(ref: np.ndarray, mov: np.ndarray, clicks, region: str) -> dict:
    """Euclidean ECC (intensity, sigma 1.5 px) of mov onto ref inside a mask; displacement at the two clicks."""
    m = np.zeros((IMG_H, IMG_W), np.uint8)
    m[:, :600] = 255                                          # gripper, holder and flange
    if region == "holder":
        m[90:200, 1290:1440] = 255                            # and the two neck clips of the vial holder
        m[540:660, 1400:1550] = 255
    a, b = gray(ref, 1.5) / 255.0, gray(mov, 1.5) / 255.0
    W = np.eye(2, 3, dtype=np.float32)
    cc, W = cv2.findTransformECC(a, b, W, cv2.MOTION_EUCLIDEAN,
                                 (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6), m, 5)
    P = np.asarray(clicks, float)
    d = P @ W[:, :2].T.astype(float) + W[:, 2].astype(float) - P
    return {"cc": float(cc), "rot_deg": float(np.degrees(np.arctan2(W[1, 0], W[0, 0]))),
            "d_lip": d[0], "d_base": d[1], "W": W}                 # W: ref pixel -> mov pixel


def registration(caps: dict, tr: dict, clicks) -> dict:
    stored = tr["side"]
    out = {"outline_new": outline_midline(caps["surface"], clicks),
           "rim_new": rim_along_axis(caps["surface"], clicks), "rim_stored": rim_along_axis(stored, clicks),
           "wall_new": inner_wall(caps["masks"]["surface"], clicks), "wall_stored": inner_wall(tr["side_mask"], clicks),
           "ecc_scoop": ecc_shift(caps["surface"], caps["scoop"], clicks, "holder"),
           "ecc_deposit": ecc_shift(caps["surface"], caps["deposit"], clicks, "holder"),
           "ecc_gripper": ecc_shift(stored, caps["surface"], clicks, "gripper")}
    o = out["outline_new"]
    out["rim_diff_px"] = out["rim_new"]["median_px"] - out["rim_stored"]["median_px"]
    worst = max(abs(o["base_px"]), abs(o["lip_px"]), abs(out["rim_diff_px"]),
                float(np.linalg.norm(out["ecc_scoop"]["d_lip"])), float(np.linalg.norm(out["ecc_deposit"]["d_lip"])))
    out["worst_px"], out["pass"] = worst, worst < REG_GATE_PX
    return out


def print_registration(reg: dict, tr: dict) -> None:
    o, wn, ws = reg["outline_new"], reg["wall_new"], reg["wall_stored"]
    st = f"{TOP_TRIAL['material']} trial {tr['trial']}"
    print(f"\nRegistration of vialSideView.png (2026-10-02) against the runs (clicks {VIAL_CLICKS_TXT}):")
    print(f"  outer glass edges, 2026-10-02: midline {o['base_px']:+.2f} px from the click axis at the base click, "
          f"{o['lip_px']:+.2f} px at the lip click, {o['angle_deg']:+.3f} deg to it ({o['kept'][0]}/{o['kept'][1]} "
          f"profiles, rms {o['rms_px']:.2f} px; outer diameter {o['dia_px']:.1f} px)")
    print(f"  lip rim along the axis beyond click 0: {reg['rim_new']['median_px']:+.1f} px (2026-10-02) vs "
          f"{reg['rim_stored']['median_px']:+.1f} px ({st}): {reg['rim_diff_px']:+.1f} px")
    print(f"  inner lower wall from the material masks, 2026-10-02 minus {st}: {wn['base_px'] - ws['base_px']:+.2f} px at "
          f"the base click, {wn['angle_deg'] - ws['angle_deg']:+.3f} deg ({wn['lip_px'] - ws['lip_px']:+.1f} px "
          f"extrapolated to the lip click); inner base end {wn['base_end_px'] - ws['base_end_px']:+.1f} px along the axis")
    g = reg["ecc_gripper"]
    print(f"  gripper and holder (ECC, x < 600 px), 2026-10-02 vs {st}: ({g['d_base'][0]:+.1f}, {g['d_base'][1]:+.1f}) px "
          f"at the base click, rotation {g['rot_deg']:+.3f} deg, cc {g['cc']:.3f}: the holder moved, the vial glass did not")
    for k in ("scoop", "deposit"):
        e = reg[f"ecc_{k}"]
        print(f"  {CAPTURES[k].name} vs vialSideView.png (ECC on the gripper and the holder clips): {np.linalg.norm(e['d_lip']):.2f} px at the lip "
              f"click, {np.linalg.norm(e['d_base']):.2f} px at the base click, rotation {e['rot_deg']:+.3f} deg, cc {e['cc']:.3f}")
    print(f"  gate: worst {reg['worst_px']:.2f} px {'<' if reg['pass'] else '>='} REG_GATE_PX = {REG_GATE_PX:.0f} px -> "
          f"{'the run clicks apply; frames drawn' if reg['pass'] else 'FRAMES NOT DRAWN'}")


VIAL_CLICKS_TXT = ""


# ============================================================================= frames on the image
class ImageVialFrame:
    """vial_lip (mm) -> image pixels, anchored on the image: origin at click 0, z along the click axis, y the
    in-plane perpendicular (sign y_sign), x into the page; scale = |click0 - click1| / 57 mm at the vial axis,
    whose depth is then fx / scale; a point off the y-z plane is projected with the pinhole of K43."""

    def __init__(self, clicks, y_sign: float = 1.0, K: np.ndarray = K43):
        self.o, _, L, self.ez, ey = axis_basis(clicks)
        self.ey = y_sign * ey
        self.scale = L / VIAL_CLICK_LENGTH_MM                     # px per mm
        self.fx, self.pp = float(K[0, 0]), K[:2, 2].astype(float)
        self.Z0 = self.fx / self.scale                             # mm

    def __call__(self, P_mm) -> np.ndarray:
        P = np.atleast_2d(np.asarray(P_mm, float))
        plane = (self.o - self.pp) * self.Z0 / self.fx + P[:, 1:2] * self.ey + P[:, 2:3] * self.ez
        return self.pp + self.fx * plane / (self.Z0 + P[:, 0:1])


def calib_frame(pc: dict):
    """vial_lip (mm) -> pixels through the calibration: the vial_lip pose the CALIB_RUN observer used (fr3_link0),
    T_CAM_FR3 and K43.  Returns the mapping and the camera-frame axes."""
    L = pc[CALIB_RUN]["lip"]
    x = np.cross(L["y"], L["z"])
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, L["y"], L["z"], L["p_lip"]
    T_cam_lip = T_CAM_FR3 @ T
    return (lambda P_mm: project(T_cam_lip, np.atleast_2d(P_mm) / 1000.0)[0]), T_cam_lip


def calib_crosscheck(img_fr: ImageVialFrame, cal, T_cam_lip, contact_mm, path_mm) -> dict:
    o_i, o_c = img_fr([0, 0, 0])[0], cal([0, 0, 0])[0]
    d = o_c - o_i
    out = {"origin_img": o_i, "origin_cal": o_c, "origin_d": d, "origin_along": float(d @ img_fr.ez),
           "origin_across": float(d @ img_fr.ey)}
    for ax, k in (("z", 2), ("y", 1)):
        e = np.zeros(3)
        e[k] = 10.0
        di, dc = img_fr(e)[0] - o_i, cal(e)[0] - o_c
        out[f"{ax}_angle_deg"] = float(np.degrees(np.arctan2(di[0] * dc[1] - di[1] * dc[0], di @ dc)))
        out[f"{ax}_scale"] = float(np.linalg.norm(dc) / np.linalg.norm(di))
    xc = T_cam_lip[:3, 0]
    out["x_tilt_deg"] = float(np.degrees(np.arccos(np.clip(xc[2], -1, 1))))
    out["x_sign"] = float(np.sign(xc[2]))                     # +1: x_lip points away from the camera (into the page)
    out["y_cam"] = T_cam_lip[:3, 1]
    out["contact_d"] = float(np.linalg.norm(cal(contact_mm)[0] - img_fr(contact_mm)[0]))
    dp = np.linalg.norm(cal(path_mm) - img_fr(path_mm), axis=1)
    out["path_d_max"], out["path_d_median"] = float(dp.max()), float(np.median(dp))
    return out


# ============================================================================= the generated path
def load_path(contact_mm: np.ndarray) -> dict:
    if not SCOOP_PATH.exists():
        raise SystemExit(f"{SCOOP_PATH} missing; generate it with\n  " + SCOOP_PATH_CMD.format(
            x=contact_mm[0], y=contact_mm[1], z=contact_mm[2], sz=SZ_MM))
    d = json.loads(SCOOP_PATH.read_text())
    if abs(d["sz_mm"] - SZ_MM) > 1e-9 or np.abs(np.array(d["contact_mm"]) - contact_mm).max() > 1e-3:
        raise SystemExit(f"{SCOOP_PATH.name} was generated for contact {np.round(d['contact_mm'], 3)} mm, s_z "
                         f"{d['sz_mm']} mm; the surface now gives {np.round(contact_mm, 3)} mm. Regenerate with\n  "
                         + SCOOP_PATH_CMD.format(x=contact_mm[0], y=contact_mm[1], z=contact_mm[2], sz=SZ_MM))
    d["tip"], d["root"] = np.array(d["tip_mm"]), np.array(d["root_mm"])
    d["t"] = np.arange(d["n"]) * d["dt_s"]
    return d


def shaft_line(rgb: np.ndarray, x0=1745, x1=1930, y0=120, y1=330) -> dict:
    """The spatula shaft outside the vial, a dark rod on the grey backdrop: per column the centroid of the
    intensity dip, then a total-least-squares line."""
    G = gray(rgb, 1.5)
    pts = []
    for x in range(x0, x1, 3):
        col = G[y0:y1, x]
        dip = np.median(col) - col
        k = int(np.argmax(dip))
        if dip[k] < 15:
            continue
        lo, hi = max(0, k - 6), min(len(col), k + 7)
        w = np.clip(dip[lo:hi] - 0.5 * dip[k], 0, None)
        pts.append((x, y0 + float((w * np.arange(lo, hi)).sum() / w.sum())))
    P = np.array(pts)
    c = P.mean(0)
    v = np.linalg.svd(P - c)[2][0]
    v = v if v[0] > 0 else -v                                  # pointing out of the vial (to the right)
    res = (P - c) @ np.array([-v[1], v[0]])
    return {"c": c, "v": v, "n": len(P), "rms": float(res.std()), "angle_deg": float(np.degrees(np.arctan2(v[1], v[0])))}


def signed_angle(a, b) -> float:
    """Angle from image direction a to b, degrees, positive counter-clockwise as seen (image y down)."""
    return float(np.degrees(np.arctan2(-(a[0] * b[1] - a[1] * b[0]), a @ b)))


def entry_check(caps, img_fr: ImageVialFrame, path: dict) -> dict:
    sh = shaft_line(caps["scoop"])
    out = {"shaft": sh, "shaft_vs_axis": signed_angle(img_fr.ez, sh["v"])}
    k_close = int(round(path["closest_t_s"] / path["dt_s"]))
    for name, k in (("start", 0), ("contact", k_close)):
        tip, root = img_fr(path["tip"][k])[0], img_fr(path["root"][k])[0]
        u = (root - tip) / np.linalg.norm(root - tip)
        out[f"blade_{name}"] = {"t_s": k * path["dt_s"], "dir": u, "vs_axis": signed_angle(img_fr.ez, u),
                                "vs_shaft": signed_angle(sh["v"], u)}
    a, b = img_fr(path["tip"][0])[0], img_fr(path["tip"][k_close])[0]
    u = (a - b) / np.linalg.norm(a - b)                        # entry chord, drawn pointing out of the vial
    out["chord"] = {"vs_axis": signed_angle(img_fr.ez, u), "vs_shaft": signed_angle(sh["v"], u)}
    return out


def spatula_silhouette(mov: np.ndarray, ref: np.ndarray, W: np.ndarray, sh: dict, thr: float = SPATULA_DIFF_THR,
                       x_range=(1000, 1340), half_width=60.0, seed_x=1280.0) -> dict:
    """The blade inside the vial by background subtraction: |grey(mov) - grey(ref)| > thr, with ref =
    vialSideView.png (the same pose without the spatula) mapped onto mov by the ECC warp W of the registration
    (gripper and holder clips), inside a band of +-half_width px about the shaft line and inside x_range (the vial
    body, short of the neck threads); opened 3 x 3, closed 5 x 5.  The blade is the connected component on the shaft
    line at seed_x.  tip: its extreme pixel along the shaft axis into the vial; front: the mean of its pixels within
    6 px (along the axis) of the tip, the middle of the front edge."""
    D = np.abs(gray(mov, 1.5) - cv2.warpAffine(gray(ref, 1.5), W, (IMG_W, IMG_H), flags=cv2.INTER_LINEAR))
    c, v = sh["c"], sh["v"]
    nrm = np.array([-v[1], v[0]])
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]
    along = (xx - c[0]) * v[0] + (yy - c[1]) * v[1]
    across = (xx - c[0]) * nrm[0] + (yy - c[1]) * nrm[1]
    M = ((D > thr) & (np.abs(across) < half_width) & (xx > x_range[0]) & (xx < x_range[1])).astype(np.uint8)
    M = cv2.morphologyEx(M, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    M = cv2.morphologyEx(M, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    _, lab = cv2.connectedComponents(M)
    sy = int(round(c[1] + (seed_x - c[0]) * v[1] / v[0]))
    win = lab[sy - 15:sy + 16, int(seed_x) - 15:int(seed_x) + 16]
    ids, cnt = np.unique(win[win > 0], return_counts=True)
    if not len(ids):
        raise SystemExit(f"spatula_silhouette: no blade on the shaft line at x = {seed_x:.0f} (thr {thr})")
    comp = lab == ids[np.argmax(cnt)]
    ys, xs = np.nonzero(comp)
    a = along[ys, xs]
    j = int(np.argmin(a))
    fe = a < a[j] + 6.0
    return {"mask": comp, "area": int(comp.sum()), "tip": np.array([xs[j], ys[j]], float),
            "front": np.array([xs[fe].mean(), ys[fe].mean()]), "tip_across_px": float(across[ys[j], xs[j]]),
            "front_span_px": float(np.ptp(across[ys[fe], xs[fe]])), "thr": thr}


def nearest_on_path(q, uv: np.ndarray, dt: float) -> dict:
    """The point of the polyline uv nearest to the image point q: distance (px), time along the path (s), point."""
    a, ab = uv[:-1], np.diff(uv, axis=0)
    u = np.clip(((q - a) * ab).sum(1) / (ab * ab).sum(1), 0.0, 1.0)
    P = a + u[:, None] * ab
    d = np.linalg.norm(P - q, axis=1)
    i = int(np.argmin(d))
    return {"d_px": float(d[i]), "t_s": float((i + u[i]) * dt), "uv": P[i]}


def spatula_check(caps, reg, fr: ImageVialFrame, path: dict, ps: dict) -> dict:
    """The spatula of the scoop capture against the drawn path: the tip and the middle of the front
    edge against the nearest path point; where the tip lies in vial_lip (on the y-z plane, x = 0); and the shaft
    outside the vial against the generated tool axis at the time of the nearest point (angle, and the offset
    across the shaft where the generated axis crosses the lip plane z = 0)."""
    sh = shaft_line(caps["scoop"])
    W = reg["ecc_scoop"]["W"]
    sil = spatula_silhouette(caps["scoop"], caps["surface"], W, sh)
    tips = np.array([spatula_silhouette(caps["scoop"], caps["surface"], W, sh, thr=t)["tip"] for t in (12, 15, 20, 25)])
    uv = fr(path["tip"])
    out = {"sil": sil, "shaft": sh, "uv": uv,
           "thr_spread_px": float(max(np.linalg.norm(a - b) for a in tips for b in tips)),
           "tip": nearest_on_path(sil["tip"], uv, path["dt_s"]), "front": nearest_on_path(sil["front"], uv, path["dt_s"])}
    rel = sil["tip"] - fr.o                                    # x = 0: uv = o + scale (y ey + z ez)
    out["tip_lip_mm"] = (float(rel @ fr.ey / fr.scale), float(rel @ fr.ez / fr.scale))
    n, p = ps["T_lip_surf_mm"][:3, 2], ps["T_lip_surf_mm"][:3, 3]
    out["tip_above_surface_mm"] = float((np.r_[0.0, out["tip_lip_mm"]] - p) @ n)
    k = int(round(out["tip"]["t_s"] / path["dt_s"]))
    T3, R3 = path["tip"][k], path["root"][k]
    a, b = fr(T3)[0], fr(R3)[0]
    nrm = np.array([-sh["v"][1], sh["v"][0]])
    lip = fr(T3 + (-T3[2] / (R3[2] - T3[2])) * (R3 - T3))[0]
    out["axis"] = {"t_s": k * path["dt_s"], "vs_shaft_deg": signed_angle(sh["v"], (b - a) / np.linalg.norm(b - a)),
                   "lip_offset_px": float((lip - sh["c"]) @ nrm), "path_tip_offset_px": float((a - sh["c"]) @ nrm)}
    out["on_path"] = out["tip"]["d_px"] / fr.scale <= TIP_ON_PATH_MM
    return out


def waypoint_pose(pc: dict, path: dict, ps: dict, fr: ImageVialFrame) -> dict | None:
    """The retake waypoint (tools/scoop_waypoint_pose_tpscoop.py) as the FR3 can be sent to it: the spatula frame (the
    controller's TCP, fr3_link8 -> spatula) in vial_lip, and in fr3_link0 through the vial_lip pose the CALIB_RUN
    observer used at the presentation pose (orthonormalised).  The quaternion sign follows the runs'
    scoop_tp_gmm.approach.quat."""
    if not WAYPOINT.exists():
        return None
    w = json.loads(WAYPOINT.read_text())
    k = int(w["sample"])
    if np.abs(np.array(w["tip_vial_lip_mm"]) - path["tip"][k]).max() > 1e-3:
        raise SystemExit(f"{WAYPOINT.name} does not lie on {SCOOP_PATH.name}; regenerate it")
    cfg = json.loads(dict(zarr.open_group(str(RUNS[TOP_TRIAL["material"]]), mode="r").attrs)["config_json"])
    q_app = np.array(cfg["fr3_actions"]["scoop_tp_gmm"]["approach"]["quat"], float)
    q = np.array(w["quat_vial_lip_xyzw"], float)
    q = q if q @ q_app >= 0 else -q
    L = pc[CALIB_RUN]["lip"]
    z = L["z"] / np.linalg.norm(L["z"])
    x = np.cross(L["y"], z); x /= np.linalg.norm(x)
    T_fr3_lip = np.eye(4)
    T_fr3_lip[:3, :3] = np.c_[x, np.cross(z, x), z]
    T_fr3_lip[:3, 3] = L["p_lip"]
    T_lip_sp = T_from(np.array(w["tip_vial_lip_mm"]) / 1000.0, q)
    T = T_fr3_lip @ T_lip_sp
    q_fr3 = Rotation.from_matrix(T[:3, :3]).as_quat()
    n, p = ps["T_lip_surf_mm"][:3, 2], ps["T_lip_surf_mm"][:3, 3]
    return {"t_s": float(w["t_s"]), "k": k, "tip_mm": np.array(w["tip_vial_lip_mm"]), "q_lip": q,
            "uv": fr(path["tip"][k])[0], "above_surface_mm": float((path["tip"][k] - p) @ n),
            "vs_approach_deg": float(np.degrees((Rotation.from_quat(q_app).inv() * Rotation.from_quat(q)).magnitude())),
            "q_app": q_app, "axis_vs_bore_deg": float(w["tool_axis_vs_bore_deg"]),
            "yz_deg": float(np.degrees(np.arccos(np.clip(L["y"] @ z / np.linalg.norm(L["y"]), -1, 1)))),
            "p_fr3": T[:3, 3], "q_fr3": q_fr3, "created": pc[CALIB_RUN]["created"]}


def wall_y_at(x: float, wall: dict, clicks) -> float:
    """Image y of the inner lower wall line (inner_wall: across-axis offset s(t)) at image column x."""
    c0, c1, L, ez, ey = axis_basis(clicks)
    tt = np.linspace(0, L, 4001)
    q = c1[None] + tt[:, None] * ez[None] + np.polyval(wall["line"], tt)[:, None] * ey[None]
    return float(np.interp(x, q[:, 0], q[:, 1]))


def deposit_marks(rgb: np.ndarray, wall: dict, clicks, ref: np.ndarray, W: np.ndarray,
                  land_forward_mm: float = 6.0, scale: float = 1.0) -> dict:
    """The load on the blade (bright grains just above the shaft line, inside the vial), the blade tip
    (spatula_silhouette against vialSideView.png, ECC warp W), and the landing point of the curved arrow: on the
    inner lower wall (the source vial's wall line; same pose to ~3 px), land_forward_mm along the axis towards
    the vial base from the tip (scale: px per mm)."""
    sh = shaft_line(rgb)
    G = gray(rgb)
    up = np.array([sh["v"][1], -sh["v"][0]])
    up = up if up[1] < 0 else -up
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]
    d_up = (xx - sh["c"][0]) * up[0] + (yy - sh["c"][1]) * up[1]
    band = (d_up > -10) & (d_up < 60) & (xx > 1120) & (xx < 1335)
    load = band & (G > np.median(G[band]) + 35) & (G > 110)
    load = cv2.morphologyEx(load.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(load)
    keep = np.isin(lab, [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= 150])
    ys, xs = np.nonzero(keep)
    cx = float(xs.mean())
    # the blade's top edge under the load (the shaft line) and the wall below it, on the vertical x = cx
    y_line = sh["c"][1] + (cx - sh["c"][0]) * sh["v"][1] / sh["v"][0]
    y_wall = wall_y_at(cx, wall, clicks)
    sil = spatula_silhouette(rgb, ref, W, sh, x_range=(1060, 1340))
    ez = axis_basis(clicks)[3]
    x_land = float(sil["tip"][0] - land_forward_mm * scale * ez[0])
    return {"shaft": sh, "load_px": int(keep.sum()), "load_box": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            "load_c": (cx, float(ys.mean())), "y_blade": float(y_line), "y_wall": y_wall, "up": up, "sil": sil,
            "tip": sil["tip"], "front": sil["front"], "land": np.array([x_land, wall_y_at(x_land, wall, clicks)]),
            "land_forward_mm": land_forward_mm}


# ============================================================================= drawing helpers
def resample(img: np.ndarray, box, w_in: float, h_in: float) -> np.ndarray:
    x0, y0, x1, y1 = box
    px = (max(1, int(round(w_in * PHOTO_PPI))), max(1, int(round(h_in * PHOTO_PPI))))
    return np.asarray(Image.fromarray(np.ascontiguousarray(img[y0:y1, x0:x1])).resize(px, Image.LANCZOS))


def mask_outline(ax, mask, color, lw=None, z=3.5, min_frac=None):
    """Draw the outer boundary of each connected component of a mask (image pixel coordinates). Components smaller
    than min_frac of the mask's area are left unoutlined so isolated specks do not dominate; they stay in the fill."""
    lw = MASK_OUTLINE_LW if lw is None else lw
    min_frac = MASK_OUTLINE_MIN_FRAC if min_frac is None else min_frac
    m = np.ascontiguousarray(mask.astype(np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    total = float(m.sum()) or 1.0
    hexc = "#%02x%02x%02x" % tuple(int(v) for v in color) if not isinstance(color, str) else color
    for c in cs:
        if cv2.contourArea(c) < min_frac * total:
            continue
        q = c[:, 0, :].astype(float)
        ax.plot(np.r_[q[:, 0], q[0, 0]], np.r_[q[:, 1], q[0, 1]], color=hexc, lw=lw, zorder=z,
                solid_joinstyle="round", solid_capstyle="round")


def overlay(rgb, mask, color, alpha):
    out = rgb.astype(float)
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(color, float)
    return out.astype(np.uint8)


def image_axes(fig, rect, img, box, W, H):
    """Axes showing img[box] resampled to PHOTO_PPI at its printed size, in the box's pixel coordinates."""
    ax = fig.add_axes([rect[0] / W, rect[1] / H, rect[2] / W, rect[3] / H])
    x0, y0, x1, y1 = box
    ax.imshow(resample(img, box, rect[2], rect[3]), extent=(x0, x1, y1, y0), interpolation="none", aspect="auto")
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(MUTED); s.set_linewidth(0.5)
    return ax


def crop_to_aspect(box, aspect, size=(IMG_W, IMG_H), grow=False):
    """Shrink box about its centre to width / height = aspect (grow=True: grow it, so that the box stays inside
    the crop), inside the image."""
    x0, y0, x1, y1 = box
    cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    if (w / h > aspect) != grow:
        w = h * aspect
    else:
        h = w / aspect
    if grow and (w > size[0] or h > size[1]):
        s = min(size[0] / w, size[1] / h)
        w, h = w * s, h * s
    w, h = min(w, size[0]), min(h, size[1])
    x0, y0 = int(round(min(max(cx - w / 2, 0), size[0] - w))), int(round(min(max(cy - h / 2, 0), size[1] - h)))
    return (x0, y0, x0 + int(round(w)), y0 + int(round(h)))


def arrow(ax, a, b, color, lw=0.9, ms=6.0, z=8, style="-|>", **kw):
    ax.add_patch(FancyArrowPatch(tuple(a), tuple(b), arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 zorder=z, shrinkA=0, shrinkB=0, **kw))


def into_page(ax, o, r_px, color, z=9):
    """The out-of-plane x axis, pointing into the page: a circled cross."""
    ax.add_patch(Circle(tuple(o), r_px, fc="white", ec=color, lw=0.8, zorder=z))
    d = r_px * 0.62
    for s in (1, -1):
        ax.plot([o[0] - d, o[0] + d], [o[1] - s * d, o[1] + s * d], color=color, lw=0.8, zorder=z + 0.1,
                solid_capstyle="butt")


def triad(ax, fr: ImageVialFrame, origin_mm, L_mm, color, label, label_off_mm, axis_labels, px_per_pt):
    o = fr(origin_mm)[0]
    for k, name in ((2, "z"), (1, "y")):
        e = np.array(origin_mm, float)
        e[k] += L_mm
        tip = fr(e)[0]
        arrow(ax, o, tip, color=color, lw=1.0, ms=5.5, z=9, path_effects=HALO_W)
        if axis_labels:
            off = (tip - o) / np.linalg.norm(tip - o) * 5.5 * px_per_pt
            ax.text(*(tip + off), f"${name}$", color=color, fontsize=FS, ha="center", va="center", zorder=10,
                    path_effects=HALO_W)
    into_page(ax, o, 2.0 * px_per_pt, color)
    q = fr(np.array(origin_mm, float) + np.r_[0.0, label_off_mm])[0]
    ax.text(*q, label, color=color, fontsize=FS, ha="center", va="center", zorder=10, path_effects=HALO_W)
    return o


def text_box(fig, rect, W, H, lines, kind="balance"):
    ax = fig.add_axes([rect[0] / W, rect[1] / H, rect[2] / W, rect[3] / H])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0.04, 0.03), 0.92, 0.94, boxstyle="round,pad=0,rounding_size=0.08",
                                transform=ax.transAxes, fc=KIND[kind] + "1f", ec=KIND[kind], lw=0.8,
                                mutation_aspect=rect[2] / rect[3]))
    ax.text(0.5, 0.5, "\n".join(lines), ha="center", va="center", color=INK, fontsize=FS, linespacing=1.15,
            transform=ax.transAxes)
    return ax


def label_box(ax, x, y, s, ha="left", va="top", **kw):
    ax.text(x, y, s, color="white", fontsize=FS, ha=ha, va=va, zorder=12,
            bbox=dict(boxstyle="square,pad=0.15", fc="#000000", ec="none", alpha=0.55), **kw)


# ============================================================================= panels
def px_per_pt(rect, box) -> float:
    """Image pixels per typographic point in a panel (for offsets given in points)."""
    return (box[2] - box[0]) / (rect[2] * 72.0)


def panel_photo(fig, rect, W, H, photo, box0, tag=None):
    """A phone photo cropped about box0 to the panel's aspect, with an optional tag at the lower left."""
    a = rect[2] / rect[3]
    box = crop_to_aspect(box0 if box0 is not None else (0, 0, photo.shape[1], photo.shape[0]), a,
                         size=(photo.shape[1], photo.shape[0]), grow=box0 is not None)
    ax = image_axes(fig, rect, photo, box, W, H)
    if tag:
        label_box(ax, box[0] + 0.05 * (box[2] - box[0]), box[3] - 0.03 * (box[3] - box[1]), tag, va="bottom")
    return {"box": box}


def panel_surface(fig, rect, W, H, caps, ps, clicks, box):
    ax = image_axes(fig, rect, overlay(caps["surface"], caps["masks"]["surface"], SIDE_MASK, 0.40), box, W, H)
    mask_outline(ax, caps["masks"]["surface"], SIDE_MASK)
    k = px_per_pt(rect, box)
    c, v, p, n = ps["c"], ps["v"], ps["p_img"], ps["n_img"]
    t = np.linspace(-2000, 2000, 2)
    ax.plot(c[0] + t * v[0], c[1] + t * v[1], color=SURF_RED, lw=SURF_LINE_LW_B, zorder=4)
    a, b = clicks
    ax.plot([a[0], b[0]], [a[1], b[1]], color="white", lw=0.7, ls=(0, (3, 2)), zorder=3)
    L = 7.0 * ImageVialFrame(clicks).scale                     # n drawn 7 mm long
    arrow(ax, p, p + L * n, color="white", lw=0.9, ms=6, z=6)
    ax.text(*(p + L * n + 5.0 * k * n), r"$\mathbf{n}$", color="white", fontsize=FS, ha="center", va="center", zorder=7,
            path_effects=HALO)
    ax.plot(*p, "o", ms=3.4, mfc=SURF_RED, mec="white", mew=0.6, zorder=8)
    q = p - 9.0 * k * n
    ax.text(*q, r"$\mathbf{p}$", color="white", fontsize=FS, ha="center", va="center", zorder=8, path_effects=HALO)
    # 5 mm scale bar (at the depth of the vial axis), on the backdrop right of the neck, below the lower clip
    s5 = 5.0 * ImageVialFrame(clicks).scale
    x0_, y0_ = SCALE_BAR_XY
    ax.plot([x0_, x0_ + s5], [y0_, y0_], color="white", lw=1.4, solid_capstyle="butt", zorder=9)
    ax.text(x0_ + s5 / 2, y0_ + 2.0 * k, "5 mm", color="white", fontsize=FS, ha="center", va="top", zorder=9,
            path_effects=HALO)
    if caps["provisional"]:                                    # a mask from a model other than SAM 3
        ax.text(box[0] + 0.03 * (box[2] - box[0]), box[1] + 0.05 * (box[3] - box[1]),
                f"provisional ({caps['mask_model'].upper().replace('SAM', 'SAM ')})", color="#c00000",
                fontsize=FS, fontweight="bold", ha="left", va="top", zorder=12,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
    return ax


def panel_scoop(fig, rect, W, H, caps, ps, fr, contact_mm, path, box, draw_frames, clicks, r_in, lay, spat):
    ax = image_axes(fig, rect, caps["scoop"], box, W, H)
    k = px_per_pt(rect, box)
    c, v = ps["c"], ps["v"]
    _, c1, _, _, ey = axis_basis(clicks)
    a_, b_ = (c - c1) @ ey, v @ ey                            # the surface line inside the vial only
    t = np.sort([(-r_in - a_) / b_, (r_in - a_) / b_])
    ax.plot(c[0] + t * v[0], c[1] + t * v[1], color=SURF_RED, lw=0.8, ls=(0, (4, 2)), zorder=4)
    info = {}
    if draw_frames:
        uv = fr(path["tip"])
        ax.plot(uv[:, 0], uv[:, 1], color=PATH_C, lw=1.2, zorder=6, solid_capstyle="round", path_effects=HALO)
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(uv, axis=0), axis=1))]
        for f in lay["path_arrows"]:                           # direction of travel (no time marks)
            j = min(max(int(np.searchsorted(seg, f * seg[-1])), 1), len(uv) - 2)
            arrow(ax, uv[j - 1], uv[j + 1], color=PATH_C, lw=0, ms=7, z=7)
        info["path_uv"] = uv
        triad(ax, fr, [0.0, 0.0, 0.0], 5.0, C_VIAL, LABEL_VIAL, lay["v_label_mm"], True, k)
        triad(ax, fr, contact_mm, 4.0, C_CON, LABEL_CON, lay["c_label_mm"], False, k)
        if spat is not None and spat["on_path"]:               # the photo's tip lies on the path: mark the spot
            ax.add_patch(Circle(tuple(spat["tip"]["uv"]), 3.0 * k, fc="none", ec="white", lw=0.9, zorder=9))
        if lay["sz_label_mm"] is not None:
            q = fr(contact_mm + np.r_[0.0, lay["sz_label_mm"]])[0]
            ax.text(*q, rf"$s_z = {SZ_MM:+.0f}$ mm".replace("-", "−"), color=C_CON, fontsize=FS, ha="center",
                    va="center", zorder=10, path_effects=HALO_W)   # styled as the F_contact point label (no box)
    return ax, info


def panel_top(fig, rect, W, H, tr, margin_px):
    m = tr["top_mask"]
    ys, xs = np.nonzero(m)
    a = rect[2] / rect[3]
    cy = (ys.min() + ys.max()) / 2
    w = max(np.ptp(xs) + 2 * margin_px[0], margin_px[1])
    h = w / a
    if h < np.ptp(ys) + 200:
        h = np.ptp(ys) + 200
        w = h * a
    x0, y0 = int(max(xs.max() - 0.72 * w, 0)), int(max(cy - h / 2, 0))
    box = (x0, y0, int(min(x0 + w, IMG_W)), int(min(y0 + h, IMG_H)))
    ax = image_axes(fig, rect, overlay(tr["top"], m, TOP_MASK, 0.45), box, W, H)
    mask_outline(ax, m, TOP_MASK)
    label_box(ax, box[0] + 0.04 * (box[2] - box[0]), box[1] + 0.05 * (box[3] - box[1]),
              f"{tr['carried_mg']:.1f} mg carried")
    return {"box": box}


def panel_deposit(fig, rect, W, H, caps, dep, fr, box, n_taps):
    ax = image_axes(fig, rect, caps["deposit"], box, W, H)
    k = px_per_pt(rect, box)
    # material leaving the blade: a quadratic Bezier from just ahead of the blade tip, leaving forward along the
    # vial axis (towards the base) and ending vertical (falling) above the lower wall at the landing point
    fwd = -fr.ez                                               # towards the vial base
    p0 = dep["front"] + 2.5 * k * fwd
    p2 = dep["land"] - np.array([0.0, 4.0 * k])
    p1 = p0 + fwd * (p2[0] - p0[0]) / fwd[0]                   # on the forward line, above p2
    ax.add_patch(FancyArrowPatch(path=MplPath([p0, p1, p2], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]),
                                 arrowstyle="-|>", mutation_scale=8, color="white", lw=1.2, zorder=8,
                                 path_effects=HALO))
    # the taps: the spatula strikes along -y_lip (deposit config), drawn on the shaft outside the vial
    up = fr.ey
    sh = dep["shaft"]
    xt = TAP_MARK_X
    base = np.array([xt, sh["c"][1] + (xt - sh["c"][0]) * sh["v"][1] / sh["v"][0]])
    L = 6.0 * k
    arrow(ax, base - L * up, base + L * up, color=PATH_C, lw=1.1, ms=6, z=8, style="<|-|>", path_effects=HALO)
    ax.text(base[0] + 1.5 * k, base[1] - L * up[1] + 1.5 * k, f"{n_taps} taps", color="white", fontsize=FS,
            ha="right", va="top", zorder=9, path_effects=HALO)
    return ax


# ============================================================================= layouts
LAYOUTS = {                                    # two rows
    "2row": dict(
        rows=[["A", "B", "C"], ["D", "V", "E", "F"]], gap=0.085, row_gap=0.07, title_dx=0.13,
        crop=dict(B=SIDE_CROP, C=SIDE_CROP, E=SIDE_CROP),
        box_w_mm=22.0,                         # A's two boxes and the change-vials box ("source -> goal" is 17.1 mm)
        photo_aspect=dict(A=0.55, V=0.55, F=0.55),   # F: the weighing photo (portrait), widened to fit its title;
                                                     # A and V when their photos land
        solve="D",                             # D takes the width left, so that row 2 is as tall as row 1 and
                                               # B, C and E print at the same size from the same window
        d_margin=(140, 560),                   # D: >= 560 camera px wide, so that the blade prints at about
                                               # 870 px over 72.5 mm and above 300 ppi
        weigh_box=(900, 360, 2460, 3400),      # F (and A): the cover, the vial and the balance's top plate (photo
                                               # px), kept whole: the crop grows about it to the panel's aspect
        titles=dict(A="Tare and weigh", B="Estimate surface", C="Scoop", D="Photograph spatula", E="Deposit",
                    F="Weigh both vials"),
        a_lines=(["Tare", "goal vial"], ["Weigh", "source vial"]),   # the order of sec:benchmarking
        v_lines=["Change vials", "source → goal"],
        legend="title_row",
        v_label_mm=(-3.4, 2.6), c_label_mm=(-3.0, -3.4), sz_label_mm=(-7.0, -1.5),
        path_arrows=(0.22, 0.62, 0.9)),
}
LAYOUTS["2row_G"] = dict(                     # a cleaning step G after F. To keep B, C and E at
    LAYOUTS["2row"], rows=[["A", "B", "C"], ["D", "E", "F", "G"]],   # full size, the change-vials box becomes a
    photo_aspect=dict(LAYOUTS["2row"]["photo_aspect"], G=0.55),        # label on the D -> E arrow
    titles=dict(LAYOUTS["2row"]["titles"], G="Clean spatula"),
    gap_after=dict(D=0.40),                    # inches: room for the arrow label between D and E
    arrow_labels={("D", "E"): ["Change", "vials"]})
KIND_OF = dict(A="balance", B="vision", C="motion", D="vision", V="motion", E="motion", F="balance", G="motion")
UNLETTERED = ("V",)                            # the change-vials step: no letter, title or kind bar


def text_width_in(s: str, fontsize: float = FS) -> float:
    """Printed width of a one-line text, inches."""
    fig = plt.figure(figsize=(2, 1))
    t = fig.text(0, 0, s, fontsize=fontsize)
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    plt.close(fig)
    return w


def layout(lay, W, photos: dict):
    """Rects (x, y, w, h) in inches for the image areas, the figure height and the aspects used.  The side
    panels take the aspect of their crop; A and V are text columns of width box_w_mm unless a photo is given;
    a photo panel (F, and A or V when given) takes photo_aspect, widened if its title would not fit; D takes
    the width left in row 2, so that both rows have the height of row 1."""
    title_h, bottom = 0.20, 0.02 + (0.13 if lay.get("legend") == "bottom" else 0.0)
    box_w = lay["box_w_mm"] / 25.4

    def need(key):                                             # width the letter and title take
        return 0.0 if key in UNLETTERED else lay["title_dx"] + text_width_in(lay["titles"][key]) + 0.03

    def width(key, h):
        if key in lay["crop"]:
            x0, y0, x1, y1 = lay["crop"][key]
            return (x1 - x0) / (y1 - y0) * h
        if key in ("A", "V") and not photos.get(key):
            return max(box_w, need(key))
        return max(lay["photo_aspect"][key] * h, need(key))

    r0, r1 = lay["rows"]
    def gap_of(k):                                             # the gap after panel k
        return lay.get("gap_after", {}).get(k, lay["gap"])
    def gaps(r):
        return sum(gap_of(k) for k in r[:-1])
    h = 1.4
    if lay["solve"] == "A":                                    # 2row_G: row 2 sets the height, D held at d_min_mm
        dmin = lay["d_min_mm"] / 25.4
        for _ in range(8):
            fixed = sum(width(k, h) for k in r1 if k not in lay["crop"] and k != "D") + dmin
            h = (W - gaps(r1) - fixed) / sum((c[2] - c[0]) / (c[3] - c[1]) for k, c in lay["crop"].items() if k in r1)
        widths = {k: width(k, h) for r in lay["rows"] for k in r if k not in ("A", "D")}
        widths["D"] = dmin
        widths["A"] = W - gaps(r0) - sum(widths[k] for k in r0 if k != "A")
        if widths["A"] < need("A"):
            print(f"WARNING: A is {widths['A'] * 25.4:.1f} mm, narrower than its title needs")
    else:
        for _ in range(4):                                     # row 1 sets the height (fixed point; titles)
            fixed = sum(width(k, h) for k in r0 if k not in lay["crop"])
            h = (W - gaps(r0) - fixed) / sum((c[2] - c[0]) / (c[3] - c[1]) for k, c in lay["crop"].items() if k in r0)
        widths = {k: width(k, h) for r in lay["rows"] for k in r if k != lay["solve"]}
        widths[lay["solve"]] = W - gaps(r1) - sum(widths[k] for k in r1 if k != lay["solve"])
    H = 2 * h + 2 * title_h + lay["row_gap"] + bottom
    rects, y_top = {}, H
    for r in lay["rows"]:
        x = 0.0
        for key in r:
            rects[key] = (x, y_top - title_h - h, widths[key], h)
            x += widths[key] + gap_of(key)
        y_top -= title_h + h + lay["row_gap"]
    asp = {k: w / h for k, w in widths.items()}
    return rects, H, title_h, asp


def text_column(fig, rect, W, H, lines_list, kind, gap_in=0.05):
    """Text boxes stacked in a panel's image area, each (h - gaps) / n tall; returns the box height."""
    x, y, w, h = rect
    n = len(lines_list)
    hh = (h - gap_in * (n - 1)) / n
    for i, lines in enumerate(lines_list):
        text_box(fig, (x, y + (n - 1 - i) * (hh + gap_in), w, hh), W, H, lines, kind=kind)
    return hh


def render(name, lay, out_dir, data) -> dict:
    caps, ps, fr, tr = data["caps"], data["ps"], data["fr"], data["tr"]
    W = FIG_W_IN
    photos = {"A": data["photo_a"] is not None, "V": data["photo_v"] is not None}
    rects, H, title_h, asp = layout(lay, W, photos)
    fig = plt.figure(figsize=(W, H))
    info = {"rects": rects, "H": H, "aspects": asp, "photos": photos}
    if photos["A"]:
        info["A"] = panel_photo(fig, rects["A"], W, H, data["photo_a"], lay["weigh_box"],
                                SOURCE_WEIGH_PHOTO_VIAL if SHOW_PHOTO_TAGS else None)
    hb = text_column(fig, rects["A"], W, H, lay["a_lines"], "balance") if not photos["A"] else \
        (rects["A"][3] - 0.05) / 2
    panel_surface(fig, rects["B"], W, H, caps, ps, data["clicks"], lay["crop"]["B"])
    _, info["C"] = panel_scoop(fig, rects["C"], W, H, caps, ps, fr, data["contact_mm"], data["path"], lay["crop"]["C"],
                               data["reg"]["pass"], data["clicks"], -data["reg"]["wall_new"]["base_px"], lay,
                               data["spat"])
    info["D"] = panel_top(fig, rects["D"], W, H, tr, lay["d_margin"])
    if "V" in rects:
        x, y, w, h = rects["V"]
        if photos["V"]:
            info["V"] = panel_photo(fig, rects["V"], W, H, data["photo_v"], CHANGE_PHOTO_BOX, "change vials")
        else:                                                  # one box, the size of a weighing box, centred
            text_box(fig, (x, y + (h - hb) / 2, w, hb), W, H, lay["v_lines"], kind="motion")
        info["V_box"] = (w, hb)
    panel_deposit(fig, rects["E"], W, H, caps, data["dep"], fr, lay["crop"]["E"], TAP["n_taps"])
    info["F"] = panel_photo(fig, rects["F"], W, H, data["photo"], lay["weigh_box"],
                            WEIGH_PHOTO_VIAL if SHOW_PHOTO_TAGS else None)
    if "G" in rects:                                           # cleaning (companion photograph)
        info["G"] = panel_photo(fig, rects["G"], W, H, data["photo_g"], CLEAN_PHOTO_BOX, None)
    titles = {}
    for key, r in rects.items():                               # letter, title, kind bar
        if key in UNLETTERED:
            continue
        x, y, w, h = r
        yt = y + h + title_h
        fig.text(x / W, (yt - 0.02) / H, key, fontsize=9, fontweight="bold", ha="left", va="top", color=INK)
        titles[key] = fig.text((x + lay["title_dx"]) / W, (yt - 0.035) / H, lay["titles"][key], fontsize=FS,
                               ha="left", va="top", color=INK)
        fig.patches.append(Rectangle((x / W, (y + h + 0.035) / H), w / W, 0.022 / H, transform=fig.transFigure,
                                     fc=KIND[KIND_OF[key]], ec="none"))
    rend = fig.canvas.get_renderer()
    info["title_slack_mm"] = {k: (rects[k][0] + rects[k][2] - t.get_window_extent(rend).x1 / fig.dpi) * 25.4
                              for k, t in titles.items()}           # > 0: the title ends inside its panel
    x, y, w, h = rects["C"]                                    # kind legend, right end of the first title row
    yl, xr = y + h + title_h - 0.083, W - 0.01
    if lay.get("legend") == "bottom":                          # or, right-aligned under the second row
        yl = 0.075
    for kind, txt in reversed((("balance", "balance"), ("vision", "vision"), ("motion", "robot motion"))):
        t = fig.text(xr / W, yl / H, txt, fontsize=FS, ha="right", va="center", color=INK)
        tw = t.get_window_extent(rend).width / fig.dpi
        fig.patches.append(Rectangle(((xr - tw - 0.12) / W, (yl - 0.035) / H), 0.07 / W, 0.07 / H,
                                     transform=fig.transFigure, fc=KIND[kind], ec="none"))
        xr -= tw + 0.13 + 0.14
    order = [k for r in lay["rows"] for k in r]                # order arrows in the gaps (same row only)
    for k1, k2 in zip(order, order[1:]):
        x1, y1, w1, h1 = rects[k1]
        x2, y2, _, _ = rects[k2]
        if abs(y1 - y2) < 1e-6:
            xm, ym = (x1 + w1 + x2) / 2, y1 + h1 / 2
            label = lay.get("arrow_labels", {}).get((k1, k2))
            if label:                                          # a step without a panel: arrow in its kind colour
                half = (x2 - x1 - w1) / 2 - 0.03
                fig.patches.append(FancyArrowPatch(((xm - half) / W, ym / H), ((xm + half) / W, ym / H),
                                                   transform=fig.transFigure, arrowstyle="-|>", mutation_scale=7,
                                                   color=KIND["motion"], lw=1.1))
                fig.text(xm / W, (ym + 0.05) / H, "\n".join(label), fontsize=FS, ha="center", va="bottom",
                         color=INK, linespacing=1.1)
            else:
                fig.patches.append(FancyArrowPatch(((xm - 0.022) / W, ym / H), ((xm + 0.026) / W, ym / H),
                                                   transform=fig.transFigure, arrowstyle="-|>", mutation_scale=5,
                                                   color=INK2, lw=0.6))
    # overlap check of the in-figure text (titles and legend against each other)
    boxes = [t.get_window_extent(rend) for t in fig.texts]
    info["text_overlaps"] = sum(boxes[i].overlaps(boxes[j]) for i in range(len(boxes)) for j in range(i + 1, len(boxes)))
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"Title": None, "Author": None, "Subject": None, "Keywords": None, "Creator": None, "Producer": None}
    fig.savefig(out_dir / f"fig_cycle_{name}.pdf", dpi=PHOTO_PPI, metadata=meta)
    fig.savefig(out_dir / f"fig_cycle_{name}.png", dpi=600, metadata={"Software": None})
    plt.close(fig)
    if name == MAIN_LAYOUT:                                    # fig_cycle.* (in the paper) = the main layout only
        for ext in ("pdf", "png"):
            shutil.copyfile(out_dir / f"fig_cycle_{name}.{ext}", out_dir / f"fig_cycle.{ext}")
    return info


# ============================================================================= main
def main() -> None:
    global VIAL_CLICKS_TXT, CLEAN_PHOTO
    args = sys.argv[1:]
    if "--clean-photo" in args:                                # renders with another photo for G
        k = args.index("--clean-photo"); CLEAN_PHOTO = Path(args[k + 1]).resolve(); del args[k:k + 2]
    check_dir = None
    if "--check" in args:
        k = args.index("--check"); check_dir = Path(args[k + 1]); del args[k:k + 2]
    layout_name = MAIN_LAYOUT
    if "--layout" in args:
        k = args.index("--layout")
        if args[k + 1] not in ("2row", "2row_G"):
            raise SystemExit("layouts: 2row, 2row_G")
        layout_name = args[k + 1]
        del args[k:k + 2]
    out = Path(args[0]) if args else IMAGES

    pc = projection_check()
    print_projection_check(pc)
    tr = load_trial()
    caps = load_captures()
    clicks = np.array(json.loads((DATA / "salt_run_meta.json").read_text())["vial_clicks_px"], float)
    if not np.allclose(clicks, tr["clicks"]):
        raise SystemExit("the stored clicks of the panel-D trial differ from the run config")
    VIAL_CLICKS_TXT = f"({clicks[0, 0]:.0f}, {clicks[0, 1]:.0f}) lip / ({clicks[1, 0]:.0f}, {clicks[1, 1]:.0f}) base"

    # ---------------------------------------------------------------- segmentation provenance
    ml = caps.get("mask_log", {})
    sv = ml.get("images", {}).get(CAPTURES["surface"].name, {})
    print(f"\nMasks of the 2026-10-02 captures: {caps['mask_model']} ({ml.get('repo')} @ {ml.get('revision')}, "
          f"{ml.get('model_class')}, transformers {ml.get('transformers')}, torch {ml.get('torch')}, {ml.get('device')}); "
          f"prompts + {ml.get('prompts', {}).get('positive_xy')} - {ml.get('prompts', {}).get('negative_xy')}; "
          f"vialSideView.png: candidate #{sv.get('chosen')} of scores {np.round(sv.get('scores', []), 4).tolist()} "
          f"(gap {sv.get('score_gap_top2', float('nan')):.4f}), {sv.get('area_px')} px"
          f"{'  PROVISIONAL (not SAM 3)' if caps['provisional'] else ''}")
    for val in ml.get("validation", []):
        ch = val["candidates"][val["chosen"]]
        bm = val["candidates"][val["best_matching_candidate"]]
        print(f"  validation {val['run']} trial {val['trial']}: chosen #{val['chosen']} (gap {val['score_gap_top2']:.4f}"
              f"{', near tie' if val['near_tie'] else ''}) IoU {ch['iou']:.4f}, line "
              + (f"{ch['line_angle_diff_deg']:+.3f} deg, p {ch['p_offset_along_axis_mm']:+.3f} mm" if "line_angle_diff_deg"
                 in ch else "fit failed")
              + f"; best match #{val['best_matching_candidate']} IoU {bm['iou']:.4f}, line "
              + (f"{bm['line_angle_diff_deg']:+.3f} deg, p {bm['p_offset_along_axis_mm']:+.3f} mm" if "line_angle_diff_deg"
                 in bm else "fit failed"))

    # ---------------------------------------------------------------- surface and contact point
    chk = check_plane_pose()
    print("\nplane_pose() against the stored T_vialLip_surface / plane_center_image, all trials (max abs): " +
          "; ".join(f"{m}: {e[0]:.1e} deg, {e[1]:.1e} mm, {e[2]:.1e} px" for m, e in chk.items()))
    ps = plane_pose(caps["masks"]["surface"], clicks)
    contact = contact_point_mm(ps, SZ_MM)
    n_lip = ps["T_lip_surf_mm"][:3, 2]
    print(f"Surface on vialSideView.png: line {np.degrees(np.arctan2(ps['v'][1], ps['v'][0])):+.3f} deg in the image, "
          f"p at ({ps['p_img'][0]:.2f}, {ps['p_img'][1]:.2f}) px, {-ps['z_mm']:.2f} mm below the lip along the axis "
          f"(z = {ps['z_mm']:+.3f} mm), tilt theta = {np.degrees(ps['theta']):+.3f} deg about x_lip, n = "
          f"{np.round(n_lip, 4).tolist()} in vial_lip; {int(ps['inl'].sum())}/{len(ps['pts'])} RANSAC inliers")
    print(f"  stored surfaces, {TOP_TRIAL['material']} run: z = {np.nanmedian(rows_to_arrays(extract(RUNS[TOP_TRIAL['material']])[0])['surface_z_lip_mm']):+.2f} mm (median)")
    print(f"{{C}} for s_z = {SZ_MM:+.1f} mm: p + s_z n = {np.round(contact, 3).tolist()} mm in vial_lip "
          f"(|{SZ_MM:+.1f}| mm along n; {contact[2] - ps['z_mm']:+.3f} mm along the axis, {contact[1]:+.3f} mm along y)")

    # ---------------------------------------------------------------- registration and frames
    reg = registration(caps, tr, clicks)
    print_registration(reg, tr)
    cal, T_cam_lip = calib_frame(pc)
    y_sign = float(np.sign((T_cam_lip[:3, 1][:2]) @ axis_basis(clicks)[4]))
    fr = ImageVialFrame(clicks, y_sign=y_sign)
    path = load_path(contact)
    xc = calib_crosscheck(fr, cal, T_cam_lip, contact, path["tip"])
    print(f"\n{{V}} on the image: origin click 0 ({fr.o[0]:.0f}, {fr.o[1]:.0f}), z along the click axis, y "
          f"{'up' if y_sign > 0 else 'down'} (sign from the calibration), x into the page (calibration: x_lip "
          f"{xc['x_tilt_deg']:.1f} deg from the optical axis, {'away from' if xc['x_sign'] > 0 else 'towards'} the camera); "
          f"{fr.scale:.3f} px/mm, axis depth fx / scale = {fr.Z0:.1f} mm")
    print(f"  calibration cross-check ({CALIB_RUN} observer's vial_lip, world_to_camera_43_sensor, K43; its projection "
          f"test max {pc[CALIB_RUN]['max']:.1f} px): origin at ({xc['origin_cal'][0]:.1f}, {xc['origin_cal'][1]:.1f}) px, "
          f"{np.linalg.norm(xc['origin_d']):.1f} px from click 0 ({xc['origin_along']:+.1f} along the axis, "
          f"{xc['origin_across']:+.1f} across); z axis {xc['z_angle_deg']:+.2f} deg, y axis {xc['y_angle_deg']:+.2f} deg "
          f"from the image frame's; scale {xc['z_scale']:.4f} (z), {xc['y_scale']:.4f} (y) of the click scale; "
          f"{{C}} {xc['contact_d']:.1f} px apart; the path {xc['path_d_median']:.1f} px apart (median), "
          f"{xc['path_d_max']:.1f} px (max)")
    meta = path["model"]["meta"]
    print(f"\nGenerated path ({SCOOP_PATH.name}): {path['model']['path']}, sha256 {path['model']['sha256'][:12]}..., code_rev "
          f"{meta.get('code_rev')}, trained {meta.get('trained_at')}, {meta.get('dataset')}, "
          f"{path['model']['n_demonstrations']} demonstrations, {meta.get('n_states')} states, chart {meta.get('chart')}; "
          f"repo {path['repo']['branch']} @ {path['repo']['commit'][:7]} (package clean: {not path['repo']['package_dirty']})")
    print(f"  {path['n']} samples at {path['dt_s']} s ({(path['n'] - 1) * path['dt_s']:.2f} s), no time marks, arrowheads at "
          f"{LAYOUTS['2row']['path_arrows']} of the arc length; "
          f"converged {path['converged']}, contained {path['contained']} ({path['rounds']} round); closest to the contact "
          f"point {path['closest_to_contact_mm']:.3f} mm at {path['closest_t_s']:.2f} s; deepest z {path['deepest_z_mm']:.2f} "
          f"mm at {path['deepest_t_s']:.2f} s ({path['deepest_z_mm'] - contact[2]:+.2f} mm along the axis from the contact "
          f"point); tip x (out of plane) {path['tip'][:, 0].min():.2f} to {path['tip'][:, 0].max():.2f} mm; starts at "
          f"z = {path['tip'][0, 2]:.2f} mm (inside the vial), ends at z = {path['tip'][-1, 2]:.2f} mm; vial-pose "
          f"invariance {path['vial_pose_invariance_mm']:.1e} mm")
    ent = entry_check(caps, fr, path)
    sh = ent["shaft"]
    print(f"Entry angle: spatula shaft in {CAPTURES['scoop'].name} (outside the vial, {sh['n']} columns, rms "
          f"{sh['rms']:.2f} px) at {sh['angle_deg']:+.2f} deg in the image, {ent['shaft_vs_axis']:+.2f} deg from the vial "
          f"axis (positive = towards +y); generated blade axis {ent['blade_start']['vs_shaft']:+.2f} deg from the shaft at "
          f"t = 0 s and {ent['blade_contact']['vs_shaft']:+.2f} deg at the closest approach "
          f"(t = {ent['blade_contact']['t_s']:.2f} s), i.e. {ent['blade_start']['vs_axis']:+.2f} / "
          f"{ent['blade_contact']['vs_axis']:+.2f} deg from the axis; the path's entry chord (t = 0 to "
          f"{ent['blade_contact']['t_s']:.2f} s) {ent['chord']['vs_shaft']:+.2f} deg from the shaft")
    spat = spatula_check(caps, reg, fr, path, ps)
    si, st_, sf, sa = spat["sil"], spat["tip"], spat["front"], spat["axis"]
    print(f"\nSpatula of {CAPTURES['scoop'].name} against the drawn path (background subtraction against "
          f"{CAPTURES['surface'].name}, ECC warp, threshold {SPATULA_DIFF_THR:.0f} grey levels; blade {si['area']} px):")
    print(f"  tip (extreme along the shaft axis) at ({si['tip'][0]:.0f}, {si['tip'][1]:.0f}) px, {si['tip_across_px']:+.1f} px "
          f"across the shaft line (the blade face widens below it towards the tip); thresholds 12-25 move it by at most "
          f"{spat['thr_spread_px']:.1f} px; in vial_lip (on the y-z plane) y = {spat['tip_lip_mm'][0]:+.2f} mm, "
          f"z = {spat['tip_lip_mm'][1]:+.2f} mm, {spat['tip_above_surface_mm']:+.2f} mm above the estimated surface "
          f"along n")
    print(f"  nearest path point: {st_['d_px']:.1f} px = {st_['d_px'] / fr.scale:.2f} mm (at the vial axis depth, "
          f"{fr.scale:.3f} px/mm), at t = {st_['t_s']:.2f} s, ({st_['uv'][0]:.1f}, {st_['uv'][1]:.1f}) px; the middle "
          f"of the front edge ({si['front'][0]:.0f}, {si['front'][1]:.0f}) -> path: {sf['d_px']:.1f} px = "
          f"{sf['d_px'] / fr.scale:.2f} mm at t = {sf['t_s']:.2f} s (front edge {si['front_span_px']:.1f} px across)")
    print(f"  shaft outside the vial against the generated tool axis at t = {sa['t_s']:.2f} s: "
          f"{sa['vs_shaft_deg']:+.2f} deg; the generated axis crosses the lip plane {sa['lip_offset_px']:+.1f} px "
          f"({sa['lip_offset_px'] / fr.scale:+.2f} mm) across the measured shaft line")
    print(f"  -> {'ON the path (within ' if spat['on_path'] else 'NOT on the path (more than '}"
          f"{TIP_ON_PATH_MM:.1f} mm): {'spot marked in C' if spat['on_path'] else 'retake C with the FR3 paused on a waypoint'}")
    wp = waypoint_pose(pc, path, ps, fr)
    if wp is not None:
        print(f"  retake waypoint ({WAYPOINT.name}, tools/scoop_waypoint_pose_tpscoop.py): t = {wp['t_s']:.2f} s (sample "
              f"{wp['k']}), tip {np.round(wp['tip_mm'], 3).tolist()} mm in vial_lip, at ({wp['uv'][0]:.0f}, "
              f"{wp['uv'][1]:.0f}) px, {wp['above_surface_mm']:+.2f} mm above the estimated surface; tool axis "
              f"{wp['axis_vs_bore_deg']:.2f} deg from the bore; orientation {wp['vs_approach_deg']:.2f} deg from the "
              f"runs' scoop_tp_gmm.approach.quat {wp['q_app'].tolist()}")
        p_, q_ = wp["tip_mm"] / 1000.0, wp["q_lip"]
        print(f"    spatula frame (TCP) in vial_lip:  {{pos: [{p_[0]:.6f}, {p_[1]:.6f}, {p_[2]:.6f}], quat: "
              f"[{q_[0]:.6f}, {q_[1]:.6f}, {q_[2]:.6f}, {q_[3]:.6f}], coordinate_frame: vial_lip}}")
        p_, q_ = wp["p_fr3"], wp["q_fr3"]
        print(f"    spatula frame (TCP) in fr3_link0 with the {CALIB_RUN} observer's vial_lip ({wp['created']}, y-z "
              f"{wp['yz_deg']:.3f} deg before orthonormalising):  {{pos: [{p_[0]:.6f}, {p_[1]:.6f}, {p_[2]:.6f}], quat: "
              f"[{q_[0]:.6f}, {q_[1]:.6f}, {q_[2]:.6f}, {q_[3]:.6f}], coordinate_frame: null}}")
    dep = deposit_marks(caps["deposit"], reg["wall_new"], clicks, caps["surface"], reg["ecc_deposit"]["W"],
                        scale=fr.scale)
    print(f"Deposit: shaft in {CAPTURES['deposit'].name} at {dep['shaft']['angle_deg']:+.2f} deg (rms {dep['shaft']['rms']:.2f} px); "
          f"load on the blade {dep['load_px']} px, box {dep['load_box']}, centroid ({dep['load_c'][0]:.0f}, {dep['load_c'][1]:.0f}); "
          f"blade tip ({dep['tip'][0]:.0f}, {dep['tip'][1]:.0f}) px, front edge middle ({dep['front'][0]:.0f}, "
          f"{dep['front'][1]:.0f}) (same method as C, blade {dep['sil']['area']} px); curved arrow from the front edge, "
          f"forward along the axis, to the lower wall {dep['land_forward_mm']:.0f} mm further towards the base "
          f"({dep['land'][0]:.0f}, {dep['land'][1]:.0f}); taps along "
          f"{TAP['direction_lip']} of vial_lip ({'down' if TAP['direction_lip'][1] * y_sign < 0 else 'up'} in the image), "
          f"{TAP['n_taps']} taps")
    lo_, hi_, nlev, nall = tr["level_mass"]
    print(f"\nPanel D: {TOP_TRIAL['material']} run {tr['meta']['created_at']}, trial {tr['trial']} (median carried mass among "
          f"the {nlev} of {nall} trials at s_z = {tr['sz_mm']:+.1f} mm with a mask after the scoop, {lo_:.1f}-{hi_:.1f} mg): "
          f"{tr['carried_mg']:.2f} mg carried, mask {tr['px']} px; top frame read 2 (stored RGB)")

    data = dict(caps=caps, ps=ps, fr=fr, tr=tr, clicks=clicks, contact_mm=contact, path=path, reg=reg, dep=dep,
                spat=spat, wp=wp, photo=load_weigh_photo(),
                photo_a=load_weigh_photo(SOURCE_WEIGH_PHOTO) if SOURCE_WEIGH_PHOTO else None,
                photo_v=load_weigh_photo(CHANGE_PHOTO) if CHANGE_PHOTO else None,
                photo_g=load_weigh_photo(CLEAN_PHOTO))
    cap = capture_data(tr) if any(INPUTS[k] for k in ("pose_bag", "scoop_csv", "deposit_csv", "scoop_still",
                                                     "deposit_still")) else {"have": False}
    if cap.get("have"):
        print("Recorded capture data present (INPUTS): the generated path stays until TRAJ_GATE_OK is set after the "
              "capture check")
    lay = LAYOUTS[layout_name]
    info = render(layout_name, lay, out, data)
    r, Hn = info["rects"], info["H"]
    copied = " and fig_cycle.pdf/.png (the same figure)" if layout_name == MAIN_LAYOUT else " (draft; fig_cycle.* untouched)"
    print(f"\nwrote {out / f'fig_cycle_{layout_name}.pdf'} and .png{copied}: "
          f"{FIG_W_IN * 25.4:.1f} x {Hn * 25.4:.1f} mm; text overlaps {info['text_overlaps']}; titles end inside "
          f"their panels by {min(info['title_slack_mm'].values()):.1f} mm or more "
          f"({min(info['title_slack_mm'], key=info['title_slack_mm'].get)} tightest)")
    print("  printed areas (w x h mm): " + ", ".join(f"{k} {v[2] * 25.4:.1f} x {v[3] * 25.4:.1f}" for k, v in r.items())
          + (f"; change-vials box {info['V_box'][0] * 25.4:.1f} x {info['V_box'][1] * 25.4:.1f}"
             if "V_box" in info else "; change vials: label on the D -> E arrow"))
    x0, y0, x1, y1 = SIDE_CROP
    c0_, c1_, L_, ez_, ey_ = axis_basis(clicks)                # the vial in the window: the body's outer glass
    pu, pl = reg["outline_new"]["lines"]                        # edges (registration fit, t = 0-790 px from the base
    ts_ = reg["outline_new"]["edges"][0]                        # click) and the rim's end on the axis
    pts = np.array([c1_ + t * ez_ + np.polyval(e, t) * ey_ for t in (0.0, ts_.max()) for e in (pu, pl)]
                   + [c0_ + reg["rim_new"]["median_px"] * ez_])
    margin = min((pts[:, 0] - x0).min(), (x1 - pts[:, 0]).min(), (pts[:, 1] - y0).min(), (y1 - pts[:, 1]).min())
    print(f"  the vial in the window: body corners at the base click ({pts[0, 0]:.0f}, {pts[0, 1]:.0f}), ({pts[1, 0]:.0f}, "
          f"{pts[1, 1]:.0f}) and at t = {ts_.max():.0f} px ({pts[2, 0]:.0f}, {pts[2, 1]:.0f}), ({pts[3, 0]:.0f}, "
          f"{pts[3, 1]:.0f}), rim end on the axis ({pts[4, 0]:.0f}, {pts[4, 1]:.0f}): inside by {margin:.0f} px or more")
    print(f"  shared side window B = C = E: {SIDE_CROP}, {x1 - x0} x {y1 - y0} px (aspect {(x1 - x0) / (y1 - y0):.4f}), "
          f"{r['B'][2] * 25.4:.1f} x {r['B'][3] * 25.4:.1f} mm each, {(x1 - x0) / r['B'][2]:.0f} camera px per inch; "
          f"D aspect {info['aspects']['D']:.3f} (solved: row 2 as tall as row 1)")
    bF, bD = info["F"]["box"], info["D"]["box"]
    print(f"  F crop {bF} of {WEIGH_PHOTO.name}, {bF[2] - bF[0]} x {bF[3] - bF[1]} px for {r['F'][2] * 25.4:.1f} mm: "
          f"{(bF[2] - bF[0]) / r['F'][2]:.0f} ppi; D crop {bD}, {bD[2] - bD[0]} px over {r['D'][2]:.3f} in = "
          f"{(bD[2] - bD[0]) / r['D'][2]:.0f} ppi, {r['D'][2] * 25.4 / (bD[2] - bD[0]):.4f} mm per camera px; A: {'photo ' + str(info['A']['box']) if info['photos']['A'] else 'two text boxes'}"
          f"; change vials: {'photo ' + str(info['V']['box']) if info['photos']['V'] else 'text box'}")
    if check_dir is not None:
        write_checks(check_dir, data, reg, cal, xc, ent, dep)


def write_checks(folder: Path, data, reg, cal, xc, ent, dep) -> None:
    """Full-frame check images: registration, the two placements of the frames and the path, the entry angle and
    the deposit marks."""
    folder.mkdir(parents=True, exist_ok=True)
    caps, fr, path, clicks = data["caps"], data["fr"], data["path"], data["clicks"]
    c0, c1, L, ez, ey = axis_basis(clicks)
    fig, axs = plt.subplots(1, 2, figsize=(19.4, 6.4))
    for ax, img, ttl in ((axs[0], data["tr"]["side"], f"stored {TOP_TRIAL['material']} trial {data['tr']['trial']}"),
                         (axs[1], caps["surface"], "vialSideView.png (2026-10-02)")):
        ax.imshow(img); ax.set_axis_off(); ax.set_title(ttl, fontsize=8)
        ax.plot(clicks[:, 0], clicks[:, 1], "w--o", lw=0.8, ms=3)
    ts, e, ok = reg["outline_new"]["edges"]
    for j, col in ((0, "y"), (1, "c")):
        P = c1[None] + ts[:, None] * ez[None] + e[:, j:j + 1] * ey[None]
        axs[1].plot(P[ok, 0], P[ok, 1], ".", color=col, ms=2)
    for ax, wl in ((axs[0], reg["wall_stored"]), (axs[1], reg["wall_new"])):
        t = np.linspace(0, L, 50)
        P = c1[None] + t[:, None] * ez[None] + np.polyval(wl["line"], t)[:, None] * ey[None]
        ax.plot(P[:, 0], P[:, 1], "m-", lw=0.8)
    fig.savefig(folder / "fig_cycle_check_registration.png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(19.36, 12.16))
    ax.imshow(caps["scoop"]); ax.set_axis_off()
    for f, col, lab in ((fr, "c", "image-anchored"), (cal, "m", "calibration")):
        uv = f(path["tip"]); ax.plot(uv[:, 0], uv[:, 1], "-", color=col, lw=1.0, label=f"path, {lab}")
        o = f([0, 0, 0])[0]
        for k, ls in ((2, "-"), (1, "--")):
            e_ = np.zeros(3); e_[k] = 5.0; q = f(e_)[0]
            ax.plot([o[0], q[0]], [o[1], q[1]], ls, color=col, lw=1.4)
        q = f(data["contact_mm"])[0]; ax.plot(*q, "o", color=col, ms=5)
    s = ent["shaft"]; t = np.linspace(-1000, 150, 2)
    ax.plot(s["c"][0] + t * s["v"][0], s["c"][1] + t * s["v"][1], "y-", lw=0.8, label="shaft fit")
    k = int(round(path["closest_t_s"] / path["dt_s"]))
    for kk, col in ((0, "w"), (k, "orange")):
        a, b = fr(path["tip"][kk])[0], fr(path["root"][kk])[0]
        u = (b - a) / np.linalg.norm(b - a)
        ax.plot([a[0], a[0] + 900 * u[0]], [a[1], a[1] + 900 * u[1]], ":", color=col, lw=1.0)
    ax.set_xlim(0, IMG_W); ax.set_ylim(IMG_H, 0); ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"frames and path: cyan image-anchored, magenta calibration ({CALIB_RUN} vial_lip); yellow shaft fit; "
                 f"dotted generated blade axis at t = 0 (white) and at the closest approach (orange)", fontsize=8)
    fig.savefig(folder / "fig_cycle_check_frames.png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9.68, 6.08))
    ax.imshow(caps["deposit"]); ax.set_axis_off()
    x0, y0, x1, y1 = dep["load_box"]
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="m", lw=0.8))
    ax.contour(dep["sil"]["mask"], levels=[0.5], colors="y", linewidths=0.6)
    ax.plot(*dep["tip"], "r+", ms=10); ax.plot(*dep["front"], "c+", ms=10); ax.plot(*dep["land"], "co", ms=4)
    t = np.linspace(0, L, 50)
    P = c1[None] + t[:, None] * ez[None] + np.polyval(reg["wall_new"]["line"], t)[:, None] * ey[None]
    ax.plot(P[:, 0], P[:, 1], "m-", lw=0.6)
    ax.set_xlim(900, IMG_W); ax.set_ylim(900, 60)
    ax.set_title("deposit: load (magenta box), blade silhouette (yellow), tip (red +), front-edge middle (cyan +), "
                 "landing point on the lower wall (cyan dot, magenta line)", fontsize=7)
    fig.savefig(folder / "fig_cycle_check_deposit.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    sp = data["spat"]
    fig, axs = plt.subplots(1, 2, figsize=(19.4, 7.0))
    for ax, (xl, yl) in zip(axs, (((0, IMG_W), (IMG_H, 0)), ((950, 1400), (650, 330)))):
        ax.imshow(caps["scoop"]); ax.set_axis_off()
        ax.contour(sp["sil"]["mask"], levels=[0.5], colors="y", linewidths=0.6)
        ax.plot(sp["uv"][:, 0], sp["uv"][:, 1], "-", color="orange", lw=1.0)
        s = sp["shaft"]; t = np.linspace(-1000, 150, 2)
        ax.plot(s["c"][0] + t * s["v"][0], s["c"][1] + t * s["v"][1], "c--", lw=0.6)
        ax.plot(*sp["sil"]["tip"], "r+", ms=12); ax.plot(*sp["sil"]["front"], "c+", ms=12)
        ax.plot([sp["sil"]["tip"][0], sp["tip"]["uv"][0]], [sp["sil"]["tip"][1], sp["tip"]["uv"][1]], "r-", lw=0.8)
        wp = data["wp"]
        if wp is not None:
            ax.plot(*wp["uv"], "wo", mfc="none", ms=9)
        ax.set_xlim(*xl); ax.set_ylim(*yl)
    axs[0].set_title(f"spatula vs path: silhouette (yellow), shaft line (cyan dashed), path (orange), tip (red +) "
                     f"{sp['tip']['d_px']:.1f} px from the path (red line), waypoint (white ring)", fontsize=8)
    fig.savefig(folder / "fig_cycle_check_spatula.png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"check images written to {folder}")


if __name__ == "__main__":
    main()
