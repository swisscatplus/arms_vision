"""Figure `fig:tp_scoop` (Section 2.5, Task-Parameterized Sampling): the task-parameterized scoop that the
autonomous runs used, drawn from the deployed model and code, in the geometry the robot used.

    EXPERIMENT_2=<tp_lqt training repo> ARMS_SOLID_SAMPLING=<arms_ws/tools/solid_sampling> python paper/analysis/fig_tp_scoop.py

Writes paper/output/fig_tp_scoop.pdf (vector), paper/output/fig_tp_scoop.png (300 dpi) and paper/data/fig_tp_scoop.json (every
number the figure draws or its caption quotes). About 4 min: 33 hard-safety solves of about 3 s each.

Layout and look follow a figure of a related method paper, whose script is fig_tp_motors.py. The drawing helpers
marked "from fig_tp_motors.py" are copied from that script; no number of that figure is reused. Colours, frame
names and the view direction follow this paper's Fig. 3 (paper/analysis/fig_cycle.py).

    A  Each frame's view: the 19 NaCl demonstrations seen from F_vial lip and from F_contact point, and the K = 6
       Gaussian components as one-standard-deviation contours of the y-z marginal of their position block. Filled:
       the frame with the smaller position variance in that component (trace and determinant must agree, asserted).
    B  Transfer, as deployed: the NaCl run's estimated surface (median
       over the trials without a surface failure of the stored T_vialLip_surface: origin on the bore, rotation
       about x_lip), its point p and normal n; the contact point T_vialLip_surface @ [0, 0, s_z, 1] for the run's
       eight offsets, as arms_ws tp_gmm_action.py places it; the full paths of the deployed retrieval (product of
       Gaussians tracked by LQT, then the hard containment re-solve); component 4 of the two frames and their
       product (dashed) for three of the offsets (B_GAUSS_OFFSETS); and the vial wall at |y| = r_v.

View (Z_RIGHT): the y-z plane of each frame as Fig. 3B/3C draws the vial-lip frame, z (the bore, out of the opening)
to the right, y up, x into the page, so the spatula enters from the right and the surface and its normal lie as in
Fig. 3. The rotation of the surface is about x_lip only, so its line in the y-z plane is exact. Both frames share
the vial-lip orientation.

Code and model:
  * The paths are computed with experiment_2 `main` (package tp_lqt), exported read-only with `git archive` into a
    temporary directory. The deployed wheel (tp_scoop + its vendored pbdlib, installed in this environment) is
    solved alongside and must give the same paths; the source differences are listed in the JSON.
  * The model is the copy bundled in the wheel, tp_scoop/params/scoop_HS1.model.npz, read from the wheel file.
  * The deployed call (arms_ws domain/sub_skills/tp_gmm_action.py): start = the approach position of the run
    config in vial_lip with the model's reference orientation q_ref_eff, at rest; contact point in vial_lip;
    safety "hard"; horizon, dt and r_factor left to the artifact and code defaults (223 steps, 0.02 s, r = 0,
    R = I); anchor_end with 12 steps (tp_scoop infer). The robot streamed at most 200 waypoints of the path
    (_decimate); the figure draws all 223 samples.
  * Whether a containment constraint shaped a path is tested by re-solving the same SOCP with a vial too large to
    bind; the bore placement of the first draft is kept in the JSON as a record (`record_bore_vs_normal`).

Environment tp-scoop-fig (created 2026-10-05): python 3.12, numpy 1.26.4, scipy 1.15, cvxpy 1.6.7, clarabel, scs,
ecos, matplotlib, termcolor, zarr 2.18, and the wheel installed with
    pip install --no-deps <arms_ws>/tools/solid_sampling/vendor/tp_scoop/wheel/tp_scoop-0.1.0-py3-none-any.whl
Nothing in experiment_2 or arms_ws is written. Raw demonstration samples stay local and
are not copied into the JSON. tools/scoop_path_tpscoop.py imports the loaders and `solve` from here.
"""
from __future__ import annotations

import copy
import difflib
import hashlib
import inspect
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import warnings
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore", message="Solution may be inaccurate")   # cvxpy, on every "optimal_inaccurate"

import numpy as np                                                         # noqa: E402
import matplotlib                                                          # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402
from matplotlib import patheffects                                         # noqa: E402
from matplotlib.lines import Line2D                                        # noqa: E402
from matplotlib.patches import Ellipse                                     # noqa: E402
from scipy.spatial.transform import Rotation as R                          # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DATA, IMAGES, load                                      # noqa: E402

MM = 1000.0
EXP2 = Path(os.environ.get("EXPERIMENT_2", HERE.parent / "external" / "experiment_2"))   # not in the release
EXP2_BRANCH = "main"
DEMO_ROOT = EXP2 / "data" / "recollected_HS1_data"
SS = Path(os.environ.get("ARMS_SOLID_SAMPLING", HERE.parent / "external" / "solid_sampling"))   # not in the release
WHEEL = SS / "vendor" / "tp_scoop" / "wheel" / "tp_scoop-0.1.0-py3-none-any.whl"
MODEL_MEMBER = "tp_scoop/params/scoop_HS1.model.npz"
SALT_ZARR = DATA / "HS1_vision" / "benchmark_2026-09-10-18-52.zarr"       # extract_runs.RUNS["salt"]
OUT_PDF, OUT_PNG, OUT_JSON = IMAGES / "fig_tp_scoop.pdf", IMAGES / "fig_tp_scoop.png", DATA / "fig_tp_scoop.json"

STATE_B = 4               # panel B: the scooping component, at whose start the deepest points lie (checked)
#: Panel B draws component 4 for three of the eight offsets only, the shallowest, the one of Fig. 3C and the
#: deepest: with all eight, sixteen overlapping contours hid the paths. All eight are in the JSON.
B_GAUSS_OFFSETS = (8.0, -4.0, -15.0)
B_GAUSSIANS = False       # panel B: draw component STATE_B's Gaussians (vial, contact, product) and their labels;
                          # off in the paper's figure
N_DEMOS = 19              # the folders of recollected_HS1_data (HS1_19 absent); asserted with the data hash
#: A vial pose different from the run config's preview pose, to check that the vial-frame path does not depend on
#: the vial pose (a robot situation recorded on 2026-10-02, bore 20.5 deg above horizontal, like the demos).
VIAL_POSE_CHECK = (np.array([-0.18652280535672336, 0.3436390468871958, 0.29094667731447976]),
                   np.array([-0.15921332908640232, 0.5471138794602072, 0.7907084326689788, -0.22382513989584335]))
#: Never-binding re-solve: the hard SOCP with a vial of this radius (m) cannot be constrained.
LOOSE_RADIUS_M = 1.0
#: A path counts as shaped by a containment constraint if the hard path and the never-binding re-solve differ by
#: more than this (mm); the solver's own scatter between the two is about 0.01 mm.
ACTIVE_TOL_MM = 0.05

# --- print size: figure* at 0.92 \textwidth of the Wiley template (178 mm) ---------------------------------------
FIG_W_IN = 0.92 * 178.0 / 25.4
FS = 7.0                  # all text, as fig_cycle.py
FS_LETTER = 8.0           # panel letters, bold
Z_RIGHT = True            # z to the right (opening right), as Fig. 3B/3C
C_VIAL = "#0072B2"        # F_vial lip and its Gaussians (Okabe-Ito blue), as fig_cycle.py
C_CON = "#CC79A7"         # F_contact point and its Gaussians (Okabe-Ito reddish purple, not red), as
                          # fig_cycle.py
# s_z = +8 ... -15 mm (shallow -> deep): one-hue ramp on C_CON's OKLCH hue (346 deg), L 0.78 -> 0.35 in equal
# steps, C 0.13. Validated with the dataviz validator, `validate_palette.js --ordinal`: monotone L, adjacent
# dL >= 0.06 (smallest 0.060), light end 2.07:1 on #fcfcfb (2.13:1 on white), hue spread 1 deg. Re-checked below
# (ramp_check).
C_SZ = ("#F394C8", "#DE81B5", "#C96EA2", "#B55C8F", "#A14A7C", "#8D386A", "#792559", "#661148")
C_SURF = "#ff3030"        # the estimated surface, as fig_cycle.py (SURF_RED, dashed in Fig. 3C)
C_PATH = "#c4c4c4"        # demonstrations, context only
C_INK = "#1a1a1a"
C_MUTED = "#6b6b6b"
C_GLASS_EDGE = "#7d838c"
HALO = [patheffects.withStroke(linewidth=2.2, foreground="white")]
YZ = [1, 2]               # the y-z plane of a frame
ZY = [2, 1]               # plotting order, turned: z across, y up

plt.rcParams.update({
    "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "font.size": FS,
    "axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
    "legend.fontsize": FS, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "lines.linewidth": 1.0, "lines.solid_capstyle": "round", "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})


# =================================================================================================================
# Provenance: the code, the model, the data
# =================================================================================================================

def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_code(tmp: Path) -> dict:
    """Export experiment_2 `main`'s tp_lqt read-only (git archive, no checkout) and import it next to the installed
    wheel's tp_scoop. Checks that the installed tp_scoop/pbdlib are byte-identical to the wheel file, and that
    `main` pins the same pbdlib commit as the wheel vendors."""
    commit = git(EXP2, "rev-parse", EXP2_BRANCH)
    blob = subprocess.run(["git", "-C", str(EXP2), "archive", EXP2_BRANCH, "tp_lqt"], capture_output=True,
                          check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        tar.extractall(tmp, filter="data")
    sys.path.insert(0, str(tmp))
    import tp_lqt                                                           # noqa: F401
    import tp_lqt.infer
    import tp_lqt.io
    import tp_lqt.preprocess
    import tp_lqt.serialize
    import tp_lqt.synthesize
    import tp_lqt.constraints
    import tp_scoop
    import tp_scoop.infer
    import pbdlib
    assert Path(tp_lqt.__file__).resolve().is_relative_to(tmp.resolve()), tp_lqt.__file__

    site = Path(tp_scoop.__file__).resolve().parents[1]
    with zipfile.ZipFile(WHEEL) as z:
        members = [n for n in z.namelist() if n.startswith(("tp_scoop/", "pbdlib/")) and not n.endswith("/")]
        for n in members:
            assert (site / n).read_bytes() == z.read(n), f"installed {n} differs from the wheel"
        wheel_src = {Path(n).name: z.read(n).decode() for n in z.namelist()
                     if n.startswith("tp_scoop/") and n.endswith(".py")}
        vendored = z.read("pbdlib/VENDORED_COMMIT").decode().strip()
    pin = git(EXP2, "ls-tree", EXP2_BRANCH, "pbdlib-python").split()[2]
    assert pin == vendored, (pin, vendored)

    # source comparison, main's tp_lqt against the wheel's tp_scoop (package name normalised)
    main_src = {p.name: p.read_text() for p in (tmp / "tp_lqt").glob("*.py")}
    diff = {}
    for name in sorted(set(main_src) | set(wheel_src)):
        w, m = wheel_src.get(name), main_src.get(name)
        if w is None:
            diff[name] = "main only"
        elif m is None:
            diff[name] = "wheel only"
        else:
            lines = list(difflib.unified_diff(w.replace("tp_scoop", "tp_lqt").splitlines(), m.splitlines(),
                                              lineterm="", n=0))
            changed = [ln for ln in lines if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]
            diff[name] = "identical" if not changed else f"{len(changed)} changed lines"
    return dict(
        tp_lqt=tp_lqt, tp_scoop=tp_scoop,
        main=dict(repo=str(EXP2), branch=EXP2_BRANCH, commit=commit,
                  date=git(EXP2, "log", "-1", "--format=%cI", EXP2_BRANCH), export="git archive main tp_lqt",
                  pbdlib_pin=pin),
        wheel=dict(path=str(WHEEL), sha256=sha256_bytes(WHEEL.read_bytes()), installed_identical=True,
                   pbdlib_vendored_commit=vendored, pbdlib_imported_from=str(Path(pbdlib.__file__).parent)),
        source_diff_main_vs_wheel=diff,
        versions=dict(numpy=np.__version__, scipy=__import__("scipy").__version__,
                      cvxpy=__import__("cvxpy").__version__, clarabel=_version("clarabel"),
                      python=sys.version.split()[0]))


def _version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:                                                       # pragma: no cover
        return "unknown"


def load_model(code: dict, tmp: Path) -> dict:
    """The model bundled in the wheel, loaded by main's serialize and by the wheel's; both must agree."""
    with zipfile.ZipFile(WHEEL) as z:
        raw = z.read(MODEL_MEMBER)
    path = tmp / "scoop_HS1.model.npz"
    path.write_bytes(raw)
    main_model = code["tp_lqt"].infer.Model.load(str(path))
    wheel_model = code["tp_scoop"].infer.Model.load(str(path))
    for a in ("mu", "sigma", "priors", "trans", "init_priors", "q_ref"):
        assert np.array_equal(getattr(main_model.hmm, a), getattr(wheel_model.hmm, a)), a
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["meta"].item()))
    return dict(main=main_model, wheel=wheel_model, hmm=main_model.hmm, meta=meta, sha256=sha256_bytes(raw),
                member=f"{WHEEL.name}!{MODEL_MEMBER}")


def load_demos(code: dict, meta: dict) -> dict:
    """The 19 HS1 demonstrations, checked against the model's data hash, preprocessed as in training (main's io and
    preprocess, default calibration 2026-06-01 = the model's train_config calib)."""
    TL = code["tp_lqt"]
    data_hash = TL.serialize.compute_data_sha256(str(DEMO_ROOT))
    if data_hash != meta["data_sha256"]:
        raise SystemExit(f"STOP: data hash {data_hash} does not match the model's {meta['data_sha256']}")
    cleans = [TL.preprocess.preprocess_demo(r) for r in TL.io.load_dataset(str(DEMO_ROOT))]
    assert len(cleans) == N_DEMOS, len(cleans)
    assert meta["train_config"]["calib"].startswith("default(2026-06-01)"), meta["train_config"]
    return dict(cleans=cleans, data_sha256=data_hash, names=[c.name for c in cleans])


def load_run_config() -> dict:
    """The scoop section of the NaCl run's own config snapshot (zarr attrs config_json)."""
    import zarr
    g = zarr.open_group(str(SALT_ZARR), mode="r")
    cfg = json.loads(g.attrs["config_json"])
    act = cfg["fr3_actions"]["scoop_tp_gmm"]
    for key in ("r_factor", "horizon", "dt", "anchor_steps"):
        assert key not in act, f"run config sets {key}; the defaults are assumed"
    su = g["events"]["surface_observer"]
    sc = g["events"]["scoop_contact"]
    return dict(act=act, depths_mm=[float(v) * MM for v in cfg["benchmark"]["sweep"]["scoop_sites"]["depths_m"]],
                surface=dict(iteration=np.asarray(su["iteration"][:], int),
                             pose7=np.asarray(su["T_vialLip_surface_pose7"][:], float)),
                contact=dict(iteration=np.asarray(sc["iteration"][:], int),
                             offset_m=np.asarray(sc["surface_offset_m"][:], float)))


# =================================================================================================================
# The deployed retrieval
# =================================================================================================================

def approach_position_world(vial_pose_world, approach_pos):
    """From arms_ws domain/sub_skills/tp_gmm_action.py (approach_position_world): the scoop start position."""
    vial_pos, vial_quat = vial_pose_world
    R_world_vial = R.from_quat(np.asarray(vial_quat, np.float64)).as_matrix()
    return R_world_vial @ np.asarray(approach_pos, np.float64) + np.asarray(vial_pos, np.float64)


def q_ref_eff(hmm, vial_quat_world):
    """From tp_gmm_action.py (TpGmmPlanner.q_ref_eff): the start orientation, the model's reference re-anchored to
    the live vial (vial chart)."""
    assert hmm.orientation_ref_frame == "vial"
    return (R.from_quat(np.asarray(vial_quat_world, np.float64)) * R.from_quat(hmm.q_ref)).as_quat()


def decimate_index(n: int, max_waypoints: int) -> np.ndarray:
    """From tp_gmm_action.py (_decimate): the rows streamed to the FR3."""
    if max_waypoints <= 0 or n <= max_waypoints:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_waypoints).round().astype(int))


def solve(model, act: dict, vial_pose, contact_mm, *, main: bool, safety: str | None = None) -> dict:
    """One scoop through Model.infer exactly as TpGmmPlanner.infer calls it, mapped back to vial_lip (mm). main's
    infer defaults r_factor to 4.0; the deployed wheel defaults it to 0.0 and the deployment passed none, so main is
    given r_factor = 0.0 explicitly."""
    from tp_scoop.infer import WORLD_TO_BASE_2026_06_01 as wtb
    vial_pos, vial_quat = (np.asarray(v, float) for v in vial_pose)
    A = R.from_quat(vial_quat).as_matrix()
    start = (approach_position_world((vial_pos, vial_quat), act["approach"]["pos"]), q_ref_eff(model.hmm, vial_quat))
    kw = dict(r_factor=0.0) if main else {}
    traj = model.infer(start=start, vial_pose_world=(vial_pos, vial_quat),
                       contact_point=tuple(float(c) / MM for c in contact_mm), world_to_base=wtb,
                       safety=safety or act["safety"], chart_gate_deg=float(act["chart_gate_deg"]), **kw)
    Rwb = R.from_quat(wtb["q"])
    pos_w = Rwb.apply(traj.pos) + np.asarray(wtb["t"], float)
    quat_w = (Rwb * R.from_quat(traj.quat)).as_quat()
    tip = MM * (pos_w - vial_pos) @ A
    from tp_scoop.frames import SPATULA_TOOL_AXIS                           # tip -> handle is +y; the tool points -y
    tool = R.from_quat(quat_w).apply(SPATULA_TOOL_AXIS) @ A                # in vial_lip
    return dict(tip=tip, tool=tool, quat_world=quat_w, status=traj.meta["safety_status"],
                eta_max_deg=float(traj.meta["eta_max_deg"]), dt=float(traj.meta["dt"]),
                horizon=int(traj.meta["horizon"]), start=start)


def unconstrained(code, hmm, vial_pose, contact_mm, start, *, weaken: str | None = None):
    """The same LQT without the hard re-solve (geometry_constraint "off"), same x0. ``weaken`` scales one frame's
    covariance block by 1e6 on an in-memory copy (the artifact is not touched)."""
    TL = code["tp_lqt"]
    m = hmm
    if weaken is not None:
        m = copy.deepcopy(hmm)
        sig = np.array(m.sigma)
        j = TL.model.FRAME_ORDER.index(weaken) * 12
        sig[:, j:j + 12, j:j + 12] *= 1e6
        m.sigma = sig
    vial_pos, vial_quat = (np.asarray(v, float) for v in vial_pose)
    q_eff = q_ref_eff(hmm, vial_quat)
    eta0 = np.atleast_2d(TL.orientation.log_chart(start[1], q_eff))[0]
    x0 = np.concatenate([start[0], eta0, np.zeros(6)])
    syn = TL.synthesize.synthesize(m, vial_quat, vial_pos, np.asarray(contact_mm, float) / MM,
                                   horizon=hmm.default_horizon, dt=1.0 / hmm.rate_hz, r_factor=0.0, x0=x0,
                                   anchor_end=True, geometry_constraint="off")
    return syn


def product_of_gaussians(code, hmm, vial_pose, contact_mm):
    """The per-state product of the two frames' Gaussians that the LQT tracks, built exactly as
    tp_lqt.synthesize.synthesize builds it (synthesize.py, `q_ref, orf = ...` to `prod = prod * mods[m]`), and
    the two transported per-frame Gaussians. Position blocks returned in vial_lip, mm."""
    TL = code["tp_lqt"]
    ori, SY, MD = TL.orientation, TL.synthesize, TL.model
    vial_pos, vial_quat = (np.asarray(v, float) for v in vial_pose)
    q_ref, orf = ori.chart_anchor(hmm)
    nb_var = 6 * 2
    fr = SY.frames_from_vial_pose(vial_quat, vial_pos, np.asarray(contact_mm, float) / MM, MD.FRAME_ORDER)
    lifted = [MD.lift_config_AB(fr[n][0], fr[n][1], q_ref, 2, orientation_ref_frame=orf) for n in MD.FRAME_ORDER]
    mods = [hmm.marginal_model(slice(m * nb_var, (m + 1) * nb_var)).lintrans(lifted[m][0], lifted[m][1])
            for m in range(len(MD.FRAME_ORDER))]
    prod = mods[0]
    for m in range(1, len(mods)):
        prod = prod * mods[m]
    A = R.from_quat(vial_quat).as_matrix()

    def pos(g):
        mu = MM * (np.asarray(g.mu)[:, :3] - vial_pos) @ A
        cov = MM ** 2 * np.einsum("ji,kjl,lm->kim", A, np.asarray(g.sigma)[:, :3, :3], A)
        return mu, cov
    return dict(prod=pos(prod), vial=pos(mods[0]), contact=pos(mods[1]), prod_model=prod)


def blade_clearance(code, tip_mm, tool, geom) -> dict:
    """Post-hoc check of a path against tp_scoop's containment geometry, as solve_socp samples it: the blade from
    the tip back along the tool axis (4 samples over L_blade), each below the lip within radius_at_depth less the
    body radius, and the blade's lip-plane crossing within r_lip less the body radius. Clearances in mm."""
    rad = code["tp_lqt"].constraints.radius_at_depth
    L, rb = geom["L_blade"] * MM, geom["body_radius"] * MM
    wall = np.inf
    for s in np.linspace(0.0, 1.0, int(geom["body_samples_K"])):
        p = tip_mm + s * L * (-tool)
        below = p[:, 2] < 0
        r_eff = MM * rad(p[:, 2] / MM, r_body=geom["r_v"], r_lip=geom["r_lip"], l_shoulder=geom["l_shoulder"]) - rb
        if below.any():
            wall = min(wall, float((r_eff - np.hypot(p[:, 0], p[:, 1]))[below].min()))
    base = tip_mm + L * (-tool)
    act = tip_mm[:, 2] * base[:, 2] < 0
    s_lip = tip_mm[act, 2] / (tip_mm[act, 2] - base[act, 2])
    p_lip = tip_mm[act] + s_lip[:, None] * (base[act] - tip_mm[act])
    lip = float((geom["r_lip"] * MM - rb - np.hypot(p_lip[:, 0], p_lip[:, 1])).min())
    return dict(wall_mm=wall, lip_mm=lip, lip_crossing_steps=int(act.sum()))




def surface_frame_contact_point(T_vialLip_surface, surface_offset_m):
    """From arms_ws domain/sub_skills/tp_gmm_action.py (surface_frame_contact_point): a surface-frame offset
    (sx, sy, sz) to the vial_lip contact point, contact = (T_vialLip_surface @ [sx, sy, sz, 1])[:3]. Units follow
    the inputs (mm here)."""
    off = np.asarray(surface_offset_m, dtype=np.float64).reshape(3)
    homo = np.concatenate([off, np.array([1.0])])
    return (np.asarray(T_vialLip_surface, dtype=np.float64) @ homo)[:3]


def surface_estimate(run, d) -> dict:
    """The NaCl run's estimated surface as the deployed code used it, T_vialLip_surface (pose of the surface frame in
    vial_lip): the median, over the trials without a surface failure, of the stored poses, whose origin is on the
    bore and whose rotation is about x_lip only (both asserted). Lengths in mm."""
    pose7 = run["surface"]["pose7"]
    rotvec = R.from_quat(pose7[:, 3:7]).as_rotvec()
    assert np.allclose(rotvec[:, 1:], 0.0, atol=1e-9), "surface tilt not about x_lip"
    assert np.allclose(pose7[:, :2], 0.0), "surface origin off the vial axis"
    fail = set(d["iteration"][d["surface_fail"]].tolist())
    keep = np.array([int(i) not in fail for i in run["surface"]["iteration"]])
    z_s = float(np.median(MM * pose7[keep, 2]))
    ok = ~d["surface_fail"] & np.isfinite(d["surface_z_lip_mm"])
    assert abs(z_s - float(np.median(d["surface_z_lip_mm"][ok]))) < 1e-9, "zarr and CSV surface heights differ"
    ang = np.degrees(rotvec[:, 0])
    a = float(np.median(ang[keep]))
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec([np.radians(a), 0.0, 0.0]).as_matrix()
    T[:3, 3] = [0.0, 0.0, z_s]
    return dict(T=T, z_s=z_s, angle_deg=a, n=T[:3, 2].copy(), p=T[:3, 3].copy(), ang_deg=ang, keep=keep,
                fail=sorted(fail), n_trials=int(keep.sum()))


def loose_resolve(code, hmm, vial_pose, contact_mm, start):
    """The hard re-solve of tp_lqt.synthesize with a vial too large to bind (radius LOOSE_RADIUS_M): the same SOCP,
    cost, terminal anchor, solver and x0 as the deployed call, without an effective constraint."""
    TL = code["tp_lqt"]
    vial_pos, vial_quat = (np.asarray(v, float) for v in vial_pose)
    eta0 = np.atleast_2d(TL.orientation.log_chart(start[1], q_ref_eff(hmm, vial_quat)))[0]
    x0 = np.concatenate([start[0], eta0, np.zeros(6)])
    return TL.synthesize.synthesize(hmm, vial_quat, vial_pos, np.asarray(contact_mm, float) / MM,
                                    horizon=hmm.default_horizon, dt=1.0 / hmm.rate_hz, r_factor=0.0, x0=x0,
                                    anchor_end=True, geometry_constraint="hard", r_v=LOOSE_RADIUS_M,
                                    r_lip=LOOSE_RADIUS_M, l_shoulder=1e-6)


def tool_axis_vial(quat_world, vial_quat):
    """The spatula's pointing direction (frames.SPATULA_TOOL_AXIS) in vial_lip, per sample."""
    from tp_scoop.frames import SPATULA_TOOL_AXIS
    return R.from_quat(quat_world).apply(SPATULA_TOOL_AXIS) @ R.from_quat(vial_quat).as_matrix()


# =================================================================================================================
# Drawing helpers (copied from fig_tp_motors.py of a related method paper, adapted to mm arrays)
# =================================================================================================================

def ellipse(ax, mean, cov, color, filled, lw=0.9, zorder=3, ls="-"):
    """From fig_tp_motors.py: one-standard-deviation contour of the y-z marginal of a 3-D position Gaussian, in the
    turned plane (z across, y up)."""
    values, vectors = np.linalg.eigh(cov[np.ix_(ZY, ZY)])
    angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
    width, height = 2.0 * np.sqrt(values[1]), 2.0 * np.sqrt(values[0])
    centre = (mean[2], mean[1])
    if filled:
        ax.add_patch(Ellipse(centre, width, height, angle=angle, facecolor=color, edgecolor="none", alpha=0.22,
                             zorder=zorder))
    ax.add_patch(Ellipse(centre, width, height, angle=angle, facecolor="none", edgecolor=color,
                         lw=lw if filled else 0.65, ls=ls, zorder=zorder + 0.1))


def outside(mean, cov, direction_deg, pad):
    """From fig_tp_motors.py: the point `pad` mm beyond the one-sd contour along `direction_deg` (measured in the
    frame's y-z plane: 0 = +y, 90 = +z, 180 = -y, 270 = -z), returned in plotting coordinates (z, y)."""
    u = np.array([np.cos(np.radians(direction_deg)), np.sin(np.radians(direction_deg))])
    radius = 1.0 / np.sqrt(u @ np.linalg.solve(cov[np.ix_(YZ, YZ)], u))
    y, z = np.asarray(mean)[YZ] + (radius + pad) * u
    return np.array([z, y])


def triad(ax, color, length=6.0, lw=1.0, zorder=8):
    """From fig_tp_motors.py, with the x marker of fig_cycle.py's triad: a frame at its own origin, +z along the
    plot's horizontal axis (out of the opening), +y up, and x into the page (circled cross) with Z_RIGHT, out of
    it (circled dot) otherwise."""
    style = "-|>,head_length=0.30,head_width=0.14"
    for d in ((length, 0.0), (0.0, length)):
        ax.annotate("", xy=d, xytext=(0.0, 0.0),
                    arrowprops=dict(arrowstyle=style, lw=lw, color=color, shrinkA=0, shrinkB=0), zorder=zorder)
    ax.scatter([0.0], [0.0], s=26, facecolor="white", edgecolor=color, linewidths=0.8, zorder=zorder + 0.1)
    ax.scatter([0.0], [0.0], s=(13 if Z_RIGHT else 3), marker=("x" if Z_RIGHT else "o"), color=color,
               linewidths=0.8, zorder=zorder + 0.2)


def number(ax, z, y, k):
    """From fig_tp_motors.py: a component number at plotting coordinates (z, y)."""
    ax.text(z, y, str(k), ha="center", va="center", fontsize=FS, color=C_INK, path_effects=HALO, zorder=9)


def leader(ax, text, xy, xytext, ha="left", va="center", **kw):
    """From fig_tp_motors.py: a label with a thin leader line."""
    ax.annotate(text, xy=xy, xytext=xytext, textcoords="data", ha=ha, va=va,
                arrowprops=dict(arrowstyle="-", lw=0.4, color=C_MUTED, shrinkA=1.0, shrinkB=1.0), zorder=9, **kw)


def ramp_check(colors) -> dict:
    """OKLCH lightness of the ramp: monotone, adjacent steps >= 0.06 apart (the validator's ordinal rule)."""
    def lin(c):
        c = np.array([int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)])
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    L = np.array([0.2104542553, 0.7936177850, -0.0040720468]) @ np.cbrt(M1 @ np.array([lin(c) for c in colors]).T)
    gaps = -np.diff(L)
    assert np.all(gaps >= 0.06), gaps
    return dict(oklch_L=L.round(4), min_adjacent_dL=float(gaps.min()))


def streamed_chord(tip, idx) -> float:
    """Largest distance of a sample the robot did not stream from the chord between its two streamed neighbours."""
    chord = 0.0
    for j in np.setdiff1d(np.arange(len(tip)), idx):
        a, b = idx[idx < j].max(), idx[idx > j].min()
        u = (tip[b] - tip[a]) / np.linalg.norm(tip[b] - tip[a])
        w = tip[j] - tip[a]
        chord = max(chord, float(np.linalg.norm(w - (w @ u) * u)))
    return chord


# =================================================================================================================

def compute(code, mdl, demos, run) -> dict:
    TL, hmm, act = code["tp_lqt"], mdl["hmm"], run["act"]
    meta = mdl["meta"]
    K = int(hmm.nb_states)
    assert (K, int(hmm.nb_dim), meta["nb_var_pos"], meta["nb_deriv"]) == (6, 24, 6, 2)
    assert tuple(meta["frame_order"]) == TL.model.FRAME_ORDER == ("vial", "contact")
    sig = inspect.signature(TL.synthesize.synthesize).parameters
    geom = {k: sig[k].default for k in ("L_blade", "body_radius", "r_v", "r_lip", "l_shoulder", "body_samples_K",
                                        "anchor_steps")}

    # ---------------------------------------------------------------- the surface and the contact points, as deployed
    d = load("salt")
    surf = surface_estimate(run, d)
    p_s, n_s = surf["p"], surf["n"]
    offsets = run["depths_mm"]
    assert sorted(offsets) == sorted(set(np.round(d["sz_mm"], 3))), (offsets, set(d["sz_mm"]))
    assert offsets == sorted(offsets, reverse=True) and len(offsets) == len(C_SZ)   # ramp: shallow (light) -> deep
    contacts = {sz: surface_frame_contact_point(surf["T"], [0.0, 0.0, sz]) for sz in offsets}
    assert all(np.allclose(contacts[sz], p_s + sz * n_s) for sz in offsets)

    # ---------------------------------------------------------------- per-frame Gaussians (position block)
    gauss = {}
    for m, name in enumerate(TL.model.FRAME_ORDER):
        sl = slice(12 * m, 12 * m + 3)
        rows = []
        for k in range(K):
            mu, cov = MM * np.asarray(hmm.mu[k][sl]), MM ** 2 * np.asarray(hmm.sigma[k][sl, sl])
            rows.append(dict(mean_mm=mu, cov_mm2=cov, variance_mm2=float(np.trace(cov) / 3),
                             generalized_variance_mm2=float(np.linalg.det(cov) ** (1 / 3)),
                             sd_yz_mm=np.sqrt(np.diag(cov)[YZ])))
        gauss[name] = rows
    governs = []
    for k in range(K):
        v, c = gauss["vial"][k], gauss["contact"][k]
        by_trace = "vial" if v["variance_mm2"] < c["variance_mm2"] else "contact"
        by_det = "vial" if v["generalized_variance_mm2"] < c["generalized_variance_mm2"] else "contact"
        assert by_trace == by_det, f"component {k + 1}: trace and determinant disagree"
        governs.append(by_trace)

    # ---------------------------------------------------------------- demonstrations in both frames
    views = {"vial": [], "contact": []}
    for c in demos["cleans"]:
        A_c, b_c = c.frames["contact"]
        p_con = MM * TL.frames.project_pos_to_frame(c.pos_world, A_c, b_c)
        assert np.allclose(p_con, MM * (c.pos_vial - c.contact_pos_vial), atol=1e-9), "contact view mismatch"
        views["vial"].append(MM * c.pos_vial)
        views["contact"].append(p_con)
    rho_max = {c.name[-6:]: float(MM * np.hypot(c.pos_vial[:, 0], c.pos_vial[:, 1]).max()) for c in demos["cleans"]}
    y_max = {c.name[-6:]: float(MM * np.abs(c.pos_vial[:, 1]).max()) for c in demos["cleans"]}
    beyond = sorted(n for n, r in rho_max.items() if r > MM * geom["r_v"])

    # ---------------------------------------------------------------- the solves, as deployed
    vial_pose = (np.asarray(act["preview"]["vial_pos"], float), np.asarray(act["preview"]["vial_quat"], float))
    rows = []
    for sz, color in zip(offsets, C_SZ):
        c = contacts[sz]
        hard = solve(mdl["main"], act, vial_pose, c, main=True)
        wheel = solve(mdl["wheel"], act, vial_pose, c, main=False)
        off = unconstrained(code, hmm, vial_pose, c, hard["start"])
        loose = loose_resolve(code, hmm, vial_pose, c, hard["start"])
        tip, seq = hard["tip"], np.asarray(off.state_seq)
        assert np.array_equal(seq, np.asarray(loose.state_seq))
        tip_off, tip_loose = MM * off.pos_vial, MM * loose.pos_vial
        depth = (p_s - tip) @ n_s                        # below the estimated surface, along n
        k_deep, k_z = int(np.argmax(depth)), int(np.argmin(tip[:, 2]))
        gap = np.linalg.norm(tip - c, axis=1)
        k_close = int(np.argmin(gap))
        dev = np.linalg.norm(tip - tip_off, axis=1)
        k_dev = int(np.argmax(dev))
        loose_dev = float(np.linalg.norm(tip - tip_loose, axis=1).max())
        clear = blade_clearance(code, tip, hard["tool"], geom)
        clear_off = blade_clearance(code, tip_off, tool_axis_vial(off.quat_world, vial_pose[1]), geom)
        idx = decimate_index(len(tip), int(act["stream_max_waypoints"]))
        active = bool(loose_dev > ACTIVE_TOL_MM or min(clear["wall_mm"], clear["lip_mm"]) <= 0.0)
        rows.append(dict(
            sz_mm=sz, color=color, contact_mm=c, tip=tip, root=tip + MM * geom["L_blade"] * (-hard["tool"]),
            state_seq=seq + 1, status_main=hard["status"], status_wheel=wheel["status"],
            accepted=hard["status"] in act["require_safety_status"],
            main_vs_wheel_max_mm=float(np.abs(tip - wheel["tip"]).max()),
            resolve_max_mm=float(dev[k_dev]), resolve_at_s=k_dev * hard["dt"], resolve_component=int(seq[k_dev]) + 1,
            resolve_components1to4_max_mm=float(dev[seq <= 3].max()),
            hard_vs_never_binding_max_mm=loose_dev, constraint_active=active,
            clearance=clear, clearance_unconstrained=clear_off, eta_max_deg=hard["eta_max_deg"],
            closest_mm=float(gap[k_close]), closest_t_s=k_close * hard["dt"], closest_component=int(seq[k_close]) + 1,
            deepest_below_surface_mm=float(depth[k_deep]), deepest_t_s=k_deep * hard["dt"],
            deepest_component=int(seq[k_deep]) + 1, deepest_point_mm=tip[k_deep],
            deepest_z_mm=float(tip[k_z, 2]), deepest_z_component=int(seq[k_z]) + 1,
            unconstrained_deepest_below_surface_mm=float(((p_s - tip_off) @ n_s).max()),
            n=len(tip), n_streamed=len(idx), streamed_chord_max_mm=streamed_chord(tip, idx),
            start_mm=tip[0], end_mm=tip[-1], dt=hard["dt"], horizon=hard["horizon"],
            max_rho_mm=float(np.hypot(tip[:, 0], tip[:, 1]).max()),
            y_range_mm=(float(tip[:, 1].min()), float(tip[:, 1].max())),
            x_range_mm=(float(tip[:, 0].min()), float(tip[:, 0].max()))))
        print(f"s_z {sz:+5.1f} mm: contact {np.round(c, 2)} mm; {hard['status']} (wheel {wheel['status']}, max "
              f"|main - wheel| {rows[-1]['main_vs_wheel_max_mm']:.1e} mm); re-solve moved it {dev[k_dev]:.3f} mm "
              f"(component {seq[k_dev] + 1}), {loose_dev:.4f} mm from the never-binding re-solve; clearance wall "
              f"{clear['wall_mm']:.2f} mm, lip {clear['lip_mm']:.2f} mm; deepest {depth[k_deep]:+.2f} mm below the "
              f"surface (component {seq[k_deep] + 1}, {k_deep * hard['dt']:.2f} s); closest {gap[k_close]:.2f} mm",
              flush=True)
    assert all(r["accepted"] for r in rows), "STOP: a hard re-solve returned a status the deployed gate rejects"
    assert all(r["status_main"] == r["status_wheel"] for r in rows)
    assert max(r["main_vs_wheel_max_mm"] for r in rows) < 1e-9, "main and the wheel give different paths"
    assert all(r["clearance"]["wall_mm"] > 0 and r["clearance"]["lip_mm"] > 0 for r in rows)
    assert {r["deepest_component"] for r in rows} <= {STATE_B - 1, STATE_B}, [r["deepest_component"] for r in rows]

    sz_arr = np.array(offsets)
    deep = np.array([r["deepest_below_surface_mm"] for r in rows])
    fit = np.polyfit(sz_arr, deep, 1)
    slope_n = float(-fit[0])                             # mm deeper below the surface per mm of -s_z, along n
    resid = float(np.abs(deep - np.polyval(fit, sz_arr)).max())
    slope_n_off = float(-np.polyfit(sz_arr, [r["unconstrained_deepest_below_surface_mm"] for r in rows], 1)[0])
    assert abs(slope_n - slope_n_off) < 5e-3, (slope_n, slope_n_off)
    slope_z = float(np.polyfit(sz_arr, [r["deepest_z_mm"] for r in rows], 1)[0])

    # the references the LQT tracks: per component, how far the product mean moves along n per commanded mm
    pogs = {sz: product_of_gaussians(code, hmm, vial_pose, contacts[sz]) for sz in offsets}
    lo, hi = min(offsets), max(offsets)
    prod_shift = {k + 1: float((pogs[hi]["prod"][0][k] - pogs[lo]["prod"][0][k]) @ n_s / (hi - lo)) for k in range(K)}
    con_shift = [float((pogs[hi]["contact"][0][k] - pogs[lo]["contact"][0][k]) @ n_s / (hi - lo)) for k in range(K)]
    assert np.allclose(con_shift, 1.0), con_shift            # the contact frame's Gaussians move with the contact point
    assert all(np.allclose(p["vial"][0], pogs[lo]["vial"][0]) for p in pogs.values())   # the vial frame's stay

    # mechanism: one frame's term weakened by 1e6 on an in-memory copy, unconstrained solves at the extreme offsets
    start = (approach_position_world(vial_pose, act["approach"]["pos"]), q_ref_eff(hmm, vial_pose[1]))

    def resp(weaken):
        dd = [float(((p_s - MM * unconstrained(code, hmm, vial_pose, contacts[s], start, weaken=weaken).pos_vial)
                     @ n_s).max()) for s in (lo, hi)]
        return float((dd[0] - dd[1]) / (hi - lo))
    response = {"as_fitted": resp(None), "vial_term_weakened_1e6": resp("vial"),
                "contact_term_weakened_1e6": resp("contact")}

    # vial-pose invariance of the vial-frame path (one offset)
    sz_chk = -4.0 if -4.0 in offsets else offsets[0]
    other = solve(mdl["main"], act, VIAL_POSE_CHECK, contacts[sz_chk], main=True)
    invariance = float(np.abs(other["tip"] - rows[offsets.index(sz_chk)]["tip"]).max())

    # ---------------------------------------------------------------- record: the bore placement of the first draft
    sz_of = {int(i): float(o[2]) * MM for i, o in zip(run["contact"]["iteration"], run["contact"]["offset_m"])}
    per_offset = []
    for sz in offsets:
        sel = surf["keep"] & np.array([np.isclose(sz_of.get(int(i), np.nan), sz)
                                       for i in run["surface"]["iteration"]])
        a = np.radians(surf["ang_deg"][sel])
        dvec = sz * np.stack([np.zeros_like(a), -np.sin(a), np.cos(a) - 1.0], axis=1)   # normal minus bore
        mag = np.linalg.norm(dvec, axis=1)
        per_offset.append(dict(sz_mm=sz, n_trials=int(sel.sum()), tilt_deg_median=float(np.median(np.abs(
            surf["ang_deg"][sel]))), difference_mm_median=float(np.median(mag)),
            difference_mm_range=(float(mag.min()), float(mag.max())),
            dy_mm_median=float(np.median(dvec[:, 1])), dz_mm_median=float(np.median(dvec[:, 2]))))
    bore = []
    for sz, r in zip(offsets, rows):
        c_b = p_s + np.array([0.0, 0.0, sz])
        s_b = solve(mdl["main"], act, vial_pose, c_b, main=True)
        bore.append(dict(sz_mm=sz, contact_mm=c_b, status=s_b["status"], deepest_z_mm=float(s_b["tip"][:, 2].min()),
                         deepest_below_perpendicular_surface_mm=float(p_s[2] - s_b["tip"][:, 2].min()),
                         max_distance_from_drawn_path_mm=float(np.linalg.norm(s_b["tip"] - r["tip"], axis=1).max())))

    ang_keep = np.abs(surf["ang_deg"][surf["keep"]])
    return dict(K=K, geom=geom, surface=surf, z_s=surf["z_s"], n_surface=surf["n_trials"],
                surface_failures=surf["fail"], tilt=dict(median_deg=float(np.median(ang_keep)),
                                                         min_deg=float(ang_keep.min()), max_deg=float(ang_keep.max())),
                offsets=offsets, contacts=contacts, gauss=gauss, governs=governs, views=views,
                rho_max=rho_max, y_max=y_max, beyond=beyond, rows=rows, slope_n=slope_n,
                slope_n_intercept=float(fit[1]), slope_n_max_residual=resid, slope_n_unconstrained=slope_n_off,
                slope_z=slope_z, prod_shift=prod_shift, response=response, pogs=pogs, invariance=invariance,
                sz_chk=sz_chk, record=dict(per_offset=per_offset, bore_solves=bore), vial_pose=vial_pose,
                ramp=ramp_check(C_SZ))


def check_reference(code, hmm, res) -> float:
    """The product built here is the LQT's reference: the per-component RMS of an unconstrained solve against it
    equals the code's own per_state_rms (components 1-5; the last segment holds the terminal anchor). Returns the
    largest difference, m."""
    c = res["contacts"][res["offsets"][0]]
    vial_pos, vial_quat = res["vial_pose"]
    start = (approach_position_world(res["vial_pose"], res["approach_pos"]), q_ref_eff(hmm, vial_quat))
    syn = unconstrained(code, hmm, res["vial_pose"], c, start)
    prod = product_of_gaussians(code, hmm, res["vial_pose"], c)["prod_model"]
    ref = np.asarray(prod.mu)[syn.state_seq][:, :3]
    err = 0.0
    for seg in syn.per_state_rms[:-1]:
        i, j = seg["start"], seg["end"] + 1
        rms = float(np.sqrt(np.mean(np.sum((syn.pos_world[i:j] - ref[i:j]) ** 2, axis=1))))
        err = max(err, abs(rms - seg["rms_m"]))
    return err


# =================================================================================================================
# The figure
# =================================================================================================================

def draw(res) -> dict:
    K, gauss, governs = res["K"], res["gauss"], res["governs"]
    rows, k_b = res["rows"], STATE_B - 1
    p_s, n_s = res["surface"]["p"], res["surface"]["n"]
    r_v = MM * res["geom"]["r_v"]
    BOTTOM_IN, TOP_IN = 0.34, 0.44                       # x labels below; letters, keys and titles above
    VIEW_GAP_IN = 0.48                                   # between the stacked views of A
    YLAB_IN, B_GAP_IN, RIGHT_IN = 0.46, 0.50, 0.06
    views = (("vial", (-60.0, 8.0), (-12.5, 13.5)), ("contact", (-12.0, 56.0), (-12.5, 13.5)))
    span_z = views[0][1][1] - views[0][1][0]
    span_y = views[0][2][1] - views[0][2][0]
    b_zlim, b_ylim = (-57.5, -18.5), (-15.0, 15.0)
    assert b_zlim[1] <= -MM * res["geom"]["l_shoulder"]  # panel B lies below the shoulder: the wall is at r_v
    r_b = (b_zlim[1] - b_zlim[0]) / (b_ylim[1] - b_ylim[0])
    # equal mm scales within each panel, A and B the same height, filling the page width (fig_tp_motors.py)
    w_view = (FIG_W_IN - YLAB_IN - B_GAP_IN - RIGHT_IN - r_b * VIEW_GAP_IN) / (1.0 + 2.0 * r_b * span_y / span_z)
    h_view = w_view * span_y / span_z
    h_a = 2.0 * h_view + VIEW_GAP_IN
    w_b = r_b * h_a
    fig_h = BOTTOM_IN + h_a + TOP_IN
    fig = plt.figure(figsize=(FIG_W_IN, fig_h))

    def axes_at(x_in, y_in, w_in, h_in):
        return fig.add_axes([x_in / FIG_W_IN, y_in / fig_h, w_in / FIG_W_IN, h_in / fig_h])

    def zlim_of(lim):
        return lim if Z_RIGHT else (lim[1], lim[0])

    # where each component's number sits, outside its own contour (degrees in the frame's y-z plane: 0 = +y,
    # 90 = +z, 180 = -y, 270 = -z), as in fig_tp_motors.py
    label_dir = {"vial": (180, 180, 180, 270, 0, 0), "contact": (180, 180, 180, 270, 0, 0)}
    titles = {"vial": r"seen from $F_{\mathrm{vial\ lip}}$", "contact": r"seen from $F_{\mathrm{contact\ point}}$"}
    axes = {}
    for name, zlim, ylim in views:
        y0 = BOTTOM_IN + (h_view + VIEW_GAP_IN if name == "vial" else 0.0)
        ax = axes_at(YLAB_IN, y0, w_view, h_view)
        axes[name] = ax
        color = C_VIAL if name == "vial" else C_CON
        if name == "vial":                               # the lip at z = 0, outside the mouth radius r_lip
            rl = MM * res["geom"]["r_lip"]
            for s in (1, -1):
                ax.plot([0, 0], [s * rl, s * 40.0], color=C_GLASS_EDGE, lw=2.0, solid_capstyle="butt", zorder=1)
        for path in res["views"][name]:
            ax.plot(path[:, 2], path[:, 1], color=C_PATH, lw=0.3, zorder=1)
        for k in range(K):
            g = gauss[name][k]
            ellipse(ax, g["mean_mm"], g["cov_mm2"], color, filled=governs[k] == name)
        for k in range(K):
            g = gauss[name][k]
            number(ax, *outside(g["mean_mm"], g["cov_mm2"], label_dir[name][k], 1.6), k + 1)
        triad(ax, color)
        ax.set_xlim(*zlim_of(zlim))
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box", anchor="SW")
        ax.set_yticks([-10, 0, 10])
        ax.set_ylabel(r"$y$ (mm)", labelpad=1)
        ax.set_xlabel(r"$z$ (mm)", labelpad=1)
        ax.set_xticks([-60, -40, -20, 0] if name == "vial" else [0, 20, 40])
        ax.set_title(titles[name], fontsize=FS, pad=3)
    top = 1.0 - 0.03 / fig_h
    fig.text(0.02 / FIG_W_IN, top, "A", fontsize=FS_LETTER, fontweight="bold", ha="left", va="top")
    key = [Line2D([], [], marker="o", ls="none", ms=6.0, mfc=(0.5, 0.5, 0.5, 0.3), mec="#555555", mew=0.9,
                  label="smaller position variance"),
           Line2D([], [], marker="o", ls="none", ms=6.0, mfc="none", mec="#555555", mew=0.6, label="larger")]
    fig.legend(handles=key, loc="upper left", bbox_to_anchor=(0.20 / FIG_W_IN, top + 0.01 / fig_h), ncol=2,
               handletextpad=0.2, columnspacing=0.8, borderaxespad=0.0)

    # B transfer, as deployed ------------------------------------------------------------------------------------
    x_b = YLAB_IN + w_view + B_GAP_IN
    ax = axes_at(x_b, BOTTOM_IN, w_b, h_a)
    axes["B"] = ax
    for s in (1, -1):                                    # the vial wall (r_v of tp_scoop's containment)
        ax.axhline(s * r_v, color=C_GLASS_EDGE, lw=0.8, zorder=1)
    u = np.array([0.0, n_s[2], -n_s[1]])                 # along the surface line, in the y-z plane
    ends = [p_s + t * u for t in (-r_v / abs(u[1]), r_v / abs(u[1]))]
    ax.plot([e[2] for e in ends], [e[1] for e in ends], color=C_SURF, lw=1.0, ls=(0, (4, 2)), zorder=2)
    deep_end = p_s + 1.06 * min(res["offsets"]) * n_s    # the normal through p, dotted on the material side
    ax.plot([deep_end[2], p_s[2]], [deep_end[1], p_s[1]], color=C_MUTED, lw=0.6, ls=(0, (1, 1.5)), zorder=2)
    n_tip = p_s + N_ARROW_MM * n_s
    ax.annotate("", xy=(n_tip[2], n_tip[1]), xytext=(p_s[2], p_s[1]),
                arrowprops=dict(arrowstyle="-|>,head_length=0.3,head_width=0.12", lw=0.7, color=C_INK, shrinkA=0,
                                shrinkB=0), zorder=6)
    q = p_s + (N_ARROW_MM + 1.4) * n_s
    ax.text(q[2], q[1], r"$\mathbf{n}$", ha="center", va="center", color=C_INK, path_effects=HALO, zorder=9)
    ax.scatter([p_s[2]], [p_s[1]], s=16, color=C_SURF, edgecolors="white", linewidths=0.6, zorder=8)
    q = p_s + P_LABEL_OFFSET_MM
    ax.text(q[2], q[1], r"$\mathbf{p}$", ha="center", va="center", color=C_INK, path_effects=HALO, zorder=9)
    if B_GAUSSIANS:
        pv = res["pogs"][res["offsets"][0]]["vial"]
        ellipse(ax, pv[0][k_b], pv[1][k_b], C_VIAL, filled=False, lw=0.9)
        shown = [r for r in rows if r["sz_mm"] in B_GAUSS_OFFSETS]
        assert len(shown) == len(B_GAUSS_OFFSETS)
        for r in shown:
            pc, pp = res["pogs"][r["sz_mm"]]["contact"], res["pogs"][r["sz_mm"]]["prod"]
            ellipse(ax, pc[0][k_b], pc[1][k_b], r["color"], filled=False, lw=0.8)
            ellipse(ax, pp[0][k_b], pp[1][k_b], r["color"], filled=False, lw=0.8, ls=(0, (2.4, 1.4)))
    for r in rows:
        ax.plot(r["tip"][:, 2], r["tip"][:, 1], color=r["color"], lw=0.9, zorder=5)
    for r in rows:
        c = r["contact_mm"]
        ax.scatter([c[2]], [c[1]], s=13, color=r["color"], edgecolors="white", linewidths=0.5, zorder=7)
    lab = B_LABELS
    ax.text(*lab["surface"], "estimated\nsurface", ha="center", va="center", color=C_MUTED, linespacing=1.05)
    ax.text(*lab["wall"], "vial wall", ha="right", va="bottom", color=C_MUTED)
    if B_GAUSSIANS:
        ax.text(*lab["component"], "component %d" % STATE_B, ha="left", va="bottom")
        pc = res["pogs"][shown[-1]["sz_mm"]]["contact"]   # the deepest offset's contact-frame contour
        pp = res["pogs"][shown[-1]["sz_mm"]]["prod"]      # and its product contour
        leader(ax, r"$F_{\mathrm{vial\ lip}}$", tuple(outside(pv[0][k_b], pv[1][k_b], 0, 0.0)), lab["F_vial"],
               ha="center", va="bottom")
        leader(ax, r"$F_{\mathrm{contact\ point}}$", tuple(outside(pc[0][k_b], pc[1][k_b], 180, 0.0)),
               lab["F_con"], ha="left", va="center")
        leader(ax, "product", tuple(outside(pp[0][k_b], pp[1][k_b], 180, 0.0)), lab["product"], ha="center",
               va="center")
    ax.set_xlim(*zlim_of(b_zlim))
    ax.set_ylim(*b_ylim)
    ax.set_aspect("equal", adjustable="box", anchor="SW")
    ax.set_xticks([-55, -50, -45, -40, -35, -30, -25, -20])
    ax.set_yticks([-10, -5, 0, 5, 10])
    ax.set_xlabel(r"$z$ in $F_{\mathrm{vial\ lip}}$ (mm)", labelpad=1)
    ax.set_ylabel(r"$y$ (mm)", labelpad=1)
    sz_key = [Line2D([], [], color=r["color"], lw=1.4, label=("$s_z=%+g$" if r is rows[0] else
                                                               "%+g mm" if r is rows[-1] else "%+g") % r["sz_mm"])
              for r in rows]
    leg = ax.legend(handles=sz_key, loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncol=8, handlelength=0.9,
                    handletextpad=0.25, columnspacing=0.45, borderaxespad=0.15)
    for t in leg.get_texts():
        t.set_text(t.get_text().replace("-", "−"))
    fig.text((x_b - 0.30) / FIG_W_IN, top, "B", fontsize=FS_LETTER, fontweight="bold", ha="left", va="top")
    return dict(fig=fig, axes=axes, size_in=(FIG_W_IN, fig_h), w_view=w_view, h_view=h_view, w_b=w_b, h_a=h_a,
                b_lim=(b_zlim, b_ylim), views=views)


#: Panel B labels, plotting coordinates (z, y) in mm, placed against the rendered figure.
N_ARROW_MM = 10.0                                         # the drawn normal n
P_LABEL_OFFSET_MM = np.array([0.0, -1.6, 1.4])           # the label p, from p (x, y, z)
B_LABELS = {"surface": (-28.5, -10.9), "wall": (-19.0, 12.8), "component": (-57.0, -14.6),
            "F_vial": (-53.0, 9.6), "F_con": (-57.2, -11.0), "product": (-39.5, -11.0)}


def text_check(fig) -> None:
    """No text outside the figure and no two text boxes overlapping (labels, ticks, legends, numbers)."""
    import matplotlib.text as mtext
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    boxes = [(t.get_text(), t.get_window_extent(r)) for t in fig.findobj(mtext.Text)
             if t.get_visible() and t.get_text().strip()]
    outside_fig = [s for s, b in boxes if b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > W + 0.5 or b.y1 > H + 0.5]
    overlaps = [(a[0], b[0]) for i, a in enumerate(boxes) for b in boxes[i + 1:] if a[1].overlaps(b[1])]
    assert not outside_fig and not overlaps, (outside_fig, overlaps)


# =================================================================================================================

def main() -> None:
    with tempfile.TemporaryDirectory(prefix="fig_tp_scoop_") as td:
        tmp = Path(td)
        code = load_code(tmp)
        mdl = load_model(code, tmp)
        demos = load_demos(code, mdl["meta"])
        run = load_run_config()
        res = compute(code, mdl, demos, run)
        res["approach_pos"] = run["act"]["approach"]["pos"]
        ref_err = check_reference(code, mdl["hmm"], res)
        assert ref_err < 1e-12, ref_err
        out = draw(res)
        text_check(out["fig"])
        out["fig"].savefig(OUT_PDF)
        out["fig"].savefig(OUT_PNG, dpi=300)
        write_json(code, mdl, demos, run, res, out, ref_err)
    print(f"wrote {OUT_PDF}\n      {OUT_PNG}\n      {OUT_JSON}")


def write_json(code, mdl, demos, run, res, out, ref_err) -> None:
    hmm, meta = mdl["hmm"], mdl["meta"]
    rows, surf = res["rows"], res["surface"]
    numbers = dict(
        what="fig:tp_scoop, the task-parameterized scoop of Section 2.5 drawn from the deployed model and code, in "
             "the geometry the robot used",
        code=dict(main=code["main"], wheel=code["wheel"], source_diff_main_vs_wheel=code["source_diff_main_vs_wheel"],
                  paths_drawn_from="experiment_2 main (tp_lqt), r_factor = 0 passed explicitly",
                  main_vs_wheel_max_path_difference_mm=max(r["main_vs_wheel_max_mm"] for r in rows),
                  versions=code["versions"]),
        model=dict(member=mdl["member"], sha256=mdl["sha256"], meta=meta, K=res["K"], nb_dim=int(hmm.nb_dim),
                   state_layout=dict(per_frame=["x (3, m)", "eta (3, tangent-chart orientation in the vial chart)",
                                                "x_dot (3)", "eta_dot (3)"],
                                     per_frame_dims=12, frames=list(meta["frame_order"]), total_dims=int(hmm.nb_dim),
                                     nb_var_pos=meta["nb_var_pos"], nb_deriv=meta["nb_deriv"],
                                     position_block_of_frame={"vial": [0, 3], "contact": [12, 15]}),
                   schedule=[(int(k) + 1, float(f)) for k, f in hmm.schedule],
                   horizon=int(hmm.default_horizon), rate_hz=float(hmm.rate_hz)),
        demonstrations=dict(n=len(demos["cleans"]), root=str(DEMO_ROOT), names=[n[-6:] for n in demos["names"]],
                            data_sha256=demos["data_sha256"], matches_model=demos["data_sha256"] == meta["data_sha256"],
                            max_rho_mm=res["rho_max"], max_abs_y_mm=res["y_max"],
                            beyond_r_v=res["beyond"], r_v_mm=MM * res["geom"]["r_v"],
                            wall_in_panel_A=False,
                            wall_note=("not drawn in A: the demonstrations listed in beyond_r_v reach further from the "
                                       "bore axis than r_v but stay within |y| < r_v, so a wall drawn in the y-z "
                                       "projection would hide the crossing. Drawn in B, where no drawn path or "
                                       "Gaussian contour reaches it")),
        vial_geometry_tp_scoop={k: (MM * v if k in ("L_blade", "body_radius", "r_v", "r_lip", "l_shoulder") else v)
                                for k, v in res["geom"].items()},
        run=dict(zarr=str(SALT_ZARR), scoop_config=run["act"], offsets_mm=res["offsets"]),
        surface=dict(source="T_vialLip_surface_pose7 of the NaCl run, median over the trials without a surface "
                            "failure", n_trials=res["n_surface"], failures_excluded=res["surface_failures"],
                     p_vial_lip_mm=surf["p"], z_lip_mm_median=res["z_s"],
                     rotation_about_x_lip_deg_median=surf["angle_deg"], n_vial_lip=surf["n"],
                     tilt_from_perpendicular_to_bore_deg=res["tilt"], T_vialLip_surface_mm=surf["T"],
                     drawn="its line in the y-z plane (exact: the rotation is about x_lip only), with p and n"),
        view=dict(z_right=Z_RIGHT, note="as Fig. 3B/3C: z right (out of the opening), y up, x into the page"),
        panel_A=dict(projection="y-z plane of each frame; 1-sd contour of the y-z marginal of the position block",
                     governing_frame_by_component=res["governs"],
                     components={f: [dict(component=k + 1, mean_mm=g["mean_mm"], cov_mm2=g["cov_mm2"],
                                          variance_mm2=g["variance_mm2"],
                                          generalized_variance_mm2=g["generalized_variance_mm2"],
                                          sd_yz_mm=g["sd_yz_mm"]) for k, g in enumerate(res["gauss"][f])]
                                 for f in ("vial", "contact")}),
        panel_B=dict(
            placement="contact point = T_vialLip_surface @ [0, 0, s_z, 1] (tp_gmm_action.surface_frame_contact_point), "
                      "the contact frame keeping the vial-lip orientation",
            component_drawn=STATE_B, gaussian_offsets_drawn=list(B_GAUSS_OFFSETS) if B_GAUSSIANS else [], wall_drawn_at_abs_y_mm=MM *
            res["geom"]["r_v"],
            start=f"approach position {run['act']['approach']['pos']} m in vial_lip, q_ref_eff, at rest "
                  "(tp_gmm_action.py)",
            vial_pose=dict(source="run config fr3_actions.scoop_tp_gmm.preview", pos=res["vial_pose"][0],
                           quat_xyzw=res["vial_pose"][1]),
            vial_pose_invariance_mm=res["invariance"], vial_pose_invariance_offset_mm=res["sz_chk"],
            product_is_lqt_reference_max_rms_difference_m=ref_err,
            deepest_below_surface_slope_mm_per_mm=res["slope_n"],
            deepest_below_surface_slope_note="depth of the deepest point below the estimated surface, along n, per mm "
                                             "of s_z into the material; linear fit over the eight offsets",
            deepest_below_surface_fit_intercept_mm=res["slope_n_intercept"],
            deepest_below_surface_fit_max_residual_mm=res["slope_n_max_residual"],
            deepest_below_surface_slope_unconstrained_mm_per_mm=res["slope_n_unconstrained"],
            deepest_z_along_bore_slope_mm_per_mm=res["slope_z"],
            deepest_point_response_along_n_mm_per_mm=res["response"],
            product_mean_shift_along_n_by_component_mm_per_mm=res["prod_shift"],
            constraint_test=f"hard path against the same SOCP with a vial of radius {LOOSE_RADIUS_M} m; active if "
                            f"they differ by more than {ACTIVE_TOL_MM} mm or a clearance is not positive",
            gaussians_component_drawn={str(sz): dict(
                vial_mean_mm=p["vial"][0][STATE_B - 1], vial_cov_mm2=p["vial"][1][STATE_B - 1],
                contact_mean_mm=p["contact"][0][STATE_B - 1], contact_cov_mm2=p["contact"][1][STATE_B - 1],
                product_mean_mm=p["prod"][0][STATE_B - 1], product_cov_mm2=p["prod"][1][STATE_B - 1])
                for sz, p in res["pogs"].items()},
            rows=[{k: v for k, v in r.items() if k not in ("tip", "root", "state_seq", "color")}
                  | dict(tip_mm=np.round(r["tip"], 4)) for r in rows],
            component_seq=rows[0]["state_seq"]),
        record_bore_vs_normal=dict(
            note=("record only: the first draft moved the contact point along the bore, (0, 0, z_s + s_z); the robot "
                  "moved it along n. difference = s_z (0, -sin a, cos a - 1), |difference| = 2 |s_z| sin(|a|/2), "
                  "per trial with that trial's own tilt a"),
            per_offset=res["record"]["per_offset"], bore_solves=res["record"]["bore_solves"]),
        style=dict(ramp=list(C_SZ), ramp_check=res["ramp"], contact_colour=C_CON, vial_colour=C_VIAL,
                   surface_colour=C_SURF, size_in=out["size_in"], size_mm=[v * 25.4 for v in out["size_in"]],
                   font_pt=FS, letter_pt=FS_LETTER),
        dropped_from_the_ga_figure=["the screw-rate first-order mapping and its Monte Carlo check (TP-scoop's position "
                                    "block is Euclidean position)", "the learned reference M_ref (+): TP-scoop has a "
                                    "reference orientation q_ref but no reference pose"],
    )
    OUT_JSON.write_text(json.dumps(numbers, indent=1, default=_jsonable))


def _jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(type(value))


if __name__ == "__main__":
    main()
