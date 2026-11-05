"""The spatula pose at one waypoint of the TP-scoop path of fig:cycle, for a re-taken side-camera photo.

For the path fig_cycle.py draws, paper/data/fig_cycle/scoop_path_tpscoop_sz-4.json (tools/scoop_path_tpscoop.py),
it solves that path again exactly
as scoop_path_tpscoop.py does (the deployed model and code through fig_tp_scoop.py's loaders, the vial pose and the
contact point stored in the path file), checks that the solve reproduces the stored tip path, and writes the pose
of the spatula frame at one sample, in the keys fig_cycle.py reads.

The sample is either given (--t-s) or the one nearest, in the y-z plane of vial_lip, to the blade tip of the
current capture (vial_sideVewWithSpatula.png) as fig_cycle.py prints it (--photo-tip-mm), so that the re-taken
photo shows the spatula where the current one does.

    python paper/analysis/tools/scoop_waypoint_pose_tpscoop.py \\
        --photo-tip-mm -3.96 -26.15

Writes paper/data/fig_cycle/scoop_waypoint_tpscoop.json (WAYPOINT in fig_cycle.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np                                                         # noqa: E402
from scipy.spatial.transform import Rotation as R                          # noqa: E402

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
import fig_tp_scoop as F                                                   # noqa: E402

OUT_DIR = F.DATA / "fig_cycle"
MM = F.MM


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    when = ap.add_mutually_exclusive_group(required=True)
    when.add_argument("--t-s", type=float, help="time along the path, s")
    when.add_argument("--photo-tip-mm", nargs=2, type=float, metavar=("Y", "Z"),
                      help="blade tip of the current capture in vial_lip (y, z), mm, as fig_cycle.py prints it")
    ap.add_argument("--path", type=Path, default=OUT_DIR / "scoop_path_tpscoop_sz-4.json")
    args = ap.parse_args()
    stored = json.loads(args.path.read_text())
    contact = np.array(stored["contact_mm"], float)
    vial_pose = (np.asarray(stored["vial_pose"]["pos"], float), np.asarray(stored["vial_pose"]["quat_xyzw"], float))

    with tempfile.TemporaryDirectory(prefix="scoop_waypoint_tpscoop_") as td:
        tmp = Path(td)
        code = F.load_code(tmp)
        mdl = F.load_model(code, tmp)
        if mdl["sha256"] != stored["model"]["sha256"]:
            raise SystemExit(f"the model changed since {args.path.name} was written")
        act = F.load_run_config()["act"]
        hard = F.solve(mdl["main"], act, vial_pose, contact, main=True)
    if hard["status"] not in act["require_safety_status"]:
        raise SystemExit(f"the hard re-solve returned {hard['status']}")
    tip, dt = hard["tip"], hard["dt"]
    rep = float(np.abs(tip - np.array(stored["tip_mm"])).max())
    print(f"[waypoint] re-solve: status {hard['status']}; max |tip - stored path| {rep:.2e} mm", flush=True)
    if rep > 1e-3:
        raise SystemExit("the re-solve does not reproduce the stored path")

    if args.t_s is not None:
        k = int(round(args.t_s / dt))
        how = f"given, t = {args.t_s} s"
    else:
        photo = np.array(args.photo_tip_mm, float)
        d = np.linalg.norm(tip[:, 1:] - photo, axis=1)
        k = int(np.argmin(d))
        how = (f"nearest sample, in the y-z plane, to the blade tip of the current capture at (y, z) = "
               f"{photo.tolist()} mm: {d[k]:.2f} mm away")
    q_vial_sp = (R.from_quat(vial_pose[1]).inv() * R.from_quat(np.asarray(hard["quat_world"])[k])).as_quat()
    if q_vial_sp[3] < 0:
        q_vial_sp = -q_vial_sp
    p_tip_mm = tip[k]
    root_from_q = p_tip_mm + R.from_quat(q_vial_sp).apply([0.0, 1.0, 0.0]) * stored["model"]["blade_length_m"] * MM
    root_err = float(np.linalg.norm(root_from_q - np.array(stored["root_mm"][k])))
    tool_axis = R.from_quat(q_vial_sp).apply([0.0, -1.0, 0.0])
    ang_bore = float(np.degrees(np.arccos(np.clip(-tool_axis[2], -1, 1))))   # tool axis vs -z (into the vial)
    out = {
        "what": "pose of the spatula frame at one waypoint of the TP-scoop scoop path (fig:cycle retake)",
        "t_s": k * dt, "sample": k, "dt_s": dt, "chosen": how,
        "tip_vial_lip_mm": p_tip_mm.tolist(), "quat_vial_lip_xyzw": q_vial_sp.tolist(),
        "tool_axis_vial_lip": tool_axis.tolist(), "tool_axis_vs_bore_deg": ang_bore,
        "root_check_mm": root_err, "resolve_vs_stored_mm": rep,
        "frame": "spatula tool frame (origin at the tip, +y tip -> handle) in vial_lip; command with "
                 "coordinate_frame: vial_lip",
        "path_file": str(args.path), "model": stored["model"]["path"], "model_sha256": stored["model"]["sha256"],
        "repo_commit": stored["repo"]["commit"],
    }
    path = OUT_DIR / "scoop_waypoint_tpscoop.json"
    path.write_text(json.dumps(out, indent=1))
    p_m = p_tip_mm / MM
    print(f"[waypoint] t = {k * dt:.2f} s (sample {k}; {how}): tip {np.round(p_tip_mm, 3).tolist()} mm in vial_lip; "
          f"quat (xyzw) {np.round(q_vial_sp, 6).tolist()}; tool axis {ang_bore:.2f} deg from the bore; "
          f"root check {root_err:.1e} mm", flush=True)
    print(f"[waypoint] as a pose block:  {{pos: [{p_m[0]:.6f}, {p_m[1]:.6f}, {p_m[2]:.6f}], quat: "
          f"[{q_vial_sp[0]:.6f}, {q_vial_sp[1]:.6f}, {q_vial_sp[2]:.6f}, {q_vial_sp[3]:.6f}], "
          f"coordinate_frame: vial_lip}}", flush=True)
    print(f"[waypoint] wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
