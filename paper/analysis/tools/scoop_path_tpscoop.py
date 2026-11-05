"""Generate the illustrative scoop path of fig:cycle (Fig. 3C) with the deployed TP-scoop retrieval.

The scoop path in the JSON schema fig_cycle.py reads, for the method the autonomous runs used: the
model bundled in the deployed wheel (tp_scoop/params/scoop_HS1.model.npz), the product of the two frames' Gaussians
tracked by LQT and the hard containment re-solve, called exactly as arms_ws tp_gmm_action.py calls it (start at
the approach position with the model's reference orientation, safety "hard", horizon, dt and r_factor from the
artifact and code defaults). The path is computed with experiment_2 `main` (tp_lqt) and checked against the wheel,
as in fig_tp_scoop.py, whose loaders this script imports. It is a visualization, not the executed trajectory.

    python paper/analysis/tools/scoop_path_tpscoop.py \\
        --contact-mm 0 -3.020 -36.293 --sz-mm -4

--contact-mm is the contact point fig_cycle.py prints for s_z (p + s_z n on vialSideView.png); fig_cycle.py checks
the file's contact point against its own when it loads the path. Writes
paper/data/fig_cycle/scoop_path_tpscoop_sz<sz>.json. fig_cycle.py draws Fig. 3C with it (its SCOOP_PATH).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np                                                         # noqa: E402

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
import fig_tp_scoop as F                                                   # noqa: E402

OUT_DIR = F.DATA / "fig_cycle"
MM = F.MM


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--contact-mm", nargs=3, type=float, required=True, help="contact point in vial_lip, mm")
    ap.add_argument("--sz-mm", type=float, required=True, help="the offset s_z the contact point was built for")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    contact = np.array(args.contact_mm, float)

    with tempfile.TemporaryDirectory(prefix="scoop_path_tpscoop_") as td:
        tmp = Path(td)
        code = F.load_code(tmp)
        mdl = F.load_model(code, tmp)
        demos = F.load_demos(code, mdl["meta"])
        run = F.load_run_config()
        act, hmm, TL = run["act"], mdl["hmm"], code["tp_lqt"]
        geom = {k: v.default for k, v in __import__("inspect").signature(TL.synthesize.synthesize).parameters.items()
                if k in ("L_blade", "body_radius", "r_v", "r_lip", "l_shoulder", "body_samples_K")}
        vial_pose = (np.asarray(act["preview"]["vial_pos"], float), np.asarray(act["preview"]["vial_quat"], float))

        hard = F.solve(mdl["main"], act, vial_pose, contact, main=True)
        wheel = F.solve(mdl["wheel"], act, vial_pose, contact, main=False)
        other = F.solve(mdl["main"], act, F.VIAL_POSE_CHECK, contact, main=True)
        tip = hard["tip"]
        root = tip + MM * geom["L_blade"] * (-hard["tool"])
        main_vs_wheel = float(np.abs(tip - wheel["tip"]).max())
        assert main_vs_wheel < 1e-9, main_vs_wheel
        assert hard["status"] in act["require_safety_status"], hard["status"]
        clear = F.blade_clearance(code, tip, hard["tool"], geom)
        seq = TL.model.expand_schedule(hmm.schedule, hard["horizon"])
        gap = np.linalg.norm(tip - contact, axis=1)
        k_close, k_deep = int(np.argmin(gap)), int(np.argmin(tip[:, 2]))
        dt = hard["dt"]
        idx = F.decimate_index(len(tip), int(act["stream_max_waypoints"]))
        meta = dict(mdl["meta"], n_states=int(hmm.nb_states),
                    chart=f"tangent chart eta about q_ref, anchored to the {hmm.orientation_ref_frame} frame")
        out = {
            "what": ("spatula-tip path generated offline with the deployed TP-scoop retrieval (product of the two "
                     "frames' Gaussians tracked by LQT, then the hard containment re-solve), for visualization"),
            "frame": "vial_lip (model vial frame): +z along the bore out of the opening, lip plane z = 0; mm",
            "sz_mm": args.sz_mm, "contact_mm": contact.tolist(), "dt_s": dt, "n": len(tip),
            "tip_mm": np.round(tip, 5).tolist(), "root_mm": np.round(root, 5).tolist(),
            "state_seq": [int(k) + 1 for k in seq],
            "converged": bool(hard["status"] in act["require_safety_status"]),
            "contained": bool(clear["wall_mm"] > 0 and clear["lip_mm"] > 0),
            "rounds": 1,
            "closest_to_contact_mm": float(gap[k_close]), "closest_t_s": k_close * dt,
            "deepest_z_mm": float(tip[k_deep, 2]), "deepest_t_s": k_deep * dt,
            "vial_pose_invariance_mm": float(np.abs(other["tip"] - tip).max()),
            "start": (f"deployed start (tp_gmm_action.py): the approach position {act['approach']['pos']} m in vial_lip "
                      "with the model's reference orientation q_ref_eff, at rest"),
            "vial_pose": {"source": "run config fr3_actions.scoop_tp_gmm.preview (NaCl run of 2026-09-10)",
                          "pos": vial_pose[0].tolist(), "quat_xyzw": vial_pose[1].tolist()},
            "model": {"path": mdl["member"], "sha256": mdl["sha256"], "meta": meta,
                      "n_demonstrations": len(demos["cleans"]), "blade_length_m": geom["L_blade"]},
            "repo": {"path": code["main"]["repo"], "commit": code["main"]["commit"], "branch": code["main"]["branch"],
                     "package_dirty": False, "export": code["main"]["export"],
                     "wheel": code["wheel"]["path"], "wheel_sha256": code["wheel"]["sha256"]},
            "safety": act["safety"], "safety_status": hard["status"], "solver": "CLARABEL (cvxpy %s, clarabel %s)" % (
                code["versions"]["cvxpy"], code["versions"]["clarabel"]),
            "main_vs_wheel_max_mm": main_vs_wheel,
            "clearance_mm": {"wall": clear["wall_mm"], "lip": clear["lip_mm"]},
            "streamed_waypoints": len(idx),
            "schema_note": ("same keys as scoop_path_tpga_sz-4.json. converged: TP-scoop has no iterative "
                            "retrieval; True means the hard re-solve returned a status the deployed gate accepts "
                            "(safety_status). contained: the blade clears tp_scoop's wall and lip (clearance_mm). "
                            "rounds: one hard re-solve. The contact point is used as given (along the surface "
                            "normal, as fig_cycle.py and fig_tp_scoop.py build it)."),
        }
    path = args.out or OUT_DIR / f"scoop_path_tpscoop_sz{args.sz_mm:+.0f}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"[path] {hard['status']}; closest to the contact point {gap[k_close]:.3f} mm at {k_close * dt:.2f} s; "
          f"deepest z {tip[k_deep, 2]:.3f} mm at {k_deep * dt:.2f} s; clearance wall {clear['wall_mm']:.2f} mm, lip "
          f"{clear['lip_mm']:.2f} mm; vial-pose invariance {out['vial_pose_invariance_mm']:.1e} mm; wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
