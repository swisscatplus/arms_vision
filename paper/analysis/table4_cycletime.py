"""Table IV (tab:cycletime): inter-event durations per trial.

Every duration is the gap between two OBSERVATION INSTANTS recorded by the run
logger, so it includes the UR3e transport between them. The recorder does not
stamp the tare steps; they are inside the inter-trial gap. T_v (segmentation
latency) is not in the run logs and is entered from a separate measurement.
"""
from __future__ import annotations

import numpy as np

from common import LABEL, MATERIALS, NS, load

T_V_SEC: float | None = None  # <- fill from the timed /segment_image measurement (mean, s)

# (label, from-key, to-key) in execution order
ARCS = [
    ("Weigh source vial (before) $\\to$ trial-start image", "t_source_before_ns", "t_obs0_ns"),
    ("Surface estimate (side image)", "t_obs0_ns", "t_surface_ns"),
    ("Move to photo pose (before-scoop image)", "t_surface_ns", "t_obs1_ns"),
    ("Scoop, return to photo pose (after-scoop image)", "t_obs1_ns", "t_obs2_ns"),
    ("Vial exchange (before-deposit image)", "t_obs2_ns", "t_obs3_ns"),
    ("Deposit, return to photo pose (after-deposit image)", "t_obs3_ns", "t_obs4_ns"),
    ("Weigh source vial (after)", "t_obs4_ns", "t_source_after_ns"),
    ("Weigh goal vial", "t_source_after_ns", "t_goal_ns"),
]


def main() -> None:
    for mat in MATERIALS:
        try:
            d = load(mat)
        except FileNotFoundError:
            continue
        print(f"\n=== Table IV: {LABEL[mat]} (n = {len(d['iteration'])} trials) ===")
        total = np.zeros(len(d["iteration"]))
        for label, a, b in ARCS:
            g = (d[b] - d[a]) / NS
            valid = np.isfinite(g) & (d[a] > 0) & (d[b] > 0)
            g = g[valid]
            total[valid] += g
            print(f"  {label:<55s} {g.mean():6.1f} $\\pm$ {g.std(ddof=1):4.1f} s   (n={len(g)})")
        span = (d["t_goal_ns"] - d["t_source_before_ns"]) / NS
        per = np.diff(np.sort(d["t_source_before_ns"])) / NS
        gap = per - span[:-1]
        print(f"  {'Within-trial span (first weigh -> last weigh)':<55s} {span.mean():6.1f} $\\pm$ {span.std(ddof=1):4.1f} s")
        print(f"  {'Between trials (clean, restart, return, tare, fetch)':<55s} {gap.mean():6.1f} $\\pm$ {gap.std(ddof=1):4.1f} s")
        print(f"  {'Trial period':<55s} {per.mean():6.1f} $\\pm$ {per.std(ddof=1):4.1f} s   (max {per.max():.0f})")
        tb_goal = (d["t_goal_ns"] - d["t_source_after_ns"]) / NS
        tb_src = (d["t_source_after_ns"] - d["t_obs4_ns"]) / NS
        print(f"  Balance arcs: goal-vial weigh {tb_goal.mean():.1f} s, source-vial weigh (after) {tb_src.mean():.1f} s; four balance visits per trial")
        if T_V_SEC is not None:
            N, k = 40, 5
            print(f"  T_v = {T_V_SEC:.2f} s -> replacing the goal-vial weigh on {N-k} of {N} trials saves {(N-k)*(tb_goal.mean()-T_V_SEC)/60:.1f} min per {N}-trial run ({100*(N-k)*(tb_goal.mean()-T_V_SEC)/(N*per.mean()):.0f} % of run time)")
        else:
            print("  T_v NOT SET (table4_cycletime.T_V_SEC) -- fill from the timed segmentation call before quoting a saving")


if __name__ == "__main__":
    main()
