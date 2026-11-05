"""Merged Table II (tab:depth_precision): per offset and material, the balance statistics of the
carried mass (mean, SD, CV) over every valid trial, next to the vision estimate's detection count and
leave-one-out MAE over the scoops that were not empty (N_det/N; "---" where every scoop at an offset was empty).

Replaces the separate Table IV (tab:vision_precision). Numbers come from table1_depth_mass.per_depth
and table3_vision_fit.analyse, so they are identical to what those scripts print.
"""
from __future__ import annotations

from common import fmt, load
from table1_depth_mass import per_depth
from table3_vision_fit import analyse

MATS = ["salt", "sugar", "polenta"]


def main() -> None:
    rows = {}
    for m in MATS:
        d = load(m)
        bal = {r["sz"]: r for r in per_depth(d, "carried_mg")}
        vis = {w["sz"]: w for w in analyse(m, d)["rows"]}
        rows[m] = (bal, vis)
    szs = sorted(rows["salt"][0], reverse=True)
    print("--- LaTeX rows: sz, then per material (mean, SD, CV, Ndet, MAE) ---")
    for sz in szs:
        cells = [f"${sz:+.1f}$"]
        for m in MATS:
            b, v = rows[m][0][sz], rows[m][1][sz]
            dag = "$^{\\dagger}$" if b["n_fail"] else ""
            cells += [fmt(b["mean"]) + dag, fmt(b["sd"]), fmt(b["cv"], 1) if b["cv"] < 100 else fmt(b["cv"], 0),
                      f"{v['n_det']}/{v['n']}" if v["n"] else "---", fmt(v["mae"])]
        print("        " + " & ".join(cells) + " \\\\")


if __name__ == "__main__":
    main()
