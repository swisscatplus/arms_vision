"""Table tab:trials (sec:stats): the trials of the three reported runs, followed to every statistic.

Each bold row is a set of trials that statistics use; the indented rows under it split that set.
  All trials              timing (table4_cycletime.py) and the after-deposit images
    surface failure       common.surface_failures: plotted, excluded from every statistic below
  Valid surface estimate  f_depth (table1_depth_mass.py; tab:depth_precision, unshaded; fig:benchmark A)
    empty scoop           common.empty_scoops: kept in f_depth, excluded below
  Scoop not empty         common.loaded_scoops: f_vision (table3_vision_fit.py) and the deposit (table1_depth_mass.py,
                          fig_deposit.py), split in full: mask returned + no mask. The deposit is reported as
                          measured, so it does not split this set.

Every selection is imported from where the other scripts take it; none is re-implemented here. The asserts
check that every split adds up to its set.

    python paper/analysis/table5_trials.py
"""
from __future__ import annotations

from common import EMPTY_SCOOP_MG, LABEL, load, loaded_scoops
from table3_vision_fit import analyse

MATS = ["salt", "sugar", "polenta"]
# (row label as printed in the paper, key, level): level 0 is a set (bold), 1 and 2 split the set above
ROWS = [
    ("All trials", "all", 0),
    ("Surface-estimation failure", "surface", 1),
    ("Valid surface estimate", "valid", 0),
    (f"Empty scoop ($<{EMPTY_SCOOP_MG}$\\,mg)", "empty", 1),
    ("Scoop not empty", "loaded", 0),
    ("Mask returned", "mask", 1),
    ("Mis-segmented", "misseg", 2),
    ("No mask (non-detection)", "nomask", 1),
]


def counts(mat: str) -> dict[str, int]:
    d = load(mat)
    r = analyse(mat, d)
    valid = ~d["surface_fail"]
    loaded = loaded_scoops(d)
    c = dict(all=len(valid), surface=int((~valid).sum()), valid=int(valid.sum()), empty=int(d["empty_scoop"].sum()),
             loaded=int(loaded.sum()), mask=r["detected_loaded"], misseg=len(r["misseg"]), nomask=len(r["nondet"]),
             after_mask=int((d["px_obs4"] > 0).sum()))
    assert c["valid"] == c["all"] - c["surface"], mat
    assert not (d["empty_scoop"] & d["surface_fail"]).any(), mat
    assert c["loaded"] == c["valid"] - c["empty"], mat
    assert r["n_ok"] == c["loaded"], mat                     # the vision statistics use exactly this set
    assert c["mask"] + c["nomask"] == c["loaded"], mat
    assert c["misseg"] <= c["mask"], mat
    return c


def main() -> None:
    per = {m: counts(m) for m in MATS}
    tot = {k: sum(per[m][k] for m in MATS) for k in per[MATS[0]]}
    print(f"=== tab:trials: {', '.join(LABEL[m] for m in MATS)}, total ===")
    for k, (label, key, level) in enumerate(ROWS):
        if level == 0 and k:
            print("        \\midrule")
        name = f"\\textbf{{{label}}}" if level == 0 else ("\\quad " if level == 1 else "\\qquad ") + label
        print(f"        {name} & " + " & ".join(str(per[m][key]) for m in MATS) + f" & {tot[key]} \\\\")
    after = ", ".join(f"{LABEL[m]} {per[m]['after_mask']}" for m in MATS)
    print(f"After-deposit images with a mask (all trials): {tot['after_mask']} of {tot['all']} ({after})")


if __name__ == "__main__":
    main()
