"""Table III (tab:vision_precision) and the failure taxonomy (sec:stats / sec:results_vision).

Fit: carried mass vs after-scoop mask pixel count (px_obs2), m = a * px^b, per material.
Every statistic here is over common.loaded_scoops: a valid surface estimate and not an empty scoop
(carried < EMPTY_SCOOP_MG, which the balance cannot tell from an empty spatula; their basis is printed at the end).
Failure taxonomy (constants in common.py, quoted in the paper):
  non-detection    : px_obs2 == 0 on a loaded scoop
  mis-segmentation : px_obs2 > 0 and LOO error > MISSEG_REL relative AND > MISSEG_ABS_MG absolute
"""
from __future__ import annotations

import numpy as np

from common import (EMPTY_SCOOP_MG, LABEL, MATERIALS, MISSEG_ABS_MG, MISSEG_REL, SMALL_SCOOP_MG, fmt, load,
                    loaded_scoops)
from ssvision.mass import power_law_fit


def analyse(mat: str, d) -> dict:
    x = d["px_obs2"]
    y = d["carried_mg"]
    ok = loaded_scoops(d)          # valid surface estimate, not an empty scoop
    fitmask = ok & (x > 0) & (y > 0)
    a, b, r2, err, loo, loo_pred = power_law_fit(x[fitmask], y[fitmask])
    rel = loo / y[fitmask]
    it = d["iteration"][fitmask]
    res = dict(a=a, b=b, r2=r2, n=int(fitmask.sum()), n_total=len(x),
               err_med=np.median(err), loo_med=np.median(loo), loo_mean=loo.mean(), loo_max=loo.max(),
               loo_rel_med=100 * np.median(rel), loo_rel_90=100 * np.percentile(rel, 90),
               mass_lo=y[fitmask].min(), mass_hi=y[fitmask].max(), loo_err=loo, y_fit=y[fitmask])
    res["misseg"] = [int(i) for i, r, e in zip(it, rel, loo) if r > MISSEG_REL and e > MISSEG_ABS_MG]
    res["worst"] = [(int(it[k]), float(y[fitmask][k]), float(loo_pred[k]), float(100 * rel[k])) for k in np.argsort(-rel)[:3]]
    zero = ok & (x == 0)
    res["nondet"] = [(int(i), float(c)) for i, c in zip(d["iteration"][zero], y[zero])]
    res["n_empty"] = int(d["empty_scoop"].sum())
    both = ok & (x > 0) & (d["px_obs3"] > 0)
    dd = 100 * (d["px_obs3"][both] - x[both]) / x[both]
    res["repeat_med"], res["repeat_q1"], res["repeat_q3"], res["repeat_max"] = np.median(dd), np.percentile(dd, 25), np.percentile(dd, 75), np.max(np.abs(dd))
    res["detected_loaded"] = int((ok & (x > 0)).sum())
    res["n_ok"] = int(ok.sum())
    res["post_deposit_nonzero"] = int((d["px_obs4"] > 0).sum())
    # per-depth block for the table
    pred_all = np.where(x > 0, a * np.maximum(x, 1) ** b, np.nan)
    # Per-offset MAE is a LEAVE-ONE-OUT error (paper sec:stats): every vision error the
    # paper reports comes from a fit that did not see the trial it predicts. loo_pred is
    # indexed over fitmask, so scatter it back to full-length before slicing by offset.
    loo_all = np.full(len(x), np.nan)
    loo_all[fitmask] = loo_pred
    rows = []
    for sz in sorted(set(d["sz_mm"]), reverse=True):
        m = (d["sz_mm"] == sz) & ok
        det = m & (x > 0)
        mae_m = m & fitmask          # detected AND in the fit (positive carried mass)
        rows.append(dict(sz=sz, n_det=int(det.sum()), n=int(m.sum()), scale=y[m].mean() if m.any() else np.nan,
                         vis=pred_all[det].mean() if det.any() else np.nan,
                         vis_sd=pred_all[det].std(ddof=1) if det.sum() > 1 else np.nan,
                         mae=np.abs(loo_all[mae_m] - y[mae_m]).mean() if mae_m.any() else np.nan))
    res["rows"] = rows
    return res


def fit_table_rows(results: dict) -> list[str]:
    """LaTeX rows for the per-material fit table (tab:vision_fit):
    material, n, mass range, a, b, R2, LOO median, LOO max, LOO relative median, 90th pct."""
    out = []
    for mat, r in results.items():
        mant, ex = f"{r['a']:.2e}".split("e")
        a_tex = f"${mant}\\times10^{{{int(ex)}}}$"
        out.append(f"        {LABEL[mat]} & {r['n']} & {r['mass_lo']:.1f}--{r['mass_hi']:.0f} & {a_tex} & {r['b']:.2f} & {r['r2']:.3f} "
                   f"& {r['loo_med']:.2f} & {r['loo_max']:.1f} & {r['loo_rel_med']:.0f} & {r['loo_rel_90']:.0f} \\\\")
    return out


def main() -> None:
    results = {}
    loaded = {}
    for mat in MATERIALS:
        try:
            d = load(mat)
        except FileNotFoundError:
            continue
        r = analyse(mat, d)
        results[mat] = r
        loaded[mat] = d
        print(f"\n=== Table III: {LABEL[mat]} ===")
        print(f"fit m = a px^b vs CARRIED mass: a = {r['a']:.3e}, b = {r['b']:.3f}, R2 = {r['r2']:.3f}, n = {r['n']} of {r['n_ok']} (range {r['mass_lo']:.2f}-{r['mass_hi']:.1f} mg)")
        print(f"abs error: in-sample median {r['err_med']:.2f} mg; LOO median {r['loo_med']:.2f}, mean {r['loo_mean']:.2f}, max {r['loo_max']:.2f} mg; LOO relative median {r['loo_rel_med']:.1f} %, 90th pct {r['loo_rel_90']:.0f} %")
        print(f"{'sz(mm)':>7} {'Ndet':>6} {'scale':>7} {'vision':>7} {'SD':>6} {'MAE':>6}")
        for w in r["rows"]:
            print(f"{w['sz']:>7.1f} {w['n_det']}/{w['n']:<4} {fmt(w['scale']):>7} {fmt(w['vis']):>7} {fmt(w['vis_sd']):>6} {fmt(w['mae']):>6}")
        print("--- LaTeX rows (sz, Ndet, scale, vision, SD, MAE) ---")
        for w in r["rows"]:
            print(f"        {w['sz']:+.1f} & {w['n_det']}/{w['n']} & {fmt(w['scale'])} & {fmt(w['vis'])} & {fmt(w['vis_sd'])} & {fmt(w['mae'])} \\\\")
        print(f"Loaded-blade detection: {r['detected_loaded']}/{r['n_ok']}; empty scoops (< {EMPTY_SCOOP_MG} mg) excluded: {r['n_empty']}")
        print(f"Non-detections (0 px on a loaded scoop): {len(r['nondet'])} {r['nondet']}")
        print(f"Mis-segmentations (LOO err > {MISSEG_REL*100:.0f} % AND > {MISSEG_ABS_MG} mg): {len(r['misseg'])} {r['misseg']}")
        print("  worst relative LOO errors: " + "; ".join(f"trial {i}: {c:.2f} mg carried, {p:.2f} predicted ({e:.0f} %)" for i, c, p, e in r['worst']))
        print(f"Repeatability of the two loaded-blade reads: median {r['repeat_med']:+.1f} %, IQR {r['repeat_q1']:+.1f}..{r['repeat_q3']:+.1f} %, max |.| {r['repeat_max']:.1f} %")
        print(f"After-deposit reads with a non-empty mask: {r['post_deposit_nonzero']} of {r['n_total']}")
        if mat == "polenta_saltprompt":
            det = ~d["surface_fail"] & (d["px_obs2"] > 0)
            hi, lo = np.argmax(np.where(det, d["carried_mg"], -1)), np.argmin(np.where(det, d["carried_mg"], np.inf))
            print(f"  unsuitable prompt: heaviest detected scoop {d['carried_mg'][hi]:.1f} mg -> {int(d['px_obs2'][hi])} px; "
                  f"lightest detected {d['carried_mg'][lo]:.2f} mg -> {int(d['px_obs2'][lo])} px")
    # basis of EMPTY_SCOOP_MG, from the balance alone (reported runs, surface failures excluded)
    rep = [m for m in loaded if m != "polenta_saltprompt"]
    val = {m: ~loaded[m]["surface_fail"] for m in rep}
    cat = lambda key: np.concatenate([loaded[m][key][val[m]] for m in rep])  # noqa: E731
    carried, lost, emp = cat("carried_mg"), cat("lost_mg"), cat("empty_scoop")
    loss_min = ", ".join(f"{LABEL[m]} {loaded[m]['lost_mg'][val[m]].min():+.3f}" for m in rep)
    n_emp = ", ".join(f"{LABEL[m]} {int(loaded[m]['empty_scoop'].sum())}" for m in rep)
    print(f"\nBasis of EMPTY_SCOOP_MG = {EMPTY_SCOOP_MG} mg ({', '.join(LABEL[m] for m in rep)}):")
    print(f"  most negative deposit loss {lost.min():+.3f} mg ({loss_min}); most negative carried mass {carried.min():+.3f} mg")
    print(f"  empty scoops: n = {int(emp.sum())} ({n_emp}), "
          f"carried {carried[emp].mean():+.3f} +- {carried[emp].std(ddof=1):.3f} mg (mean +- SD), max {carried[emp].max():.3f} mg")
    print(f"  gap: no scoop between {carried[emp].max():.3f} mg (largest empty scoop) and {carried[~emp].min():.3f} mg (smallest loaded scoop), "
          f"so any threshold in between gives the same empty scoops")
    nomask = [(m, i) for m in rep for i, e, p in zip(loaded[m]["iteration"], loaded[m]["empty_scoop"], loaded[m]["px_obs2"]) if e and p == 0]
    print(f"  vision on the empty scoops: {len(nomask)} of {int(emp.sum())} returned no mask; with a mask:")
    for m in rep:
        d = loaded[m]
        for k in np.where(d["empty_scoop"] & (d["px_obs2"] > 0))[0]:
            v = results[m]["a"] * d["px_obs2"][k] ** results[m]["b"]
            print(f"    {LABEL[m]} trial {int(d['iteration'][k])} at sz {d['sz_mm'][k]:+.1f} mm: {int(d['px_obs2'][k])} px, "
                  f"vision {v:.3f} mg, balance {d['carried_mg'][k]:.3f} mg")
    print(f"  loaded scoops: {sum(int(loaded_scoops(loaded[m]).sum()) for m in rep)}, "
          f"detected {sum(results[m]['detected_loaded'] for m in rep)}, non-detections {sum(len(results[m]['nondet']) for m in rep)} "
          f"(carried {min(c for m in rep for _, c in results[m]['nondet']):.2f}-{max(c for m in rep for _, c in results[m]['nondet']):.2f} mg)")
    # range of the per-offset in-sample MAE (Table IV) over the reported materials
    maes = [w["mae"] for m, r in results.items() if m != "polenta_saltprompt" for w in r["rows"] if np.isfinite(w["mae"])]
    print(f"Per-offset MAE (Table IV) over the reported materials: {min(maes):.2f}-{max(maes):.2f} mg")
    # LOO error on scoops of up to SMALL_SCOOP_MG, compared with the deposit loss over the same range (Deposit Action results)
    rep = [m for m in results if m != "polenta_saltprompt"]
    yy = np.concatenate([results[m]["y_fit"] for m in rep])
    ee = np.concatenate([results[m]["loo_err"] for m in rep])
    s = yy <= SMALL_SCOOP_MG
    print(f"LOO error on scoops of up to {SMALL_SCOOP_MG:.0f} mg ({', '.join(LABEL[m] for m in rep)}): n = {int(s.sum())}, "
          f"median {np.median(ee[s]):.2f} mg, max {ee[s].max():.2f} mg")
    # how the fits differ: pixels needed for the same mass, and mass read at the same pixel count, relative to NaCl
    if "salt" in results:
        for m in ("sugar", "polenta"):
            if m not in results:
                continue
            px = [((mass / results[m]["a"]) ** (1 / results[m]["b"])) / ((mass / results["salt"]["a"]) ** (1 / results["salt"]["b"])) for mass in (2, 10, 50)]
            mm = [(results[m]["a"] * n ** results[m]["b"]) / (results["salt"]["a"] * n ** results["salt"]["b"]) for n in (1000, 5000, 15000)]
            print(f"{LABEL[m]} vs NaCl: pixels for the same mass x{px[0]:.2f}/{px[1]:.2f}/{px[2]:.2f} at 2/10/50 mg; "
                  f"mass at the same pixel count x{mm[0]:.2f}/{mm[1]:.2f}/{mm[2]:.2f} at 1k/5k/15k px")
    print("\n--- LaTeX rows for tab:vision_fit (material, n, range mg, a, b, R2, LOO med, LOO max, LOO rel med %, 90th %) ---")
    for line in fit_table_rows(results):
        print(line)


if __name__ == "__main__":
    main()
