"""Table I (tab:depth_precision): carried mass by commanded depth, per material.

Surface-estimate failures (common.SURFACE_JUMP_MM) are listed separately and
excluded from the per-depth mean/SD -- see the paper, sec:stats. Empty scoops (common.EMPTY_SCOOP_MG)
stay in the per-depth statistics: the offset was run and the scoop came back empty. They are counted
per offset, and the deposit is reported on the other scoops (common.loaded_scoops) as measured: the
fraction of the carried mass delivered and the mass lost, with no success criterion.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from common import EMPTY_SCOOP_MG, LABEL, MATERIALS, SMALL_SCOOP_MG, fmt, load, loaded_scoops

REPORTED = ("salt", "sugar", "polenta")             # the three runs in the paper
DEPOSIT = {}                                         # per reported material, filled by main() for the summary
SURFACE = {}                                         # per reported material: |surface estimate - run median|, for the summary


def surface_summary() -> None:
    """Discussion, failure modes: how far the surface failures and every other trial lie from their run median
    (the median common.surface_failures uses, over all trials of the run)."""
    if not SURFACE:
        return
    dev = {m: v["dev"][~v["fail"] & np.isfinite(v["dev"])] for m, v in SURFACE.items()}
    top = max(dev, key=lambda m: dev[m].max())
    fails = ", ".join(f"{LABEL[m]} trial {i}: {x:.2f}" for m, v in SURFACE.items() for i, x in zip(v["it"][v["fail"]], v["dev"][v["fail"]]))
    print(f"\n=== Surface estimate vs its run median, {', '.join(LABEL[m] for m in SURFACE)} ===")
    print(f"surface failures: {fails} mm from the run median")
    print(f"largest deviation of any other trial: {dev[top].max():.2f} mm ({LABEL[top]}); per material "
          + ", ".join(f"{LABEL[m]} {v.max():.2f}" for m, v in dev.items()) + " mm")


def deposit_summary() -> None:
    """fig:deposit over the three reported materials, as measured: the negative losses, the delivered fraction,
    and the mass lost on scoops up to and above SMALL_SCOOP_MG."""
    if not DEPOSIT:
        return
    c = np.concatenate([v["c"] for v in DEPOSIT.values()])
    lost = np.concatenate([v["lost"] for v in DEPOSIT.values()])
    mat = np.concatenate([[m] * len(v["c"]) for m, v in DEPOSIT.items()])
    print(f"\n=== Deposit loss (fig:deposit), {', '.join(LABEL[m] for m in DEPOSIT)}: {len(c)} trials ===")
    neg = lost < 0
    if neg.any():
        print(f"negative losses (delivered > carried): {int(neg.sum())} ({', '.join(f'{LABEL[m]} {int((neg & (mat == m)).sum())}' for m in DEPOSIT)}); "
              f"{lost[neg].min():+.3f} to {lost[neg].max():+.3f} mg")
    frac = (c - lost) / c                     # fraction of the carried mass delivered to the goal vial
    minority = frac <= 0.5
    print(f"delivered more than half of the carried mass: {int((~minority).sum())}/{len(c)}; exceptions (material, carried mg, delivered %): "
          + str([(LABEL[m], round(float(a), 2), round(100 * float(f), 1)) for m, a, f in zip(mat[minority], c[minority], frac[minority])]))
    s = c <= SMALL_SCOOP_MG
    print(f"scoops of up to {SMALL_SCOOP_MG:.0f} mg: {int(s.sum())} trials; loss median {np.median(lost[s]):.2f} mg, max {lost[s].max():.2f} mg; "
          f"below 1 mg: {int((lost[s] < 1).sum())}/{int(s.sum())}")
    print(f"mass lost per material (median, 90th percentile), scoops of up to {SMALL_SCOOP_MG:.0f} mg | above {SMALL_SCOOP_MG:.0f} mg:")
    for m in DEPOSIT:
        k = mat == m
        lo, hi = k & s, k & ~s
        rel = 100 * lost[hi] / c[hi]  # above SMALL_SCOOP_MG the loss grows with the carried mass, so also as a percentage
        print(f"  {LABEL[m]:<8} n = {int(lo.sum()):>2}: {np.median(lost[lo]):.2f}, {np.percentile(lost[lo], 90):.2f} mg | "
              f"n = {int(hi.sum()):>2}: {np.median(lost[hi]):.2f}, {np.percentile(lost[hi], 90):.2f} mg "
              f"({np.median(rel):.1f}, {np.percentile(rel, 90):.1f} % of the carried mass)")


def per_depth(d, key="carried_mg"):
    ok = ~d["surface_fail"]
    rows = []
    for sz in sorted(set(d["sz_mm"]), reverse=True):
        m = (d["sz_mm"] == sz) & ok
        v = d[key][m]
        n_fail = int(((d["sz_mm"] == sz) & ~ok).sum())
        sd = v.std(ddof=1) if len(v) > 1 else np.nan
        rows.append(dict(sz=sz, n=len(v), n_fail=n_fail, mean=v.mean() if len(v) else np.nan, sd=sd,
                         cv=100 * sd / v.mean() if len(v) > 1 and v.mean() > 0 else np.nan, vals=v))
    return rows


def main() -> None:
    for mat in MATERIALS:
        try:
            d = load(mat)
        except FileNotFoundError:
            continue
        ok = ~d["surface_fail"]
        print(f"\n=== Table I: {LABEL[mat]} (n={len(d['iteration'])} trials, {int((~ok).sum())} surface failures excluded) ===")
        print(f"{'sz(mm)':>7} {'N':>4} {'carried mean':>12} {'SD':>6} {'CV%':>6} | {'delivered mean':>14} {'SD':>6}")
        rc = per_depth(d, "carried_mg")
        rd = per_depth(d, "delivered_mg")
        for a, b in zip(rc, rd):
            print(f"{a['sz']:>7.1f} {a['n']:>4} {fmt(a['mean']):>12} {fmt(a['sd']):>6} {fmt(a['cv'],1):>6} | {fmt(b['mean']):>14} {fmt(b['sd']):>6}"
                  + (f"   ({a['n_fail']} surface-failure trial excluded)" if a["n_fail"] else ""))
        # LaTeX rows
        print("--- LaTeX rows (sz, N, carried mean, SD, CV) ---")
        for a in rc:
            print(f"        {a['sz']:+.1f} & {a['n']} & {fmt(a['mean'])} & {fmt(a['sd'])} & {fmt(a['cv'],1)} \\\\")
        rho, p = spearmanr(-d["sz_mm"][ok], d["carried_mg"][ok])
        sds = [r["sd"] for r in rc if np.isfinite(r["sd"])]
        print(f"Spearman rho (depth into material vs carried mass) = {rho:.3f} (p = {p:.1e}, n = {int(ok.sum())})")
        print(f"Pooled within-depth SD (RMS) = {np.sqrt(np.mean(np.square(sds))):.2f} mg; SD range {min(sds):.2f}-{max(sds):.2f} mg")
        top = max(rc, key=lambda r: r["mean"] if np.isfinite(r["mean"]) else -1)
        print(f"Largest mean (blade capacity proxy): {top['mean']:.1f} mg at sz = {top['sz']:+.1f} mm (CV {top['cv']:.1f} %)")
        emp = d["empty_scoop"]
        per = ", ".join(f"{sz:+.1f} mm: {int((emp & (d['sz_mm'] == sz)).sum())} of {int((ok & (d['sz_mm'] == sz)).sum())}"
                        for sz in sorted(set(d["sz_mm"][emp]), reverse=True))
        print(f"Empty scoops (carried < {EMPTY_SCOOP_MG} mg, kept in the statistics above): {int(emp.sum())} of {int(ok.sum())}"
              + (f" ({per}); carried {d['carried_mg'][emp].min():+.3f} to {d['carried_mg'][emp].max():+.3f} mg" if emp.any() else ""))
        if (~ok).any():
            for i, z, c in zip(d["iteration"][~ok], d["surface_z_lip_mm"][~ok], d["carried_mg"][~ok]):
                print(f"  surface failure: trial {i}, surface z = {z:.1f} mm vs run median {np.nanmedian(d['surface_z_lip_mm']):.1f} mm, carried {c:.2f} mg")
            # SD at those depths with the surface-failure trial included (quoted in sec:results_depth)
            for r in rc:
                if r["n_fail"]:
                    m = d["sz_mm"] == r["sz"]
                    print(f"  SD at sz = {r['sz']:+.1f} with the failure included: {d['carried_mg'][m].std(ddof=1):.2f} mg (excluded: {r['sd']:.2f} mg)")
        # surface-estimate repeatability: residual SD of the estimated height about its linear trend over the run
        z = d["surface_z_lip_mm"]; it = d["iteration"]
        good = ok & np.isfinite(z)
        if good.sum() > 3:
            slope, icpt = np.polyfit(it[good], z[good], 1)
            resid = z[good] - (slope * it[good] + icpt)
            print(f"Surface estimate over the run: drift {1000*slope:+.0f} um/trial, residual SD about the trend {resid.std(ddof=1):.2f} mm, "
                  f"max |resid| {np.abs(resid).max():.2f} mm (n={int(good.sum())}); run median height {np.nanmedian(z[good]):.1f} mm below the lip")
        if mat in REPORTED:
            SURFACE[mat] = dict(dev=np.abs(z - np.nanmedian(z)), fail=~ok, it=d["iteration"])
        lost = d["lost_mg"][ok]
        print(f"Transfer loss (carried - delivered): median {np.median(lost):.2f} mg, max {lost.max():.2f} mg at sz = {d['sz_mm'][ok][np.argmax(lost)]:+.1f} mm; "
              f"deepest row loss mean {np.mean(d['lost_mg'][ok & (d['sz_mm'] == d['sz_mm'].min())]):.2f} mg")
        # deposit by differential weighing, as measured: fraction of carried mass that reached the goal vial,
        # and the mass lost (carried - delivered); no success criterion
        loaded = loaded_scoops(d)
        eff = d["delivered_mg"][loaded] / d["carried_mg"][loaded]
        lossv = d["lost_mg"][loaded]
        print(f"Deposit (differential weighing, {int(loaded.sum())} scoops that were not empty): delivered/carried median {100*np.median(eff):.1f} %, "
              f"IQR {100*np.percentile(eff,25):.1f}-{100*np.percentile(eff,75):.1f} %, min {100*eff.min():.1f} %")
        print(f"  loss vs carried (fig:deposit): median loss {np.median(lossv):.2f} mg; negative losses: {int((lossv < 0).sum())}")
        if mat in REPORTED:
            DEPOSIT[mat] = dict(c=d["carried_mg"][loaded], lost=lossv)
        for sz in sorted(set(d["sz_mm"]), reverse=True):
            m = loaded & (d["sz_mm"] == sz)
            if m.sum():
                e = d["delivered_mg"][m] / d["carried_mg"][m]
                print(f"    sz={sz:+.1f}: delivered/carried mean {100*e.mean():.1f} % (n={int(m.sum())})")
        near_zero = d["delivered_mg"][np.abs(d["delivered_mg"]) < 0.5]
        if len(near_zero):
            print(f"Near-empty delivered readings: {near_zero.min():+.2f} to {near_zero.max():+.2f} mg (n={len(near_zero)}) -> balance-cycle repeatability")
    surface_summary()
    deposit_summary()


if __name__ == "__main__":
    main()
