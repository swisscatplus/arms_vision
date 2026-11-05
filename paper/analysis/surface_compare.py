"""Vision surface estimate against a force probe: paper/data/surface_compare_2026-09-15-11-22.csv.
Prints every number sec:surface_compare / sec:results_surface / Limitations quote.

Protocol (from the CSV header): NaCl, 5 fill levels x 3 blocks; the vial was rotated about its axis once before
each block; each block = 3 vision reads then 1 force probe at the vial centre.
Column convention: the CSV's two heights are related by 56.70 mm on vision rows and 54.00 mm on probe rows, so the
methods differ by 2.7 mm in `height_above_vial_base_mm` by bookkeeping alone. All comparisons here use
`surface_z_mm` (below the lip), pending confirmation of the probe script's convention (SURFACE_Z_COLUMN)."""
from __future__ import annotations
import csv
import numpy as np
from common import DATA

CSV = DATA / "surface_compare_2026-09-15-11-22.csv"
SURFACE_Z_COLUMN = "surface_z_mm"  # compare in z below the lip (see docstring)


def load():
    with CSV.open() as f:
        rows = [r for r in csv.DictReader(line for line in f if not line.startswith("#"))]
    vis, probe, t_vis, t_probe = {}, {}, [], {}
    prev = None
    for r in rows:
        k = (int(r["level"]), int(r["block"])); t = float(r["t_wall"])
        if r["method"] == "vision":
            vis.setdefault(k, []).append((float(r[SURFACE_Z_COLUMN]), float(r["tilt_deg"]),
                                          float(r["height_above_vial_base_mm"])))
            if r["repeat"] != "1":
                t_vis.append(t - prev)
        elif r["method"] == "probe":
            probe[k] = float(r[SURFACE_Z_COLUMN]); t_probe[k] = t - prev
        prev = t
    return rows, vis, probe, np.array(t_vis), t_probe


if __name__ == "__main__":
    rows, vis, probe, t_vis, t_probe = load()
    keys = sorted(vis)
    levels = sorted({k[0] for k in keys}); blocks = sorted({k[1] for k in keys})
    n_reads = sum(len(v) for v in vis.values()); n_probe = len(probe)
    zv = np.array([np.mean([x[0] for x in vis[k]]) for k in keys])
    hv = np.array([np.mean([x[2] for x in vis[k]]) for k in keys])
    zp = np.array([probe[k] for k in keys])
    blk = np.array([k[1] for k in keys]); lv = np.array([k[0] for k in keys])
    conv = {m: np.mean([float(r["height_above_vial_base_mm"]) + abs(float(r["surface_z_mm"])) for r in rows if r["method"] == m])
            for m in ("vision", "probe")}
    print(f"=== Surface estimate vs force probe (NaCl): {len(levels)} levels x {len(blocks)} blocks; {n_reads} vision reads, {n_probe} probes ===")
    print(f"column convention height+|z|: vision {conv['vision']:.2f} mm, probe {conv['probe']:.2f} mm -> comparing in {SURFACE_Z_COLUMN}")
    print(f"fill range (vision): z {zv.max():.1f} to {zv.min():.1f} mm below lip = {hv.max():.1f} to {hv.min():.1f} mm above base")
    within = np.array([np.std([x[0] for x in vis[k]], ddof=1) for k in keys])
    print(f"vision repeatability, 3 reads without moving anything: SD median {np.median(within):.2f} mm, max {within.max():.2f} mm")
    across = [np.std([zv[i] for i in range(len(keys)) if lv[i] == L], ddof=1) for L in levels]
    print(f"vision across the {len(blocks)} rotations, per level: SD {min(across):.2f} to {max(across):.2f} mm")
    tilt = np.array([x[1] for v in vis.values() for x in v])
    print(f"tilt of the surface line in the image: {tilt.min():.0f} to {tilt.max():.0f} deg over all reads (mean {tilt.mean():.1f})")
    d = zv - zp
    print(f"vision - probe: mean {d.mean():+.2f} mm, SD {d.std(ddof=1):.2f}, range {d.min():+.2f} to {d.max():+.2f} mm")
    first = d[blk == blocks[0]]
    print(f"  first probe of each level (full -> empty): {np.round(first, 2)} mm")
    print(f"  probe resolution: values {sorted(set(zp))} -> 1 mm steps")
    for L in levels:
        p = [probe[(L, b)] for b in blocks]; v = [zv[i] for i in range(len(keys)) if lv[i] == L]
        if len(set(p)) > 1:
            print(f"  level {L}: probe read {p} (later probes {p[0]-min(p):.0f} mm lower than the first); vision moved {max(v)-min(v):.2f} mm")
    tp = np.array([t_probe[k] for k in keys])
    slope = np.polyfit(hv, tp, 1)[0]
    print(f"time per estimate: vision median {np.median(t_vis):.1f} s (range {t_vis.min():.1f}-{t_vis.max():.1f}); probe {tp.min():.0f} to {tp.max():.0f} s, "
          f"{-slope:.1f} s more per mm of fill removed; ratio {tp.min()/np.median(t_vis):.0f} to {tp.max()/np.median(t_vis):.0f}x")

    # Linear agreement with the probe (quoted in the abstract as R^2)
    b1, b0 = np.polyfit(zp, zv, 1)
    r2 = 1 - np.sum((zv - (b0 + b1 * zp)) ** 2) / np.sum((zv - zv.mean()) ** 2)
    print(f"linear agreement: vision = {b1:.3f} * probe + {b0:.2f} mm, R^2 = {r2:.4f} (n = {len(zv)} blocks)")
