"""Table tab:prompts from data/sam3_prompt_sweep_new.csv: per material, phrasings screened,
the chosen prompt and its validated detection rate / R^2 on the stored frames."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

CSV = Path(__file__).resolve().parent.parent / "data" / "sam3_prompt_sweep_new.csv"
NAME = {"NaCl": "NaCl", "polenta": "Polenta", "glass_beads": "Glass beads"}


def main() -> None:
    rows = list(csv.DictReader(CSV.open()))
    print("=== Stage 0 (blade) ===")
    s0 = [r for r in rows if r["stage"] == "stage0_tool"]
    print(f"  phrasings screened: {len(set(r['prompt'] for r in s0))}")
    for r in s0:
        if r["verdict"]:
            print(f"  {r['prompt']!r}: {r['verdict']} -- {r['notes']}")
    print("\n=== Stage 1 (material) ===")
    by = defaultdict(list)
    for r in rows:
        if r["stage"] == "stage1_material":
            by[r["material"]].append(r)
    print(f"{'material':<12} {'screened':>8} {'chosen prompt':<32} {'detected':>9} {'R2':>6} {'frames':>7}")
    for mat, rs in by.items():
        screened = len(set(r["prompt"] for r in rs))
        val = [r for r in rs if r["detected"] and r["verdict"].lower().startswith(("recommended", "validated"))]
        for v in val[:1]:
            print(f"{NAME.get(mat, mat):<12} {screened:>8} {v['prompt']!r:<32} {v['detected']:>9} {v['r2']:>6} {v['n_material_frames']:>7}   {v['notes'][:60]}")
        orig = [r for r in rs if "ORIGINAL" in r["verdict"]]
        for o in orig:
            print(f"{'':<12} {'':>8} {o['prompt']!r:<32} {o['detected']:>9} {o['r2']:>6}   (original config: {o['verdict']})")
    print("\n=== Confidence margins of the selected prompts on the pilot frames (quoted in sec:results_materials) ===")
    for r in rows:
        if r["margin"] and r["verdict"] in ("", "RECOMMENDED") and r["prompt"] in ("scoop", "white material on the scoop", "yellow material on the scoop") and r["material"] != "glass_beads":
            print(f"  {r['stage']:<15} {r['material']:<8} {r['prompt']!r:<32} min on loaded {r['min_score_material']} ({r['n_material_frames']} frames), "
                  f"max on empty {r['max_score_empty']} ({r['n_empty_frames']} frames), margin {r['margin']} -- {r['notes']}")
    print("\nLaTeX rows (material, phrasings screened, chosen prompt, validated detection):")
    for mat, rs in by.items():
        screened = len(set(r["prompt"] for r in rs))
        val = [r for r in rs if r["detected"] and r["verdict"].lower().startswith(("recommended", "validated"))]
        if val:
            v = val[0]
            print(f"        {NAME.get(mat, mat)} & {screened} & \\texttt{{\"{v['prompt']}\"}} & {v['detected']} \\\\")


if __name__ == "__main__":
    main()
