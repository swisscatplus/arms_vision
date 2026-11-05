"""Reproduce every number, table and data figure of the paper from the trial-level data in paper/data/.

    python paper/reproduce.py

1. Numbers: analysis/make_all.py prints every number of the Results and Tables 1-4. Its output goes to
   output/make_all.txt and is compared line by line with expected/make_all.txt. The three lines of extract_runs.py
   appear only when the run logs (zarr) are in data/; without them they are left out of the comparison.
2. Figures, written to output/: Figures 6, 7 and 8 from the trial-level data, and Figure 5 when the run logs are in
   data/ (python scripts/download_data.py runs). Figures 1 to 3 are illustrations built from photographs, which are
   not published; they are drawn only where the photographs are in data/photos/. Figure 4 needs the scooping model's
   code, which is not public (README.md). Each figure is compared with the pixels of the file the paper includes
   (expected/figures.json).

Exits with 1 if a number differs from the expected output or a script fails. A figure whose pixels differ from the
paper's is reported but does not fail the check: rendering can change with the platform and the matplotlib version.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANALYSIS, OUT, EXPECTED, DATA = ROOT / "analysis", ROOT / "output", ROOT / "expected", ROOT / "data"
EXTRACT_LINE = re.compile(r"^\S+\s+\d+ trials -> ")      # extract_runs.py, printed only with the run logs present
RUN_LOGS = [DATA / "HS1_vision" / "benchmark_2026-09-10-18-52.zarr",
            DATA / "sugar_vision" / "benchmark_2026-09-10-13-46.zarr",
            DATA / "polenta_vision" / "benchmark_2026-09-11-09-08.zarr"]
PHOTOS = [DATA / "photos" / "vision_paper_segmented_materials" / "prompts.txt", DATA / "photos" / "workbench_nobg.png",
          DATA / "photos" / "vialSideView.png"]
# (figure, command run in analysis/, the PNG it writes, what it needs beyond the trial-level data)
FIGURES = [("Figure 1 (fig:materials)", ["fig_materials.py", "--faded"], "fig_materials_faded.png", {"photos"}),
           ("Figure 2 (fig:platform)", ["fig_cameras.py", "--variant", "compact"], "fig_cameras_compact.png", {"photos"}),
           ("Figure 3 (fig:cycle)", ["fig_cycle.py"], "fig_cycle.png", {"photos", "run logs"}),
           ("Figure 5 (fig:examples)", ["fig3_examples.py"], "fig3_examples.png", {"run logs"}),
           ("Figure 6 (fig:benchmark)", ["fig2.py", "--linear", "--name=fig2_benchmark_linear_floor"],
            "fig2_benchmark_linear_floor.png", set()),
           ("Figure 7 (fig:surface_height)", ["fig_surface_height.py"], "fig_surface_height.png", set()),
           ("Figure 8 (fig:deposit)", ["fig_deposit.py"], "fig_deposit_linear.png", set())]


def run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], cwd=ANALYSIS, capture_output=True, text=True)


def pixels_sha256(png: Path) -> str:
    """SHA-256 of the decoded pixels (mode, size and raw bytes), so that file metadata does not count."""
    from PIL import Image
    with Image.open(png) as im:
        return hashlib.sha256(f"{im.mode} {im.size[0]}x{im.size[1]}\n".encode() + im.tobytes()).hexdigest()


def compare(got: str, record: Path) -> bool:
    a, b = record.read_text().splitlines(), got.splitlines()
    if not any(EXTRACT_LINE.match(x) for x in b):          # run logs absent: extract_runs.py printed only [skip]
        a = [x for x in a if not EXTRACT_LINE.match(x)]
    if a == b:
        print(f"  make_all.py: all {len(b)} lines match {record.relative_to(ROOT)}")
        return True
    print(f"  make_all.py: DIFFERS from {record.relative_to(ROOT)}:")
    for line in list(difflib.unified_diff(a, b, "expected", "this run", lineterm="", n=0))[:60]:
        print("    " + line)
    return False


def main() -> int:
    OUT.mkdir(exist_ok=True)
    ok = True
    print("1. Numbers (analysis/make_all.py)")
    r = run(["make_all.py"])
    (OUT / "make_all.txt").write_text(r.stdout)
    if r.returncode:
        print(r.stderr)
        ok = False
    ok &= compare(r.stdout, EXPECTED / "make_all.txt")

    print("2. Figures (written to output/)")
    have = {"photos": all(p.exists() for p in PHOTOS), "run logs": all(z.exists() for z in RUN_LOGS)}
    paper = json.loads((EXPECTED / "figures.json").read_text())
    for label, args, png, needs in FIGURES:
        missing = sorted(n for n in needs if not have[n])
        if missing:
            why = (["the photographs, which are not published"] if "photos" in missing else []) + \
                  (["the run logs (python scripts/download_data.py runs)"] if "run logs" in missing else [])
            print(f"  {label}: skipped, it needs {', and '.join(why)}")
            continue
        out = OUT / png
        before = out.stat().st_mtime_ns if out.exists() else -1
        r = run(args)
        if r.returncode or not out.exists() or out.stat().st_mtime_ns == before:
            print(f"  {label}: {' '.join(args)} FAILED (exit {r.returncode}): {r.stderr.strip()[-300:]}")
            ok = False
            continue
        same = pixels_sha256(out) == paper[png]["pixels_sha256"]
        print(f"  {label}: {' '.join(args)} -> output/{png}, "
              + ("pixel-identical to the paper's figure" if same else
                 "pixels differ from the paper's figure (rendering depends on the platform and matplotlib version)"))
    print("  Figure 4 (fig:tp_scoop) not run: it needs the scooping model's code, see README.md")

    print("\nDone: the numbers match the expected output." if ok else "\nSome output differs; see above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
