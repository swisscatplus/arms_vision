"""Run every analysis script and print one consolidated block. Nothing in the paper's
Results should be typed from anywhere else."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ["extract_runs.py", "table1_depth_mass.py", "table3_vision_fit.py", "table2_merged.py", "table5_trials.py", "table4_cycletime.py", "prompt_sweep_table.py", "surface_compare.py"]

if __name__ == "__main__":
    rc = 0
    for s in SCRIPTS:
        print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78, flush=True)   # before the script's own output
        r = subprocess.run([sys.executable, str(HERE / s)], cwd=HERE)
        rc |= r.returncode
    sys.exit(rc)
