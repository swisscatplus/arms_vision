"""The demo runs and prints what demo/expected/ holds: the stored pixel counts exactly, the stored surface height and
tilt to float32 rounding (each script exits with 1 otherwise), and the a and b of Table 4 of the paper."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script", ["demo.py", "estimate_surface.py", "estimate_mass.py", "calibrate_mass.py"])
def test_demo_script(script):
    r = subprocess.run([sys.executable, str(ROOT / "demo" / script)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout == (ROOT / "demo" / "expected" / script.replace(".py", ".txt")).read_text()
