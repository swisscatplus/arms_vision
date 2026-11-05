"""paper/reproduce.py passes: every number of the paper matches paper/expected/make_all.txt, and every figure it can
draw with the data present is written."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_reproduce():
    r = subprocess.run([sys.executable, str(ROOT / "paper" / "reproduce.py")], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "the numbers match the expected output" in r.stdout
