"""Download the data that is not in this repository from its Zenodo record, and unzip it where the scripts read it.

    python scripts/download_data.py demo      # full camera frames of the three demo trials (17 MB), for --sam3
    python scripts/download_data.py runs      # run logs of the three runs (zarr, 3 x 0.35 GB)
    python scripts/download_data.py all
    python scripts/download_data.py runs --from-dir ~/Downloads    # zip files already downloaded by hand

Zenodo record: https://doi.org/10.5281/zenodo.XXXXXXX (to be filled in when the record is published: RECORD below).

Each part is one or more zip files of the record. The script downloads each one, checks it against the MD5 checksum
the record lists, unzips it into place and deletes the zip (unless --keep-zip). An existing destination is left as it
is (delete it to download again); the demo frames are added to demo/data/full/. The run logs are zarr stores (zarr format 2), each zipped from inside the store; the
analysis scripts read them from paper/data/<material>_vision/, and paper/analysis/extract_runs.py also reads a zip
without unzipping it (see paper/README.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = None            # the Zenodo record number, e.g. "1234567", once the record is published
DOI = "10.5281/zenodo.XXXXXXX"
# part -> [(file in the record, folder it is unzipped into)]. The run logs are named after their material in the
# record, and unzip into folders named after the robot's run (the time it started), where the scripts read them.
PARTS = {
    "demo": [("demo_full_frames.zip", ROOT / "demo" / "data" / "full")],
    "runs": [("sodium_chloride_run.zarr.zip", ROOT / "paper/data/HS1_vision/benchmark_2026-09-10-18-52.zarr"),
             ("sugar_run.zarr.zip", ROOT / "paper/data/sugar_vision/benchmark_2026-09-10-13-46.zarr"),
             ("polenta_run.zarr.zip", ROOT / "paper/data/polenta_vision/benchmark_2026-09-11-09-08.zarr")],
}


def record_files() -> dict[str, dict]:
    """name -> {url, size, md5} of every file of the Zenodo record."""
    if RECORD is None:
        raise SystemExit(f"The Zenodo record (doi:{DOI}) is not published yet, so there is nothing to download. Once it "
                         f"is, set RECORD in {Path(__file__).name}, or download the zip files by hand and pass --from-dir.")
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{RECORD}", timeout=60) as r:
        meta = json.load(r)
    out = {}
    for f in meta["files"]:
        algo, _, digest = f.get("checksum", "").partition(":")
        out[f["key"]] = {"url": f["links"].get("self") or f"https://zenodo.org/api/records/{RECORD}/files/{f['key']}/content",
                         "size": f.get("size"), "md5": digest if algo == "md5" else None}
    return out


def download(url: str, size: int | None, md5: str | None, dest: Path) -> None:
    part = dest.with_name(dest.name + ".part")
    h = hashlib.md5()
    done = 0
    with urllib.request.urlopen(url, timeout=60) as r, part.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            h.update(chunk)
            done += len(chunk)
            if size:
                print(f"\r  {dest.name}: {done / 1e6:,.0f} of {size / 1e6:,.0f} MB", end="", flush=True)
    print()
    if md5 and h.hexdigest() != md5:
        part.unlink()
        raise SystemExit(f"{dest.name}: MD5 {h.hexdigest()} does not match the record's {md5}; nothing was unzipped")
    part.rename(dest)


def unzip(zip_path: Path, dest: Path, merge: bool = False) -> int:
    """Unzip into dest (refusing any member that would land outside it) -> number of files. Without merge, the zip is
    unzipped next to dest and moved into place at the end, so an interrupted unzip leaves no partial dest."""
    tmp = dest if merge else dest.with_name(dest.name + ".unzipping")
    if tmp.exists() and not merge:
        shutil.rmtree(tmp)
    with zipfile.ZipFile(zip_path) as z:
        for m in z.infolist():
            if not (tmp / m.filename).resolve().is_relative_to(tmp.resolve()):
                raise SystemExit(f"{zip_path.name}: member {m.filename!r} would be written outside {dest}")
        z.extractall(tmp)
        n = sum(not m.is_dir() for m in z.infolist())
    if not merge:
        tmp.rename(dest)
    return n


def check_demo_frames() -> None:
    """The full frames against the SHA-256 the demo records list."""
    bad = []
    for rec_path in sorted((ROOT / "demo" / "data").glob("*_t*.json")):
        rec = json.loads(rec_path.read_text())
        for name in ("side", "side_mask", "top", "top_mask"):
            f = rec["full_frames"][name]
            p = ROOT / "demo" / "data" / f["file"]
            if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != f["sha256"]:
                bad.append(f["file"])
    print("  full frames match the SHA-256 of demo/data/*.json" if not bad else f"  MISSING OR DIFFERENT: {bad}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("parts", nargs="+", choices=[*PARTS, "all"])
    ap.add_argument("--from-dir", type=Path, help="take the zip files from this folder instead of downloading them")
    ap.add_argument("--keep-zip", action="store_true", help="keep the downloaded zip files (next to their folder)")
    args = ap.parse_args()
    parts = list(PARTS) if "all" in args.parts else list(dict.fromkeys(args.parts))
    files = None
    for part in parts:
        print(f"{part}:")
        for name, dest in PARTS[part]:
            merge = part == "demo"                       # demo/data/full may also hold trials of make_demo_data.py
            if dest.exists() and not merge:
                print(f"  {dest.relative_to(ROOT)} exists, left as it is")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            if args.from_dir:
                zip_path = args.from_dir.expanduser() / name
                if not zip_path.exists():
                    print(f"  {name} not in {args.from_dir}, skipped")
                    continue
                keep = True                               # never delete a file the user put there
            else:
                files = files or record_files()
                if name not in files:
                    raise SystemExit(f"{name} is not in the Zenodo record {RECORD}")
                zip_path = dest.with_name(name)
                download(files[name]["url"], files[name]["size"], files[name]["md5"], zip_path)
                keep = args.keep_zip
            n = unzip(zip_path, dest, merge)
            print(f"  {name} -> {dest.relative_to(ROOT)} ({n} files)")
            if not keep:
                zip_path.unlink()
        if part == "demo" and PARTS["demo"][0][1].exists():
            check_demo_frames()
    return 0


if __name__ == "__main__":
    sys.exit(main())
