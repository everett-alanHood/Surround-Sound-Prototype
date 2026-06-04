#!/usr/bin/env python3
"""
Download and unpack the FSD50K dataset for the Events head.

- Downloads multi-part dev/eval audio zips + metadata zips from Zenodo.
- Verifies MD5 checksums.
- Supports parallel downloads with --workers.
- Supports resumable downloads (Range requests).
- Merges split zips (z01, z02, ..., .zip) into a single unsplit .zip.
- Extracts split zips using 7-Zip (handles multi-disk zip format).
- Falls back to Python's zipfile for simple single-part zips.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm


# ── 7-Zip detection ───────────────────────────────────────────────────────────

def find_7zip() -> Optional[str]:
    """Locate 7-Zip executable on Windows or Unix."""
    candidates = [
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
        "7z",   # Unix / on PATH
        "7za",
    ]
    for c in candidates:
        try:
            result = subprocess.run(
                [c, "i"], capture_output=True, timeout=5
            )
            if result.returncode == 0:
                return c
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None

SEVENZ = find_7zip()


# ── Remote file registry ──────────────────────────────────────────────────────

@dataclass
class RemoteFile:
    filename: str
    url: str
    md5: str


REMOTES: Dict[str, List[RemoteFile]] = {
    "FSD50K.dev_audio": [
        RemoteFile("FSD50K.dev_audio.zip",  "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.zip?download=1",  "c480d119b8f7a7e32fdb58f3ea4d6c5a"),
        RemoteFile("FSD50K.dev_audio.z01",  "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z01?download=1",  "faa7cf4cc076fc34a44a479a5ed862a3"),
        RemoteFile("FSD50K.dev_audio.z02",  "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z02?download=1",  "8f9b66153e68571164fb1315d00bc7bc"),
        RemoteFile("FSD50K.dev_audio.z03",  "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z03?download=1",  "1196ef47d267a993d30fa98af54b7159"),
        RemoteFile("FSD50K.dev_audio.z04",  "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z04?download=1",  "d088ac4e11ba53daf9f7574c11cccac9"),
        RemoteFile("FSD50K.dev_audio.z05",  "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z05?download=1",  "81356521aa159accd3c35de22da28c7f"),
    ],
    "FSD50K.eval_audio": [
        RemoteFile("FSD50K.eval_audio.zip", "https://zenodo.org/record/4060432/files/FSD50K.eval_audio.zip?download=1", "6fa47636c3a3ad5c7dfeba99f2637982"),
        RemoteFile("FSD50K.eval_audio.z01", "https://zenodo.org/record/4060432/files/FSD50K.eval_audio.z01?download=1", "3090670eaeecc013ca1ff84fe4442aeb"),
    ],
    "ground_truth": [
        RemoteFile("FSD50K.ground_truth.zip", "https://zenodo.org/record/4060432/files/FSD50K.ground_truth.zip?download=1", "ca27382c195e37d2269c4c866dd73485"),
    ],
    "metadata": [
        RemoteFile("FSD50K.metadata.zip", "https://zenodo.org/record/4060432/files/FSD50K.metadata.zip?download=1", "b9ea0c829a411c1d42adb9da539ed237"),
    ],
    "documentation": [
        RemoteFile("FSD50K.doc.zip", "https://zenodo.org/record/4060432/files/FSD50K.doc.zip?download=1", "3516162b82dc2945d3e7feba0904e800"),
    ],
}


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    root: Path
    skip_audio: bool
    skip_meta: bool
    skip_extract: bool
    workers: int


def parse_args() -> Config:
    ap = argparse.ArgumentParser(
        description="Download and unpack the FSD50K dataset for Events."
    )
    ap.add_argument("--root", type=Path, default=Path("data/events/FSD50K"),
                    help="Root directory where FSD50K will live (default: data/events/FSD50K)")
    ap.add_argument("--skip-audio", action="store_true",
                    help="Skip downloading dev/eval audio archives.")
    ap.add_argument("--skip-meta", action="store_true",
                    help="Skip downloading ground_truth/metadata/doc archives.")
    ap.add_argument("--skip-extract", action="store_true",
                    help="Skip extraction/unsplitting; only download files.")
    ap.add_argument("--workers", type=int, default=4,
                    help="Number of parallel download threads (default: 4).")
    args = ap.parse_args()
    return Config(
        root=args.root,
        skip_audio=args.skip_audio,
        skip_meta=args.skip_meta,
        skip_extract=args.skip_extract,
        workers=args.workers,
    )


# ── MD5 ───────────────────────────────────────────────────────────────────────

def compute_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_md5(path: Path, expected: str) -> bool:
    actual = compute_md5(path)
    if actual != expected:
        raise RuntimeError(
            f"MD5 mismatch for {path.name} "
            f"(expected {expected}, got {actual})"
        )
    return True


# ── Resumable download ────────────────────────────────────────────────────────

# Lock so tqdm bars from parallel threads don't interleave
_print_lock = threading.Lock()


def download_file(
    url: str,
    dest: Path,
    expected_md5: Optional[str] = None,
    overall_pbar: Optional[tqdm] = None,
) -> None:
    """
    Download url to dest with resume support.
    If dest exists and MD5 matches, skip entirely.
    If dest exists but is incomplete (partial), resume from current size.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    # Already complete?
    if dest.exists():
        if expected_md5:
            try:
                verify_md5(dest, expected_md5)
                with _print_lock:
                    print(f"[SKIP] {dest.name} (verified)")
                if overall_pbar:
                    overall_pbar.update(1)
                return
            except RuntimeError:
                with _print_lock:
                    print(f"[WARN] {dest.name} MD5 mismatch, re-downloading")
                dest.unlink()
        else:
            with _print_lock:
                print(f"[SKIP] {dest.name} already exists")
            if overall_pbar:
                overall_pbar.update(1)
            return

    # Resume from partial?
    resume_pos = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={resume_pos}-"} if resume_pos > 0 else {}

    with _print_lock:
        action = f"[RESUME@{resume_pos//1024//1024}MB]" if resume_pos else "[DOWNLOAD]"
        print(f"{action} {dest.name}")

    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        # 416 = range not satisfiable (server doesn't support resume or file is complete)
        if r.status_code == 416:
            resume_pos = 0
            r = requests.get(url, stream=True, timeout=60)
            r.raise_for_status()
        else:
            r.raise_for_status()

        total = int(r.headers.get("Content-Length", 0)) or None
        if total and resume_pos:
            total += resume_pos  # full file size for display

        mode = "ab" if resume_pos else "wb"
        with tqdm(
            total=total,
            initial=resume_pos,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest.name[:30],
            leave=False,
        ) as pbar:
            with tmp.open(mode) as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if not chunk:
                        continue
                    f.write(chunk)
                    pbar.update(len(chunk))

    # Rename tmp -> dest
    tmp.replace(dest)

    # Verify
    if expected_md5:
        verify_md5(dest, expected_md5)

    if overall_pbar:
        overall_pbar.update(1)


# ── Parallel download group ───────────────────────────────────────────────────

def download_group(
    name: str,
    files: List[RemoteFile],
    root: Path,
    workers: int,
) -> None:
    print(f"\n=== Download group: {name} ({len(files)} files, {workers} workers) ===")

    with tqdm(total=len(files), unit="file", desc=name, position=0) as overall_pbar:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    download_file,
                    rf.url,
                    root / rf.filename,
                    rf.md5,
                    overall_pbar,
                ): rf
                for rf in files
            }
            for future in as_completed(futures):
                rf = futures[future]
                try:
                    future.result()
                except Exception as e:
                    with _print_lock:
                        print(f"[ERROR] {rf.filename}: {e}")


# ── Merge + extract ───────────────────────────────────────────────────────────

def extract_split_zip_7z(first_part: Path, dest: Path) -> None:
    """
    Extract a split zip using 7-Zip. Point at the first part (z01 or zip).
    7-Zip handles multi-disk reassembly automatically.
    """
    if SEVENZ is None:
        raise RuntimeError(
            "7-Zip not found. Install it with: winget install 7zip.7zip\n"
            "Then restart your terminal and re-run."
        )
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[EXTRACT] {first_part.name} (split zip) -> {dest}  [via 7-Zip]")
    cmd = [SEVENZ, "x", str(first_part), f"-o{dest}", "-y"]
    result = subprocess.run(cmd, text=True)
    if result.returncode not in (0, 1):  # 7z returns 1 for warnings
        raise RuntimeError(f"7-Zip failed with return code {result.returncode}")


def extract_zip_python(zip_path: Path, dest: Path) -> None:
    """Extract a simple single-part zip using Python's zipfile."""
    if not zip_path.exists():
        print(f"[WARN] Missing archive, cannot extract: {zip_path}")
        return
    print(f"[EXTRACT] {zip_path.name} -> {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        with tqdm(total=len(members), unit="file", desc=zip_path.name[:30]) as pbar:
            for member in members:
                zf.extract(member, dest)
                pbar.update(1)


def extract_all(cfg: Config) -> None:
    root = cfg.root
    print(f"\n=== Extracting archives under {root} ===")

    if SEVENZ:
        print(f"[INFO] 7-Zip found: {SEVENZ}")
    else:
        print("[WARN] 7-Zip not found — split zip extraction will fail.")
        print("       Install with: winget install 7zip.7zip")

    if not cfg.skip_audio:
        # Dev audio — point 7-Zip at the first part (.z01)
        dev_parts = sorted(
            [root / rf.filename for rf in REMOTES["FSD50K.dev_audio"]],
            key=lambda p: p.suffix,
        )
        if all(p.exists() for p in dev_parts):
            # First part for 7-Zip is the .z01
            first = next((p for p in dev_parts if p.suffix == ".z01"), dev_parts[0])
            extract_split_zip_7z(first, root)
        else:
            missing = [p.name for p in dev_parts if not p.exists()]
            print(f"[WARN] Missing dev audio parts: {missing}")

        # Eval audio
        eval_parts = sorted(
            [root / rf.filename for rf in REMOTES["FSD50K.eval_audio"]],
            key=lambda p: p.suffix,
        )
        if all(p.exists() for p in eval_parts):
            first = next((p for p in eval_parts if p.suffix == ".z01"), eval_parts[0])
            extract_split_zip_7z(first, root)
        else:
            missing = [p.name for p in eval_parts if not p.exists()]
            print(f"[WARN] Missing eval audio parts: {missing}")
    else:
        print("[INFO] Skipping audio extraction (--skip-audio)")

    if not cfg.skip_meta:
        for key in ("ground_truth", "metadata", "documentation"):
            for rf in REMOTES[key]:
                extract_zip_python(root / rf.filename, root)
    else:
        print("[INFO] Skipping metadata/doc extraction (--skip-meta)")

    print("\n[INFO] Extraction finished. Expected directories:")
    for d in ["FSD50K.dev_audio", "FSD50K.eval_audio", "FSD50K.ground_truth", "FSD50K.metadata", "FSD50K.doc"]:
        status = "✓" if (root / d).exists() else "✗ MISSING"
        print(f"  {status}  {root / d}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = parse_args()
    cfg.root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] FSD50K root: {cfg.root}")
    print(f"[INFO] Workers:     {cfg.workers}")

    if not cfg.skip_audio:
        download_group("FSD50K.dev_audio",  REMOTES["FSD50K.dev_audio"],  cfg.root, cfg.workers)
        download_group("FSD50K.eval_audio", REMOTES["FSD50K.eval_audio"], cfg.root, cfg.workers)
    else:
        print("[INFO] Skipping audio downloads (--skip-audio)")

    if not cfg.skip_meta:
        # Metadata files are small — download sequentially to avoid Zenodo rate limits
        for group in ("ground_truth", "metadata", "documentation"):
            download_group(group, REMOTES[group], cfg.root, workers=1)
    else:
        print("[INFO] Skipping metadata/doc downloads (--skip-meta)")

    if not cfg.skip_extract:
        extract_all(cfg)
    else:
        print("[INFO] Skipping extraction (--skip-extract)")

    print("\n[DONE] FSD50K download script finished.")


if __name__ == "__main__":
    main()