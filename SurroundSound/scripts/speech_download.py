#!/usr/bin/env python3
"""
Download speech datasets for the Speech classification head.

Sources:
  1. Mozilla Common Voice 25.0 (English) — single-speaker read speech
  2. CHiME-6 (OpenSLR 150) — multi-speaker far-field meeting speech

Labels:
  single_speaker  — Common Voice clips
  multi_speaker   — CHiME-6 speech segments
  no_speech       — CHiME-6 silence/non-speech gaps

Usage:
  python scripts/speech_download.py --root data/speech
  python scripts/speech_download.py --root data/speech --skip-chime
  python scripts/speech_download.py --root data/speech --skip-cv
  python scripts/speech_download.py --root data/speech --workers 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import requests
from tqdm import tqdm


# ── Remote file registry ──────────────────────────────────────────────────────

@dataclass
class RemoteFile:
    filename: str
    url: str
    md5: Optional[str] = None


# CHiME-6 from OpenSLR
CHIME6_FILES: List[RemoteFile] = [
    RemoteFile(
        filename="CHiME6_train.tar.gz",
        url="https://openslr.trmal.net/resources/150/CHiME6_train.tar.gz",
    ),
    RemoteFile(
        filename="CHiME6_dev.tar.gz",
        url="https://openslr.trmal.net/resources/150/CHiME6_dev.tar.gz",
    ),
    RemoteFile(
        filename="CHiME6_eval.tar.gz",
        url="https://openslr.trmal.net/resources/150/CHiME6_eval.tar.gz",
    ),
    RemoteFile(
        filename="CHiME6_transcriptions.tar.gz",
        url="https://openslr.trmal.net/resources/150/CHiME6_transcriptions.tar.gz",
    ),
]

# Mozilla Common Voice 25.0 — dataset ID on Mozilla Data Collective
CV_DATASET_ID = "cmndapwry02jnmh07dyo46mot"
CV_FILENAME   = "common-voice-en-25.0.tar.gz"


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    root: Path
    skip_cv: bool
    skip_chime: bool
    skip_extract: bool
    workers: int
    chime_splits: List[str]


def parse_args() -> Config:
    ap = argparse.ArgumentParser(
        description="Download speech datasets (Common Voice + CHiME-6)."
    )
    ap.add_argument("--root", type=Path, default=Path("data/speech"),
                    help="Root directory for speech data (default: data/speech)")
    ap.add_argument("--skip-cv", action="store_true",
                    help="Skip Mozilla Common Voice download.")
    ap.add_argument("--skip-chime", action="store_true",
                    help="Skip CHiME-6 download.")
    ap.add_argument("--skip-extract", action="store_true",
                    help="Download only, skip extraction.")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel download threads (default: 4).")
    ap.add_argument("--chime-splits", nargs="+",
                    default=["train", "dev", "eval", "transcriptions"],
                    choices=["train", "dev", "eval", "transcriptions"],
                    help="Which CHiME-6 splits to download (default: all).")
    args = ap.parse_args()
    return Config(
        root=args.root,
        skip_cv=args.skip_cv,
        skip_chime=args.skip_chime,
        skip_extract=args.skip_extract,
        workers=args.workers,
        chime_splits=args.chime_splits,
    )


# ── Download utilities ────────────────────────────────────────────────────────

_print_lock = threading.Lock()


def compute_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(
    url: str,
    dest: Path,
    expected_md5: Optional[str] = None,
    overall_pbar: Optional[tqdm] = None,
    headers: Optional[dict] = None,
) -> None:
    """Resumable download with optional MD5 verification. Skips if already done."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    # Already complete?
    if dest.exists():
        if expected_md5:
            actual = compute_md5(dest)
            if actual == expected_md5:
                with _print_lock:
                    print(f"[SKIP] {dest.name} (verified)")
                if overall_pbar:
                    overall_pbar.update(1)
                return
            else:
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
    resume_pos  = tmp.stat().st_size if tmp.exists() else 0
    req_headers = dict(headers or {})
    if resume_pos > 0:
        req_headers["Range"] = f"bytes={resume_pos}-"

    with _print_lock:
        action = f"[RESUME@{resume_pos//1024//1024}MB]" if resume_pos else "[DOWNLOAD]"
        print(f"{action} {dest.name}")

    try:
        with requests.get(url, stream=True, headers=req_headers, timeout=60) as r:
            if r.status_code == 416:
                resume_pos = 0
                r = requests.get(url, stream=True, headers=headers or {}, timeout=60)
            r.raise_for_status()

            total = int(r.headers.get("Content-Length", 0)) or None
            if total and resume_pos:
                total += resume_pos

            mode = "ab" if resume_pos else "wb"
            with tqdm(
                total=total, initial=resume_pos,
                unit="B", unit_scale=True, unit_divisor=1024,
                desc=dest.name[:35], leave=False,
            ) as pbar:
                with tmp.open(mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

        tmp.replace(dest)

        if expected_md5:
            actual = compute_md5(dest)
            if actual != expected_md5:
                raise RuntimeError(f"MD5 mismatch for {dest.name}")

    except Exception as e:
        with _print_lock:
            print(f"[ERROR] {dest.name}: {e}")
        raise

    if overall_pbar:
        overall_pbar.update(1)


def download_files_parallel(
    files: List[RemoteFile],
    root: Path,
    workers: int,
    group_name: str,
    extra_headers: Optional[dict] = None,
) -> None:
    print(f"\n=== {group_name} ({len(files)} files, {workers} workers) ===")
    with tqdm(total=len(files), unit="file", desc=group_name, position=0) as overall_pbar:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    download_file,
                    rf.url,
                    root / rf.filename,
                    rf.md5,
                    overall_pbar,
                    extra_headers,
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


# ── Mozilla Common Voice download ─────────────────────────────────────────────

def get_cv_presigned_url(api_key: str) -> Optional[str]:
    """Fetch a fresh presigned download URL from Mozilla Data Collective."""
    resp = requests.post(
        f"https://mozilladatacollective.com/api/datasets/{CV_DATASET_ID}/download",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("downloadUrl") or data.get("url") or data.get("download_url")


def download_common_voice(root: Path) -> None:
    """
    Download Common Voice via Mozilla Data Collective presigned URL API.
    Requires MOZILLA_DC_API_KEY environment variable.
    Handles resume by fetching a fresh presigned URL each attempt
    (presigned URLs are single-use / time-limited, so we re-fetch on resume).
    Skips if file already exists.
    """
    dest = root / CV_FILENAME
    tmp  = dest.with_suffix(dest.suffix + ".part")

    if dest.exists():
        print(f"[SKIP] {CV_FILENAME} already exists")
        return

    api_key = os.environ.get("MOZILLA_DC_API_KEY", "").strip()
    if not api_key:
        print(
            "[ERROR] MOZILLA_DC_API_KEY not set.\n"
            "        Set it with:\n"
            "          [System.Environment]::SetEnvironmentVariable('MOZILLA_DC_API_KEY', 'your-key', 'User')\n"
            "        Or download manually from https://commonvoice.mozilla.org/en/datasets\n"
            "        and place as: data/speech/common-voice-en-25.0.tar.gz"
        )
        return

    # Presigned URLs expire, so we always fetch a fresh one.
    # If a .part file exists, we try to resume using Range header.
    MAX_RETRIES = 10
    CHUNK_SIZE  = 1 << 20  # 1 MB

    for attempt in range(1, MAX_RETRIES + 1):
        resume_pos = tmp.stat().st_size if tmp.exists() else 0

        print(f"[INFO] Fetching presigned URL (attempt {attempt}/{MAX_RETRIES})...")
        try:
            download_url = get_cv_presigned_url(api_key)
            if not download_url:
                print("[ERROR] Could not find download URL in API response.")
                return
        except Exception as e:
            print(f"[ERROR] Failed to get presigned URL: {e}")
            return

        headers = {}
        if resume_pos > 0:
            headers["Range"] = f"bytes={resume_pos}-"
            print(f"[RESUME] {CV_FILENAME} from {resume_pos // 1024 // 1024}MB")
        else:
            print(f"[DOWNLOAD] {CV_FILENAME}")

        try:
            with requests.get(
                download_url, stream=True, headers=headers,
                timeout=(10, 300),   # (connect timeout, read timeout) — 5 min read timeout
            ) as r:
                if r.status_code == 416:
                    # Presigned URL may not support range — restart
                    resume_pos = 0
                    if tmp.exists():
                        tmp.unlink()
                    r = requests.get(download_url, stream=True, timeout=(10, 300))
                r.raise_for_status()

                total = int(r.headers.get("Content-Length", 0)) or None
                if total and resume_pos:
                    total += resume_pos

                mode = "ab" if resume_pos else "wb"
                with tqdm(
                    total=total, initial=resume_pos,
                    unit="B", unit_scale=True, unit_divisor=1024,
                    desc=CV_FILENAME[:35], leave=True,
                ) as pbar:
                    with tmp.open(mode) as f:
                        for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))

            # Success — rename to final destination
            tmp.replace(dest)
            print(f"[INFO] Common Voice downloaded -> {dest}")
            return

        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError) as e:
            print(f"[WARN] Download interrupted (attempt {attempt}): {e}")
            print(f"       Will retry with fresh presigned URL...")
            if attempt < MAX_RETRIES:
                import time as _time
                _time.sleep(3)
            continue

        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            return

    print(f"[ERROR] Failed after {MAX_RETRIES} attempts. "
          f"Partial file kept at {tmp} — rerun to resume.")


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_tar(tar_path: Path, dest: Path) -> None:
    if not tar_path.exists():
        print(f"[WARN] Archive missing, skipping: {tar_path.name}")
        return
    # Skip if already extracted
    if dest.exists() and any(dest.iterdir()):
        print(f"[SKIP] {tar_path.name} already extracted to {dest}")
        return
    print(f"[EXTRACT] {tar_path.name} -> {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        members = tf.getmembers()
        with tqdm(total=len(members), unit="file", desc=tar_path.name[:35]) as pbar:
            for member in members:
                tf.extract(member, dest)
                pbar.update(1)


def extract_all_chime(root: Path, splits: List[str]) -> None:
    chime_dir = root / "CHiME6"
    chime_dir.mkdir(parents=True, exist_ok=True)

    split_map = {
        "train":          root / "CHiME6_train.tar.gz",
        "dev":            root / "CHiME6_dev.tar.gz",
        "eval":           root / "CHiME6_eval.tar.gz",
        "transcriptions": root / "CHiME6_transcriptions.tar.gz",
    }

    for split in splits:
        out_dir = chime_dir / split if split != "transcriptions" else chime_dir / "transcriptions"
        extract_tar(split_map[split], out_dir)

    print("\n[INFO] CHiME-6 extraction status:")
    for split in splits:
        d = chime_dir / (split if split != "transcriptions" else "transcriptions")
        print(f"  {'✓' if d.exists() else '✗ MISSING'}  {d}")


def extract_cv(root: Path) -> None:
    cv_tar = root / CV_FILENAME
    cv_dir = root / "CommonVoice"
    extract_tar(cv_tar, cv_dir)
    print(f"\n[INFO] Common Voice extracted to: {cv_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = parse_args()
    cfg.root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Speech data root: {cfg.root}")
    print(f"[INFO] Workers: {cfg.workers}")

    split_to_file = {
        rf.filename.replace("CHiME6_", "").replace(".tar.gz", ""): rf
        for rf in CHIME6_FILES
    }
    chime_to_download = [split_to_file[s] for s in cfg.chime_splits if s in split_to_file]

    # ── Downloads ─────────────────────────────────────────────────────────────
    if not cfg.skip_chime and chime_to_download:
        trans = [f for f in chime_to_download if "transcription" in f.filename]
        audio = [f for f in chime_to_download if "transcription" not in f.filename]

        if trans:
            download_files_parallel(trans, cfg.root, workers=1,
                                    group_name="CHiME-6 transcriptions")
            if not cfg.skip_extract:
                extract_tar(cfg.root / trans[0].filename,
                            cfg.root / "CHiME6" / "transcriptions")

        if audio:
            download_files_parallel(audio, cfg.root,
                                    workers=min(cfg.workers, len(audio)),
                                    group_name="CHiME-6 audio")
    else:
        print("[INFO] Skipping CHiME-6 download.")

    if not cfg.skip_cv:
        download_common_voice(cfg.root)
    else:
        print("[INFO] Skipping Common Voice download.")

    # ── Extraction ────────────────────────────────────────────────────────────
    if not cfg.skip_extract:
        if not cfg.skip_chime:
            audio_splits = [s for s in cfg.chime_splits if s != "transcriptions"]
            if audio_splits:
                extract_all_chime(cfg.root, audio_splits)

        if not cfg.skip_cv:
            extract_cv(cfg.root)
    else:
        print("[INFO] Skipping extraction (--skip-extract).")

    print("\n[DONE] Speech download script finished.")
    print(f"       Data root: {cfg.root.resolve()}")


if __name__ == "__main__":
    main()