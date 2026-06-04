#!/usr/bin/env python3
"""
Preprocess speech clips for the Speech classification head.

- Handles both full clips (Common Voice MP3) and windowed segments (CHiME-6 WAV)
- Resamples to 16 kHz mono
- Normalizes peak amplitude
- Pads or trims to 10 seconds
- Computes 128-band log-mel spectrogram (same spec as events pipeline)
- Saves .npy features

Input:  data/speech/manifests/speech_manifest.csv
Output: data/speech/processed/features/<clip_id>.npy

Usage:
  python scripts/speech_preprocess.py
  python scripts/speech_preprocess.py --workers 8
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


# ── Config ────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Preprocess speech clips to log-mel .npy features.")
    ap.add_argument("--manifest", type=Path,
                    default=Path("data/speech/manifests/speech_manifest.csv"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/speech/processed/features"))
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel preprocessing threads (default: 4)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-process clips that already have .npy files")
    return ap.parse_args()


SAMPLE_RATE     = 16000
TARGET_DURATION = 10.0
TARGET_SAMPLES  = int(TARGET_DURATION * SAMPLE_RATE)
N_MELS          = 128
HOP_LENGTH      = 512
FMAX            = 8000


# ── Audio loading ─────────────────────────────────────────────────────────────

def load_audio(
    audio_path: Path,
    seg_start: Optional[float] = None,
    seg_end: Optional[float] = None,
) -> np.ndarray:
    """
    Load audio from path. If seg_start/seg_end are given, load only that window.
    Returns mono float32 array at SAMPLE_RATE.
    """
    if seg_start is not None and seg_end is not None:
        offset   = seg_start
        duration = seg_end - seg_start
        y, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True,
                            offset=offset, duration=duration)
    else:
        y, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    return y


def normalize_audio(y: np.ndarray) -> np.ndarray:
    peak = np.abs(y).max()
    return y / peak if peak > 0 else y


def pad_or_trim(y: np.ndarray) -> np.ndarray:
    if len(y) < TARGET_SAMPLES:
        return np.pad(y, (0, TARGET_SAMPLES - len(y)))
    return y[:TARGET_SAMPLES]


def compute_logmel(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE,
        n_mels=N_MELS, hop_length=HOP_LENGTH, fmax=FMAX,
    )
    return librosa.power_to_db(mel, ref=np.max)  # (128, T)


# ── Per-clip processing ───────────────────────────────────────────────────────

from typing import Optional

def process_row(row: dict, out_dir: Path, overwrite: bool) -> str:
    clip_id    = row["clip_id"]
    audio_path = Path(row["audio_path"])
    out_path   = out_dir / f"{clip_id}.npy"

    if out_path.exists() and not overwrite:
        return "skip"

    if not audio_path.exists():
        return "missing"

    try:
        seg_start = float(row["seg_start"]) if row.get("seg_start") else None
        seg_end   = float(row["seg_end"])   if row.get("seg_end")   else None

        y = load_audio(audio_path, seg_start, seg_end)
        y = normalize_audio(y)
        y = pad_or_trim(y)

        logmel = compute_logmel(y)
        np.save(out_path, logmel)
        return "ok"

    except Exception as e:
        print(f"[WARN] {clip_id}: {e}")
        return "fail"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    with args.manifest.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"[INFO] Manifest rows:  {len(rows)}")
    print(f"[INFO] Output dir:     {args.out_dir}")
    print(f"[INFO] Workers:        {args.workers}")

    counts = Counter()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_row, row, args.out_dir, args.overwrite): row
            for row in rows
        }
        with tqdm(total=len(rows), unit="clip") as pbar:
            for future in as_completed(futures):
                status = future.result()
                counts[status] += 1
                pbar.update(1)
                pbar.set_postfix(
                    ok=counts["ok"],
                    skip=counts["skip"],
                    fail=counts["fail"],
                    missing=counts["missing"],
                )

    print(f"\n[DONE] ok={counts['ok']}  skip={counts['skip']}  "
          f"fail={counts['fail']}  missing={counts['missing']}")
    print(f"       Features -> {args.out_dir}")


if __name__ == "__main__":
    main()
