"""
Preprocess FSD50K Events audio clips.

V2 improvements:
- Parallel processing with --workers
- Skip already-processed clips (resume support)
- LUFS loudness normalization (via pyloudnorm) with peak fallback
- Absolute path resolution (safe to run from any directory)
- Args for manifest, output dirs, and overwrite flag

Steps:
- Resample to 16 kHz mono
- Normalize loudness (LUFS target: -23 LUFS, fallback: peak)
- Pad or trim to 10 seconds
- Compute 128-band log-mel spectrogram
- Save cleaned WAV + .npy feature

Inputs:
  data/events/manifests/fsd50k_events_manifest.csv

Outputs:
  data/events/processed/wav/<clip_id>.wav
  data/events/processed/features/<clip_id>.npy
"""

import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

try:
    import pyloudnorm as pyln
    HAVE_PYLOUDNORM = True
except ImportError:
    HAVE_PYLOUDNORM = False

# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_EVENTS  = PROJECT_ROOT / "data" / "events"

    ap = argparse.ArgumentParser(description="Preprocess FSD50K events audio to log-mel .npy features.")
    ap.add_argument("--manifest", type=Path,
                    default=DATA_EVENTS / "manifests" / "fsd50k_events_manifest.csv")
    ap.add_argument("--out-root", type=Path,
                    default=DATA_EVENTS / "processed")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel processing threads (default: 4)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-process clips that already have .npy files")
    ap.add_argument("--lufs-target", type=float, default=-23.0,
                    help="LUFS loudness target (default: -23.0). Ignored if pyloudnorm not installed.")
    return ap.parse_args()


# ── Audio constants ───────────────────────────────────────────────────────────

SAMPLE_RATE     = 16000
TARGET_DURATION = 10.0
TARGET_SAMPLES  = int(TARGET_DURATION * SAMPLE_RATE)
N_MELS          = 128
HOP_LENGTH      = 512
FMAX            = 8000


# ── Normalization ─────────────────────────────────────────────────────────────

def normalize_lufs(y: np.ndarray, target_lufs: float = -23.0) -> np.ndarray:
    """
    Normalize to target integrated loudness (LUFS) using pyloudnorm.
    Falls back to peak normalization if signal is too quiet to measure.
    """
    meter = pyln.Meter(SAMPLE_RATE)
    try:
        loudness = meter.integrated_loudness(y)
        if np.isfinite(loudness) and loudness > -70:
            return pyln.normalize.loudness(y, loudness, target_lufs)
    except Exception:
        pass
    # Fallback: peak normalize
    peak = np.abs(y).max()
    return y / peak if peak > 0 else y


def normalize_peak(y: np.ndarray) -> np.ndarray:
    peak = np.abs(y).max()
    return y / peak if peak > 0 else y


# ── Per-clip processing ───────────────────────────────────────────────────────

def preprocess_clip(
    clip_id: str,
    audio_path: Path,
    wav_out: Path,
    feat_out: Path,
    overwrite: bool,
    lufs_target: float,
) -> str:
    out_wav  = wav_out  / f"{clip_id}.wav"
    out_feat = feat_out / f"{clip_id}.npy"

    if out_feat.exists() and not overwrite:
        return "skip"

    if not audio_path.exists():
        return "missing"

    try:
        # Load
        y, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)

        # Normalize
        if HAVE_PYLOUDNORM:
            y = normalize_lufs(y, lufs_target)
        else:
            y = normalize_peak(y)

        # Pad or trim
        if len(y) < TARGET_SAMPLES:
            y = np.pad(y, (0, TARGET_SAMPLES - len(y)))
        else:
            y = y[:TARGET_SAMPLES]

        # Save cleaned WAV
        sf.write(str(out_wav), y, SAMPLE_RATE)

        # Log-mel spectrogram
        mel    = librosa.feature.melspectrogram(
            y=y, sr=SAMPLE_RATE,
            n_mels=N_MELS, hop_length=HOP_LENGTH, fmax=FMAX,
        )
        logmel = librosa.power_to_db(mel, ref=np.max)
        np.save(out_feat, logmel)

        return "ok"

    except Exception as e:
        print(f"[WARN] {clip_id}: {e}")
        return "fail"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    wav_out  = args.out_root / "wav"
    feat_out = args.out_root / "features"
    wav_out.mkdir(parents=True, exist_ok=True)
    feat_out.mkdir(parents=True, exist_ok=True)

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    if HAVE_PYLOUDNORM:
        print(f"[INFO] LUFS normalization enabled (target: {args.lufs_target} LUFS)")
    else:
        print("[INFO] pyloudnorm not installed — using peak normalization.")
        print("       Install with: pip install pyloudnorm")

    # Load manifest
    with args.manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"[INFO] Manifest rows: {len(rows)}")
    print(f"[INFO] Workers:       {args.workers}")
    print(f"[INFO] Output root:   {args.out_root}")

    counts = Counter()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                preprocess_clip,
                row["clip_id"],
                Path(row["audio_path"]),
                wav_out,
                feat_out,
                args.overwrite,
                args.lufs_target,
            ): row["clip_id"]
            for row in rows
        }
        with tqdm(total=len(rows), unit="clip", desc="Preprocessing") as pbar:
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

    print(f"\n[PREPROCESS SUMMARY]")
    print(json.dumps(dict(counts), indent=2))
    print(f"Features -> {feat_out}")


if __name__ == "__main__":
    main()