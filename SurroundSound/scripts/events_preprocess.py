"""
Preprocess FSD50K Events audio clips:

- Resample to 16 kHz mono
- Normalize (peak)
- Trim/pad to fixed length (10 s)
- Compute log-mel spectrograms
- Save cleaned WAV + .npy features

Inputs:
  data/events/manifests/fsd50k_events_manifest.csv

Outputs:
  data/events/processed/wav/<clip_id>.wav
  data/events/processed/features/<clip_id>.npy
"""

import csv
import json
from pathlib import Path
from collections import Counter

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


MANIFEST = Path("data/events/manifests/fsd50k_events_manifest.csv")

OUT_ROOT = Path("data/events/processed")
WAV_OUT = OUT_ROOT / "wav"
FEAT_OUT = OUT_ROOT / "features"

SAMPLE_RATE = 16000
TARGET_DURATION = 10.0  # seconds
N_MELS = 128
HOP_LENGTH = 512
FMAX = 8000


def normalize_audio(y: np.ndarray) -> np.ndarray:
    peak = np.abs(y).max()
    return y / peak if peak > 0 else y


def preprocess_file(src_path: Path, clip_id: str):
    # load
    y, sr = librosa.load(str(src_path), sr=SAMPLE_RATE, mono=True)

    # normalize
    y = normalize_audio(y)

    # pad or trim
    target_len = int(TARGET_DURATION * SAMPLE_RATE)
    if y.shape[0] < target_len:
        y = np.pad(y, (0, target_len - y.shape[0]))
    elif y.shape[0] > target_len:
        y = y[:target_len]

    # save cleaned wav
    out_wav = WAV_OUT / f"{clip_id}.wav"
    sf.write(str(out_wav), y, SAMPLE_RATE)

    # compute log-mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        fmax=FMAX,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)

    # save feature array
    out_feat = FEAT_OUT / f"{clip_id}.npy"
    np.save(out_feat, logmel)


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    WAV_OUT.mkdir(parents=True, exist_ok=True)
    FEAT_OUT.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    rows = []

    # Read manifest first
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Found {len(rows)} manifest entries in {MANIFEST}")

    for row in tqdm(rows, desc="Preprocessing Events"):
        clip_id = row["clip_id"]
        audio_path = Path(row["audio_path"])

        if not audio_path.exists():
            stats["missing"] += 1
            continue

        try:
            preprocess_file(audio_path, clip_id)
            stats["ok"] += 1
        except Exception as e:
            print(f"[WARN] {clip_id}: {e}")
            stats["fail"] += 1

    print("\n[PREPROCESS SUMMARY]")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
