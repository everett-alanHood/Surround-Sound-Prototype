"""
Preprocess environment audio clips from both:
- data/environment/raw
- data/environment_unbalanced/raw

Steps:
- Resample to 16 kHz mono
- Normalize loudness
- Trim/pad to fixed length (10 s)
- Generate log-mel spectrograms
- Save cleaned WAV + .npy features into data/environment/processed
"""

import json
import collections
from pathlib import Path

import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

# Directories
RAW_DIRS = [
    Path("data/environment/raw"),
    Path("data/environment_unbalanced/raw"),
]

OUT_DIR = Path("data/environment/processed")
WAV_OUT = OUT_DIR / "wav"
FEAT_OUT = OUT_DIR / "features"

SAMPLE_RATE = 16000
TARGET_DURATION = 10.0  # seconds
TARGET_SAMPLES = int(TARGET_DURATION * SAMPLE_RATE)
N_MELS = 128
HOP_LENGTH = 512

WAV_OUT.mkdir(parents=True, exist_ok=True)
FEAT_OUT.mkdir(parents=True, exist_ok=True)

def normalize_audio(y):
    peak = np.abs(y).max()
    return y / peak if peak > 0 else y

def preprocess_file(path: Path):
    stem = path.stem

    out_wav = WAV_OUT / f"{stem}.wav"
    out_feat = FEAT_OUT / f"{stem}.npy"

    # Skip if already processed
    if out_wav.exists() and out_feat.exists():
        return

    # load (resample to 16k mono)
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)

    # normalize
    y = normalize_audio(y)

    # pad or trim to fixed length
    if len(y) < TARGET_SAMPLES:
        y = np.pad(y, (0, TARGET_SAMPLES - len(y)))
    elif len(y) > TARGET_SAMPLES:
        y = y[:TARGET_SAMPLES]

    # save cleaned wav
    sf.write(out_wav, y, SAMPLE_RATE)

    # compute log-mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        fmax=8000,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)

    # save feature array
    np.save(out_feat, logmel)

def main():
    # Gather wavs from both raw dirs
    all_wavs = []
    for rd in RAW_DIRS:
        if rd.exists():
            files = sorted(rd.glob("*.wav"))
            print(f"Found {len(files)} raw wavs in {rd}")
            all_wavs.extend(files)
        else:
            print(f"[WARN] Raw dir does not exist, skipping: {rd}")

    # Unduplicate by stem, in case any overlap
    unique = {}
    for p in all_wavs:
        unique.setdefault(p.stem, p)
    wavs = list(unique.values())

    print(f"Total unique wavs to consider: {len(wavs)}")

    for wav in tqdm(wavs, desc="Processing"):
        try:
            preprocess_file(wav)
        except Exception as e:
            print(f"[WARN] {wav.name}: {e}")

    # Optional metadata summary from both metadata files
    meta_files = [
        Path("data/environment/metadata.jsonl"),
        Path("data/environment_unbalanced/metadata.jsonl"),
    ]
    c = collections.Counter()
    for mf in meta_files:
        if not mf.exists():
            continue
        with mf.open() as f:
            for line in f:
                rec = json.loads(line)
                c[rec.get("status", "unknown")] += 1

    print("\n[METADATA SUMMARY]")
    print(c)

if __name__ == "__main__":
    main()
