"""
Preprocess AudioSet environment clips from TFRecord files.

Steps:
- Read VGGish embeddings from AudioSet TFRecord files using TensorFlow
- Match each record against the environment manifest (ytid + start_time)
- Extract 10x128 embedding matrix per clip
- Save as .npy files to data/environment/processed/features

No raw audio or ffmpeg required.
"""

import csv
import collections
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────

TFRECORD_DIR = Path("data/environment/tfrecords")
MANIFEST     = Path("data/manifests/environment_segments.csv")
OUT_DIR      = Path("data/environment/processed/features")

N_FRAMES  = 10
EMBED_DIM = 128

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Manifest loader ───────────────────────────────────────────────────────────

def load_manifest(path: Path):
    keys   = set()
    labels = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            ytid  = row["ytid"].strip()
            start = int(float(row["start"]))
            keys.add((ytid, start))
            labels[(ytid, start)] = row.get("label_names", "")
    return keys, labels


# ── TFRecord parsing ──────────────────────────────────────────────────────────

def parse_record(example):
    """
    Parse one AudioSet SequenceExample.
    Returns (video_id, start_time, embeddings_tensor) or None on failure.
    """
    context_features = {
        "video_id":           tf.io.FixedLenFeature([], tf.string),
        "start_time_seconds": tf.io.FixedLenFeature([], tf.float32),
        "end_time_seconds":   tf.io.FixedLenFeature([], tf.float32),
        "labels":             tf.io.VarLenFeature(tf.int64),
    }
    sequence_features = {
        "audio_embedding": tf.io.FixedLenSequenceFeature([], tf.string),
    }
    ctx, seq = tf.io.parse_single_sequence_example(
        example,
        context_features=context_features,
        sequence_features=sequence_features,
    )

    video_id   = ctx["video_id"].numpy().decode("utf-8")
    start_time = float(ctx["start_time_seconds"].numpy())

    # Each frame is 128 uint8 bytes → decode and dequantize to float32 [-2, 2]
    raw_frames = seq["audio_embedding"].numpy()  # shape: (n_frames,), each is bytes
    frames = []
    for raw in raw_frames:
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        arr = arr / 255.0 * 4.0 - 2.0  # dequantize per AudioSet spec
        frames.append(arr)

    return video_id, start_time, frames


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"[INFO] Loading manifest: {MANIFEST}")
    manifest_keys, label_map = load_manifest(MANIFEST)
    print(f"[INFO] Manifest clips:   {len(manifest_keys)}")

    tfrecord_files = sorted(str(p) for p in TFRECORD_DIR.rglob("*.tfrecord"))
    print(f"[INFO] TFRecord files:   {len(tfrecord_files)}")

    counts = collections.Counter()

    for tf_path in tqdm(tfrecord_files, desc="TFRecord files", unit="file"):
        dataset = tf.data.TFRecordDataset(tf_path, compression_type="")
        for raw in dataset:
            try:
                video_id, start_time, frames = parse_record(raw)
            except Exception as e:
                counts["parse_error"] += 1
                continue

            start_int = int(round(start_time))
            key = (video_id, start_int)

            if key not in manifest_keys:
                counts["skipped"] += 1
                continue

            # Pad or trim to exactly N_FRAMES
            if len(frames) < N_FRAMES:
                frames += [np.zeros(EMBED_DIM, dtype=np.float32)] * (N_FRAMES - len(frames))
            frames = frames[:N_FRAMES]

            matrix = np.stack(frames, axis=0)  # (10, 128)

            stem     = f"{video_id}_{start_int}"
            out_path = OUT_DIR / f"{stem}.npy"
            np.save(out_path, matrix)
            counts["saved"] += 1

    print(f"\n[DONE] Saved: {counts['saved']}  "
          f"Skipped: {counts['skipped']}  "
          f"Errors: {counts['parse_error']}")
    print(f"       Features → {OUT_DIR}")


if __name__ == "__main__":
    main()