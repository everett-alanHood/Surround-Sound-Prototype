"""
Debug script: print first 20 records from TFRecords and compare against manifest.
"""
import csv
from pathlib import Path
import tensorflow as tf

TFRECORD_DIR = Path("data/environment/tfrecords")
MANIFEST     = Path("data/manifests/environment_segments.csv")

# Load a sample of manifest keys
manifest = {}
with MANIFEST.open() as f:
    for row in csv.DictReader(f):
        ytid  = row["ytid"].strip()
        start = float(row["start"])
        manifest[ytid] = start

context_features = {
    "video_id":           tf.io.FixedLenFeature([], tf.string),
    "start_time_seconds": tf.io.FixedLenFeature([], tf.float32),
}
sequence_features = {
    "audio_embedding": tf.io.FixedLenSequenceFeature([], tf.string),
}

tfrecord_files = sorted(str(p) for p in TFRECORD_DIR.rglob("*.tfrecord"))
print(f"Checking first 20 records across first few files...\n")

count = 0
matches = 0
for tf_path in tfrecord_files:
    for raw in tf.data.TFRecordDataset(tf_path):
        ctx, _ = tf.io.parse_single_sequence_example(
            raw, context_features=context_features, sequence_features=sequence_features
        )
        vid   = ctx["video_id"].numpy().decode()
        start = float(ctx["start_time_seconds"].numpy())
        start_int = int(round(start))

        in_manifest = vid in manifest
        match = in_manifest and int(round(manifest[vid])) == start_int

        print(f"  vid={vid}  start_raw={start:.4f}  start_int={start_int}  "
              f"manifest_start={manifest.get(vid, 'NOT IN MANIFEST')}  match={match}")

        if match:
            matches += 1
        count += 1
        if count >= 20:
            break
    if count >= 20:
        break

print(f"\nMatches in first 20: {matches}/20")