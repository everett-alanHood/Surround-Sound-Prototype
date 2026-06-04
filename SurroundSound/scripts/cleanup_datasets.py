#!/usr/bin/env python3
"""
cleanup_datasets.py

Delete all raw and processed dataset files (.npy, .wav, .mp3, .zip, .tar.gz,
.tsv, and any file over 100MB), while keeping model checkpoints, label maps,
data indexes, manifests, and taxonomy files.

Usage:
  python scripts/cleanup_datasets.py           # dry run
  python scripts/cleanup_datasets.py --delete  # actually delete
"""

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DATA_ROOT      = PROJECT_ROOT / "data"
OUTPUT_ROOT    = PROJECT_ROOT / "output"

SIZE_THRESHOLD = 100 * 1024 * 1024  # 100 MB

EXTENSIONS = {
    ".npy", ".wav", ".mp3", ".tsv",
    ".zip", ".gz", ".z01", ".z02", ".z03",
    ".z04", ".z05", ".z06", ".z07", ".z08", ".z09",
    ".part", ".tar",
}

# Directories to wipe entirely
WIPE_DIRS = [
    DATA_ROOT / "environment" / "processed" / "features",
    DATA_ROOT / "environment" / "tfrecords",
    DATA_ROOT / "environment" / "raw",
    DATA_ROOT / "events" / "processed" / "features",
    DATA_ROOT / "events" / "processed" / "wav",
    DATA_ROOT / "events" / "FSD50K" / "FSD50K.dev_audio",
    DATA_ROOT / "events" / "FSD50K" / "FSD50K.eval_audio",
    DATA_ROOT / "speech" / "processed" / "features",
    DATA_ROOT / "speech" / "processed" / "wav",
    DATA_ROOT / "speech" / "CHiME6" / "train",
    DATA_ROOT / "speech" / "CHiME6" / "dev",
    DATA_ROOT / "speech" / "CHiME6" / "eval",
    DATA_ROOT / "speech" / "CommonVoice",  # all CV raw data including TSVs
]

# Always keep these filenames regardless of extension or size
KEEP_NAMES = {
    "label_to_id.json", "id_to_label.json",
    "data_index.csv", "data_index.parquet",
    "metadata.jsonl", "speech_manifest.csv",
    "taxonomy_events.json", "vocabulary.csv",
    "hparam_search_results.csv",
    "events_training_config.json",
    "speech_training_config.json",
    "overall_metrics.csv", "per_class_metrics.csv",
    "confusion_matrix_counts.csv",
    "confusion_matrix_normalized.csv",
    "whisper_benchmark.json",
}

# Always keep paths containing these substrings
KEEP_PATHS = [
    str(OUTPUT_ROOT),       # model checkpoints
    "FSD50K.ground_truth",  # FSD50K label CSVs
    "FSD50K.metadata",      # FSD50K metadata
    "transcriptions",       # CHiME-6 transcriptions
    "manifests",            # speech manifests
    "whisper",              # Whisper model weights
    "id_to_label",
    "label_to_id",
    "data_index",
    "taxonomy",
    "ontology",
]


def fmt_size(path: Path) -> str:
    try:
        if path.is_file():
            size = path.stat().st_size
        elif path.is_dir():
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        else:
            return "0 B"
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    except Exception:
        return "?"


def should_keep(path: Path) -> bool:
    if path.name in KEEP_NAMES:
        return True
    path_str = str(path)
    return any(k in path_str for k in KEEP_PATHS)


def collect_targets():
    targets = []

    # Wipe entire directories first
    for d in WIPE_DIRS:
        if d.exists() and not should_keep(d):
            targets.append((d, "directory", None))

    dirs_to_wipe = {t[0] for t in targets if t[1] == "directory"}

    # Scan data root for files matching extension or size threshold
    for path in DATA_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if should_keep(path):
            continue
        # Skip if already covered by a directory wipe
        if any(path.is_relative_to(d) for d in dirs_to_wipe):
            continue

        try:
            size = path.stat().st_size
        except Exception:
            continue

        reason = None
        if path.suffix.lower() in EXTENSIONS:
            reason = f"extension {path.suffix}"
        elif size >= SIZE_THRESHOLD:
            reason = f"large file ({size / 1024 / 1024:.0f} MB)"

        if reason:
            targets.append((path, "file", reason))

    return targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete", action="store_true", help="Actually delete (default: dry run)")
    args = ap.parse_args()

    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Mode: {'DELETE' if args.delete else 'DRY RUN'}")
    print(f"[INFO] Size threshold: files >= 100 MB")
    print()

    targets = collect_targets()

    if not targets:
        print("[INFO] Nothing to clean up.")
        return

    total_size = 0
    print(f"{'SIZE':>10}  TYPE    PATH")
    print("-" * 90)
    for path, kind, reason in targets:
        size_str = fmt_size(path)
        label = "[DIR] " if kind == "directory" else "[FILE]"
        note  = f"  ← {reason}" if reason else ""
        print(f"{size_str:>10}  {label} {path}{note}")
        try:
            if path.is_file():
                total_size += path.stat().st_size
            elif path.is_dir():
                total_size += sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        except Exception:
            pass

    print("-" * 90)
    total_gb = total_size / (1024 ** 3)
    print(f"\n{'Total to free:':>20}  {total_gb:.2f} GB across {len(targets)} items")
    print(f"\n[KEEPING] output/  — all trained model checkpoints")
    print(f"[KEEPING] label maps, data indexes, manifests, taxonomy, CHiME-6 transcriptions")

    if not args.delete:
        print("\n[DRY RUN] No files deleted. Run with --delete to actually remove them.")
        return

    print("\nAre you sure you want to delete the above? [yes/no]: ", end="")
    if input().strip().lower() != "yes":
        print("[INFO] Aborted.")
        return

    deleted = errors = 0
    for path, kind, _ in targets:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            print(f"[DELETED] {path}")
            deleted += 1
        except Exception as e:
            print(f"[ERROR] {path}: {e}")
            errors += 1

    print(f"\n[DONE] Deleted {deleted} items. Errors: {errors}")
    print(f"       Freed ~{total_gb:.2f} GB")


if __name__ == "__main__":
    main()