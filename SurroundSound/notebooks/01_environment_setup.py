#!/usr/bin/env python3
"""
Build environment data index + label maps.

This script:
- Loads metadata from:
      data/environment/metadata.jsonl
      data/environment_unbalanced/metadata.jsonl
- Maps raw AudioSet label_names -> unified environment classes
  via environment_label.map_to_balanced()
- Builds label_to_id.json and id_to_label.json
- Attaches feature_path pointing to processed .npy files
- Writes:
      data/environment/data_index.csv
      data/environment/data_index.parquet (if available)
"""

import json
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Detect project root (because this script is in notebooks/)
# ---------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent 
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_ROOT = PROJECT_ROOT / "data"

ENV_ROOT = DATA_ROOT / "environment"
ENV_UNBAL_ROOT = DATA_ROOT / "environment_unbalanced"

METADATA_FILES = [
    ENV_ROOT / "metadata.jsonl",
    ENV_UNBAL_ROOT / "metadata.jsonl",
]

FEAT_ROOT = ENV_ROOT / "processed" / "features"

LABEL_TO_ID_PATH = ENV_ROOT / "label_to_id.json"
ID_TO_LABEL_PATH = ENV_ROOT / "id_to_label.json"
DATA_INDEX_PARQUET = ENV_ROOT / "data_index.parquet"
DATA_INDEX_CSV = ENV_ROOT / "data_index.csv"

print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
print(f"[INFO] SCRIPTS_DIR  = {SCRIPTS_DIR}")
print(f"[INFO] DATA_ROOT    = {DATA_ROOT}")

sys.path.append(str(SCRIPTS_DIR))

try:
    from environment_label import map_to_balanced
except Exception as e:
    raise SystemExit(
        f"[ERROR] Could not import map_to_balanced from environment_label.py.\n"
        f"Scripts directory: {SCRIPTS_DIR}"
    ) from e


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def load_metadata_files(paths):
    """Load .jsonl metadata into a single DataFrame."""
    records = []
    for mf in paths:
        if not mf.exists():
            print(f"[WARN] Metadata file missing, skipping: {mf}")
            continue

        print(f"[INFO] Loading metadata from: {mf}")
        with mf.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    rec["__source_metadata__"] = mf.name
                    records.append(rec)
                except Exception as e:
                    print(f"[WARN] Failed to parse line in {mf}: {e}")

    if not records:
        raise SystemExit("[ERROR] No metadata loaded; check your .jsonl files.")

    df = pd.DataFrame(records)
    print(f"[INFO] Combined metadata shape: {df.shape}")
    return df


def label_names_to_str(x):
    """Normalize label_names to a single string."""
    if isinstance(x, list):
        if not x:
            return ""
        return str(x[0])
    if isinstance(x, str):
        return x
    if pd.isna(x):
        return ""
    return str(x)


def build_label_maps(df):
    """Build label_to_id.json and id_to_label.json."""
    class_names = sorted(df["primary_label"].unique())
    label_to_id = {name: i for i, name in enumerate(class_names)}
    id_to_label = {i: name for name, i in label_to_id.items()}

    print(f"[INFO] Found {len(class_names)} classes:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")

    # Save them
    with LABEL_TO_ID_PATH.open("w", encoding="utf-8") as f:
        json.dump(label_to_id, f, indent=2)

    with ID_TO_LABEL_PATH.open("w", encoding="utf-8") as f:
        json.dump(id_to_label, f, indent=2)

    print(f"[INFO] Wrote label_to_id.json  -> {LABEL_TO_ID_PATH}")
    print(f"[INFO] Wrote id_to_label.json  -> {ID_TO_LABEL_PATH}")

    return label_to_id, id_to_label


def build_feat_lookup(feat_root: Path):
    """Map stem -> feature_path for all processed .npy features."""
    if not feat_root.exists():
        raise SystemExit(f"[ERROR] No feature directory: {feat_root}")

    feat_files = list(feat_root.glob("*.npy"))
    if not feat_files:
        raise SystemExit(f"[ERROR] No .npy features found in: {feat_root}")

    mapping = {p.stem: str(p) for p in feat_files}
    print(f"[INFO] Found {len(mapping)} feature files.")
    return mapping


def infer_stem(row):
    """
    Infers the stem for a row to match metadata -> processed .npy.
    """
    # 1) metadata already has a stem
    for key in ("stem", "feature_stem"):
        if key in row and isinstance(row[key], str) and row[key]:
            return Path(row[key]).stem

    # 2) audio or wav path in metadata
    for key in ("wav_path", "audio_path", "path", "local_path"):
        if key in row and isinstance(row[key], str) and row[key]:
            return Path(row[key]).stem

    # 3) fallback: YouTube ID (if each ytid has one clip)
    if "ytid" in row and isinstance(row["ytid"], str):
        return row["ytid"]

    return None


# ---------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------

def main():
    # Load metadata
    df = load_metadata_files(METADATA_FILES)

    # Normalize label_names
    if "label_names" not in df.columns:
        raise SystemExit("[ERROR] Expected 'label_names' column in metadata.")
    df["raw_label_str"] = df["label_names"].apply(label_names_to_str)

    print("\n[INFO] raw_label_str examples:")
    print(df["raw_label_str"].head(10).to_string(index=False))

    # Map raw label strings -> balanced primary_label
    df["primary_label"] = df["raw_label_str"].apply(map_to_balanced)

    print("\n[INFO] primary_label distribution (with None):")
    print(df["primary_label"].value_counts(dropna=False).head(20))

    before = len(df)
    df = df[df["primary_label"].notnull()].reset_index(drop=True)
    after = len(df)
    print(f"\n[INFO] Dropped {before - after} unmapped rows.")
    print(f"[INFO] Remaining rows: {after}")

    print("\n[INFO] Class counts:")
    print(df["primary_label"].value_counts().sort_index())

    # Build label_to_id / id_to_label
    label_to_id, id_to_label = build_label_maps(df)

    # Build feature lookup
    feat_by_stem = build_feat_lookup(FEAT_ROOT)

    # Infer stem for each metadata row
    df["stem"] = df.apply(infer_stem, axis=1)

    print("\n[INFO] Example stems:")
    print(df["stem"].head(10))

    before = len(df)
    df["feature_path"] = df["stem"].map(feat_by_stem)
    df = df[df["feature_path"].notnull()].reset_index(drop=True)
    after = len(df)

    print(f"\n[INFO] Rows with matching features: {after}/{before}")

    if after == 0:
        raise SystemExit(
            "[ERROR] No rows matched feature files. "
            "You must adjust infer_stem() to match your actual filenames."
        )

    # Numeric y
    df["y"] = df["primary_label"].map(label_to_id)

    # Preview
    print("\n[INFO] Final preview:")
    cols = ["__source_metadata__", "ytid", "primary_label", "y", "feature_path"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].head(10).to_string(index=False))

    # Save CSV + Parquet
    df.to_csv(DATA_INDEX_CSV, index=False)
    print(f"\n[INFO] Wrote CSV -> {DATA_INDEX_CSV}")

    try:
        df.to_parquet(DATA_INDEX_PARQUET, index=False)
        print(f"[INFO] Wrote Parquet -> {DATA_INDEX_PARQUET}")
    except Exception as e:
        print(f"[WARN] Could not write Parquet: {e}")

    print("\n[INFO] DONE.")


if __name__ == "__main__":
    main()
