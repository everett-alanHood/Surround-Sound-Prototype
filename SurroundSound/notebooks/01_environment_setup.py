#!/usr/bin/env python3
"""
Build environment data index + label maps from manifest CSV + processed features.

Replaces the old metadata.jsonl-based approach.

Reads:
  - data/manifests/environment_segments.csv  (ytid, start, end, label_ids, label_names)
  - data/environment/processed/features/*.npy

Writes:
  - data/environment/label_to_id.json
  - data/environment/id_to_label.json
  - data/environment/data_index.csv
  - data/environment/data_index.parquet
"""

import json
import sys
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

THIS_FILE   = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"
DATA_ROOT    = PROJECT_ROOT / "data"

MANIFEST_PATH       = DATA_ROOT / "manifests" / "environment_segments.csv"
FEAT_ROOT           = DATA_ROOT / "environment" / "processed" / "features"
ENV_ROOT            = DATA_ROOT / "environment"
LABEL_TO_ID_PATH    = ENV_ROOT / "label_to_id.json"
ID_TO_LABEL_PATH    = ENV_ROOT / "id_to_label.json"
DATA_INDEX_CSV      = ENV_ROOT / "data_index.csv"
DATA_INDEX_PARQUET  = ENV_ROOT / "data_index.parquet"

print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
print(f"[INFO] MANIFEST     = {MANIFEST_PATH}")
print(f"[INFO] FEAT_ROOT    = {FEAT_ROOT}")

sys.path.insert(0, str(SCRIPTS_DIR))
try:
    from environment_label import map_to_balanced
except Exception as e:
    raise SystemExit(
        f"[ERROR] Could not import map_to_balanced.\n"
        f"Scripts dir: {SCRIPTS_DIR}"
    ) from e


# ── Load manifest ─────────────────────────────────────────────────────────────

def load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"[ERROR] Manifest not found: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Manifest rows: {len(df)}")
    print(f"[INFO] Columns: {list(df.columns)}")
    return df


# ── Build feature lookup ──────────────────────────────────────────────────────

def build_feat_lookup(feat_root: Path) -> dict:
    if not feat_root.exists():
        raise SystemExit(f"[ERROR] Feature directory not found: {feat_root}")
    files = list(feat_root.glob("*.npy"))
    if not files:
        raise SystemExit(f"[ERROR] No .npy files found in: {feat_root}")
    print(f"[INFO] Feature files found: {len(files)}")
    return {p.stem: str(p) for p in files}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load manifest
    df = load_manifest(MANIFEST_PATH)

    # Map label_names -> primary_label
    if "label_names" not in df.columns:
        raise SystemExit("[ERROR] Manifest missing 'label_names' column.")

    df["primary_label"] = df["label_names"].apply(
        lambda x: map_to_balanced(str(x)) if pd.notna(x) else None
    )

    before = len(df)
    df = df[df["primary_label"].notna()].reset_index(drop=True)
    print(f"[INFO] Dropped {before - len(df)} unmapped rows. Remaining: {len(df)}")

    print("\n[INFO] Class distribution:")
    print(df["primary_label"].value_counts().sort_index().to_string())

    # Build label maps
    class_names = sorted(df["primary_label"].unique())
    label_to_id = {name: i for i, name in enumerate(class_names)}
    id_to_label = {i: name for name, i in label_to_id.items()}

    print(f"\n[INFO] {len(class_names)} classes:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")

    with LABEL_TO_ID_PATH.open("w") as f:
        json.dump(label_to_id, f, indent=2)
    with ID_TO_LABEL_PATH.open("w") as f:
        json.dump(id_to_label, f, indent=2)
    print(f"[INFO] Wrote {LABEL_TO_ID_PATH}")
    print(f"[INFO] Wrote {ID_TO_LABEL_PATH}")

    # Build feature lookup
    feat_by_stem = build_feat_lookup(FEAT_ROOT)

    # Match manifest rows to feature files
    def get_stem(row):
        return f"{row['ytid'].strip()}_{int(float(row['start']))}"

    df["stem"] = df.apply(get_stem, axis=1)
    df["feature_path"] = df["stem"].map(feat_by_stem)
    df["y"] = df["primary_label"].map(label_to_id)

    before = len(df)
    df = df[df["feature_path"].notna()].reset_index(drop=True)
    print(f"\n[INFO] Rows with matching features: {len(df)}/{before}")

    if len(df) == 0:
        raise SystemExit("[ERROR] No rows matched feature files.")

    # Preview
    print("\n[INFO] Sample rows:")
    print(df[["ytid", "start", "primary_label", "y", "feature_path"]].head(5).to_string(index=False))

    # Final class distribution
    print("\n[INFO] Final class counts:")
    print(df["primary_label"].value_counts().sort_index().to_string())

    # Save outputs
    out_df = df[["feature_path", "primary_label", "y"]].copy()
    out_df.to_csv(DATA_INDEX_CSV, index=False)
    print(f"\n[INFO] Wrote CSV     -> {DATA_INDEX_CSV}")

    try:
        out_df.to_parquet(DATA_INDEX_PARQUET, index=False)
        print(f"[INFO] Wrote Parquet -> {DATA_INDEX_PARQUET}")
    except Exception as e:
        print(f"[WARN] Could not write Parquet: {e}")

    print("\n[INFO] DONE.")


if __name__ == "__main__":
    main()