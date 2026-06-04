#!/usr/bin/env python3
"""
01_speech_setup.py

Prepare speech data for the Speech detector model.

Steps:
- Load speech_manifest.csv (from speech_manifest.py)
- Validate labels (no_speech, single_speaker, conversation)
- Match rows to preprocessed .npy log-mel features
- Download faster-whisper large-v3 model weights
- Save:
    - data/speech/label_to_id.json
    - data/speech/id_to_label.json
    - data/speech/data_index.csv
    - data/speech/data_index.parquet
"""

import json
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "data"
SPEECH_ROOT  = DATA_ROOT / "speech"

MANIFEST_PATH      = SPEECH_ROOT / "manifests" / "speech_manifest.csv"
FEAT_ROOT          = SPEECH_ROOT / "processed" / "features"
LABEL_TO_ID_PATH   = SPEECH_ROOT / "label_to_id.json"
ID_TO_LABEL_PATH   = SPEECH_ROOT / "id_to_label.json"
DATA_INDEX_CSV     = SPEECH_ROOT / "data_index.csv"
DATA_INDEX_PARQUET = SPEECH_ROOT / "data_index.parquet"

# Whisper model cache directory
WHISPER_CACHE = PROJECT_ROOT / "models" / "whisper"

VALID_LABELS = {"no_speech", "single_speaker", "conversation", "multi_speaker"}
# Remap multi_speaker -> conversation for consistency
LABEL_REMAP  = {"multi_speaker": "conversation"}

print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
print(f"[INFO] MANIFEST     = {MANIFEST_PATH}")
print(f"[INFO] FEAT_ROOT    = {FEAT_ROOT}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}\n"
            "Run speech_manifest.py first."
        )
    df = pd.read_csv(path)
    print(f"[INFO] Manifest rows: {len(df)}")
    return df


def build_feat_lookup(feat_root: Path) -> dict:
    if not feat_root.exists():
        raise FileNotFoundError(
            f"Feature directory not found: {feat_root}\n"
            "Run speech_preprocess.py first."
        )
    files = list(feat_root.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files in {feat_root}")
    print(f"[INFO] Feature files found: {len(files)}")
    return {p.stem: str(p) for p in files}


def download_whisper(cache_dir: Path) -> None:
    """
    Download faster-whisper large-v3 weights.
    Uses huggingface_hub to cache the model locally.
    """
    print("\n[INFO] Checking faster-whisper large-v3 model...")
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("[WARN] faster-whisper not installed.")
        print("       Install with: pip install faster-whisper")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "large-v3"

    if model_path.exists() and any(model_path.iterdir()):
        print(f"[SKIP] Whisper large-v3 already downloaded at {model_path}")
        return

    print("[INFO] Downloading faster-whisper large-v3 (this may take a while)...")
    print("       Model size: ~3GB")
    try:
        # Loading the model triggers the download and caches it
        model = WhisperModel(
            "large-v3",
            device="cpu",           # just for download, don't need GPU
            compute_type="int8",    # smallest footprint during download
            download_root=str(cache_dir),
        )
        del model
        print(f"[INFO] Whisper large-v3 downloaded -> {cache_dir}")
    except Exception as e:
        print(f"[WARN] Could not download Whisper model: {e}")
        print("       It will be downloaded automatically on first use.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load manifest
    df = load_manifest(MANIFEST_PATH)

    if "label" not in df.columns:
        raise ValueError("Manifest missing 'label' column.")

    # Remap multi_speaker -> conversation
    df["label"] = df["label"].replace(LABEL_REMAP)

    # Filter to valid labels
    before = len(df)
    df = df[df["label"].isin(VALID_LABELS)].reset_index(drop=True)
    print(f"[INFO] Dropped {before - len(df)} rows with invalid labels. Remaining: {len(df)}")

    print("\n[INFO] Class distribution:")
    print(df["label"].value_counts().sort_index().to_string())

    # Build label maps
    class_names = sorted(df["label"].unique())
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

    # Match features
    feat_by_stem = build_feat_lookup(FEAT_ROOT)
    df["feature_path"] = df["clip_id"].map(feat_by_stem)
    df["y"] = df["label"].map(label_to_id)

    before = len(df)
    df = df[df["feature_path"].notna()].reset_index(drop=True)
    print(f"\n[INFO] Rows with matching features: {len(df)}/{before}")

    if len(df) == 0:
        raise RuntimeError("No rows matched feature files.")

    # Preview
    cols = ["clip_id", "label", "y", "source", "feature_path"]
    cols = [c for c in cols if c in df.columns]
    print("\n[INFO] Sample rows:")
    print(df[cols].head(5).to_string(index=False))

    print("\n[INFO] Final class counts:")
    print(df["label"].value_counts().sort_index().to_string())

    # Save
    out_cols = ["feature_path", "label", "y"]
    if "source" in df.columns:
        out_cols.append("source")
    out_df = df[out_cols].copy()

    out_df.to_csv(DATA_INDEX_CSV, index=False)
    print(f"\n[INFO] Wrote CSV     -> {DATA_INDEX_CSV}")

    try:
        out_df.to_parquet(DATA_INDEX_PARQUET, index=False)
        print(f"[INFO] Wrote Parquet -> {DATA_INDEX_PARQUET}")
    except Exception as e:
        print(f"[WARN] Could not write Parquet: {e}")

    # Download Whisper
    download_whisper(WHISPER_CACHE)

    print("\n[INFO] DONE.")


if __name__ == "__main__":
    main()