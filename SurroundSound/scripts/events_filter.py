"""
events_filter.py (FSD50K version)

Builds a metadata.jsonl for events using:
- FSD50K ground truth (dev/eval CSVs)
- Preprocessed feature files in data/events/processed/features

Output:
- data/events/metadata.jsonl

Each line in metadata.jsonl looks like:
{
  "ytid": "<clip_id>",
  "start": 0.0,
  "end": 10.0,
  "label_names": "Alarm;Bell",
  "status": "ok"
}

Where:
- <clip_id> matches the FSD50K "fname" (without .wav)
- start/end come from feature filename if present:
    <clip_id>_<start>_<end>.npy
  otherwise assume a single full-clip feature: <clip_id>.npy
"""

from pathlib import Path
import json
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "data"
EVENTS_ROOT  = DATA_ROOT / "events"

FSD_ROOT     = EVENTS_ROOT / "FSD50K"
FSD_GT_DIR   = FSD_ROOT / "FSD50K.ground_truth"

DEV_CSV      = FSD_GT_DIR / "dev.csv"
EVAL_CSV     = FSD_GT_DIR / "eval.csv"

PROCESSED_DIR      = EVENTS_ROOT / "processed"
PROCESSED_FEAT_DIR = PROCESSED_DIR / "features"

METADATA_OUT = EVENTS_ROOT / "metadata.jsonl"

print("[INFO] PROJECT_ROOT =", PROJECT_ROOT)
print("[INFO] DATA_ROOT    =", DATA_ROOT)
print("[INFO] EVENTS_ROOT  =", EVENTS_ROOT)
print("[INFO] FSD_GT_DIR   =", FSD_GT_DIR)

# Basic checks
if not PROCESSED_FEAT_DIR.exists():
    raise FileNotFoundError(f"Missing processed features dir: {PROCESSED_FEAT_DIR}")

EVENTS_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD FSD50K GROUND TRUTH
# ============================================================

dfs = []

if DEV_CSV.exists():
    print("[INFO] Loading dev CSV:", DEV_CSV)
    df_dev = pd.read_csv(DEV_CSV)
    df_dev["__split__"] = "dev"
    dfs.append(df_dev)
else:
    print("[WARN] dev.csv not found at:", DEV_CSV)

if EVAL_CSV.exists():
    print("[INFO] Loading eval CSV:", EVAL_CSV)
    df_eval = pd.read_csv(EVAL_CSV)
    df_eval["__split__"] = "eval"
    dfs.append(df_eval)
else:
    print("[WARN] eval.csv not found at:", EVAL_CSV)

if not dfs:
    raise FileNotFoundError(
        f"No FSD50K ground truth CSVs found under {FSD_GT_DIR}. "
        "Expected dev.csv and/or eval.csv."
    )

df_gt = pd.concat(dfs, ignore_index=True)
print("[INFO] Combined GT shape:", df_gt.shape)
print("[INFO] GT columns:", list(df_gt.columns))

# Expect at least: 'fname', 'labels'
if "fname" not in df_gt.columns or "labels" not in df_gt.columns:
    raise RuntimeError(
        "Expected columns 'fname' and 'labels' in FSD50K ground truth CSVs."
    )

# fname is e.g. "12345.wav" → use the stem "12345"
df_gt["clip_id"] = df_gt["fname"].astype(str).str.replace(".wav", "", regex=False)

def parse_labels(labels_str: str):
    """
    FSD50K 'labels' is a comma-separated string of label names,
    e.g. 'Alarm,Bell' → ['Alarm', 'Bell'].
    """
    if not isinstance(labels_str, str):
        return []
    parts = [s.strip() for s in labels_str.split(",")]
    return [p for p in parts if p]

df_gt["label_list"] = df_gt["labels"].apply(parse_labels)

# Build mapping: clip_id -> list of label names (unduplicated)
clip_to_labels = (
    df_gt.groupby("clip_id")["label_list"]
        .apply(lambda rows: sorted({lab for lst in rows for lab in lst}))
        .to_dict()
)

print(f"[INFO] Unique clip_ids in GT: {len(clip_to_labels)}")
print("[INFO] Example clip_id -> labels:")
for i, (cid, labs) in enumerate(clip_to_labels.items()):
    print(" ", cid, "->", labs[:10])
    if i >= 4:
        break


# ============================================================
# SCAN FEATURES AND BUILD METADATA
# ============================================================

feat_files = sorted(PROCESSED_FEAT_DIR.glob("*.npy"))
print(f"[INFO] Found {len(feat_files)} feature files.")

if not feat_files:
    raise RuntimeError("No .npy files found in events/processed/features.")

# Show a few example stems for sanity check
print("[INFO] Example feature stems:")
for fp in feat_files[:10]:
    print(" ", fp.stem)

num_matched = 0
num_unmatched = 0

# Default duration to use when start/end not in filename
DEFAULT_DURATION = 10.0

with METADATA_OUT.open("w", encoding="utf-8") as f_out:
    for fp in feat_files:
        stem = fp.stem
        parts = stem.split("_")

        # <clip_id>.npy  → one feature per full clip
        if len(parts) == 1:
            clip_id = parts[0]
            start = 0.0
            end = DEFAULT_DURATION

        # <clip_id>_<start>_<end>.npy
        elif len(parts) >= 3:
            clip_id = parts[0]
            try:
                start_str, end_str = parts[-2], parts[-1]
                start = float(start_str)
                end = float(end_str)
            except ValueError:
                # Fallback if parsing fails
                print(f"[WARN] Could not parse start/end from stem, using default: {stem}")
                start = 0.0
                end = DEFAULT_DURATION
        else:
            # Very weird pattern; skip
            print(f"[WARN] Unexpected feature stem (skipping): {stem}")
            num_unmatched += 1
            continue

        labels = clip_to_labels.get(clip_id, [])
        if not labels:
            # No ground truth for this clip_id; skip
            num_unmatched += 1
            continue

        label_names_str = ";".join(labels)

        record = {
            "ytid": clip_id,
            "start": float(start),
            "end": float(end),
            "label_names": label_names_str,
            "status": "ok",
            "source": "FSD50K",
        }

        f_out.write(json.dumps(record) + "\n")
        num_matched += 1

print("\n[INFO] Wrote metadata.jsonl ->", METADATA_OUT)
print(f"[INFO] Matched feature files:   {num_matched}")
print(f"[INFO] Unmatched feature files: {num_unmatched}")
print("[INFO] DONE.")
