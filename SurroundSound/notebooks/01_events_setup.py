"""
01_events_setup.py

Prepare FSD50K-based EVENT data for multi-label classification.

Steps:
- Load metadata.jsonl (from FSD50K metadata + your previous merge step).
- Load taxonomy_events.json (canonical event classes + parents).
- Map raw FSD50K labels -> canonical multi-label set (including parent classes).
- Match rows to precomputed .npy log-mel features.
- Save:
    - data/events/label_to_id.json
    - data/events/id_to_label.json
    - data/events/data_index.csv
    - data/events/data_index.parquet
"""

import json
import ast
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

# -----------------------
# Paths / config
# -----------------------

PROJECT_ROOT = Path(r"C:\SurroundSound")
DATA_ROOT    = PROJECT_ROOT / "data"
EVENTS_ROOT  = DATA_ROOT / "events"

METADATA_MAIN   = EVENTS_ROOT / "metadata.jsonl"
TAXONOMY_PATH   = EVENTS_ROOT / "taxonomy_events.json"
FEATURES_DIR    = EVENTS_ROOT / "processed" / "features"

LABEL_TO_ID_PATH = EVENTS_ROOT / "label_to_id.json"
ID_TO_LABEL_PATH = EVENTS_ROOT / "id_to_label.json"
DATA_INDEX_CSV   = EVENTS_ROOT / "data_index.csv"
DATA_INDEX_PARQ  = EVENTS_ROOT / "data_index.parquet"

OTHER_CANONICAL  = "Other Event"  # fallback if nothing maps

print("[INFO] PROJECT_ROOT =", PROJECT_ROOT)
print("[INFO] DATA_ROOT    =", DATA_ROOT)
print("[INFO] EVENTS_ROOT  =", EVENTS_ROOT)
print("[INFO] METADATA_MAIN =", METADATA_MAIN)
print("[INFO] TAXONOMY_PATH =", TAXONOMY_PATH)
print("[INFO] FEATURES_DIR  =", FEATURES_DIR)

# -----------------------
# Helpers
# -----------------------

def ensure_list(x):
    """Ensure value is a list (useful if label_names came in as string)."""
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            if isinstance(v, list):
                return v
        except Exception:
            return [x]
    return []

# -----------------------
# Load taxonomy
# -----------------------

if not TAXONOMY_PATH.exists():
    raise FileNotFoundError(f"Missing taxonomy_events.json: {TAXONOMY_PATH}")

with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
    taxonomy = json.load(f)

print(f"[INFO] Loaded taxonomy with {len(taxonomy)} canonical entries (excluding parents-only labels).")

# Build raw_label -> set(canonical_labels) mapping
raw_to_canonical = {}
canonical_labels_set = set()
parent_labels_set = set()

for canon, info in taxonomy.items():
    canonical_labels_set.add(canon)
    specific = info.get("specific", []) or []
    parents  = info.get("parents", []) or []
    for p in parents:
        parent_labels_set.add(p)

    for raw_label in specific:
        raw_to_canonical.setdefault(raw_label, set()).add(canon)
        # parent labels will be handled at assignment time

# Include parent names as canonical output classes as well
canonical_labels_set.update(parent_labels_set)
print(f"[INFO] Canonical label base count (incl. parents): {len(canonical_labels_set)}")

# -----------------------
# Load metadata
# -----------------------

if not METADATA_MAIN.exists():
    raise FileNotFoundError(f"Missing events metadata jsonl: {METADATA_MAIN}")

print("[INFO] Loading metadata from:", METADATA_MAIN)
df_meta = pd.read_json(METADATA_MAIN, lines=True)

print("[INFO] Combined metadata shape:", df_meta.shape)

# Filter by status if present
if "status" in df_meta.columns:
    before = len(df_meta)
    df_meta = df_meta[df_meta["status"] == "ok"].copy()
    print(f"[INFO] Filtered status=='ok': {before} -> {len(df_meta)}")

# Ensure label_names column exists
if "label_names" not in df_meta.columns:
    raise KeyError("metadata.jsonl must contain a 'label_names' column with raw FSD labels.")

# Drop rows with no labels
before = len(df_meta)
df_meta["label_names"] = df_meta["label_names"].apply(ensure_list)
df_meta = df_meta[df_meta["label_names"].apply(len) > 0].copy()
print(f"[INFO] Dropped rows with no labels: {before} -> {len(df_meta)}")

print("\n[INFO] label_names examples:")
print(df_meta["label_names"].head())

# -----------------------
# Map raw -> canonical (with parents)
# -----------------------

def map_raw_to_canonical(raw_labels):
    """
    Map raw FSD50K labels to canonical classes, handling semicolon-separated
    strings like 'Animal;Bird;Water' by splitting on ';'.
    """
    mapped = set()

    for raw in raw_labels:
        # raw might be a single label ("Bird") or a semicolon-joined string
        parts = [p.strip() for p in str(raw).split(";") if p.strip()]

        for part in parts:
            if part in raw_to_canonical:
                direct = raw_to_canonical[part]
                mapped.update(direct)
                # add parents for each canonical label
                for canon in direct:
                    info = taxonomy.get(canon, {})
                    parents = info.get("parents", []) or []
                    mapped.update(parents)
            else:
                # unknown label; ignore
                continue

    if not mapped:
        mapped.add(OTHER_CANONICAL)

    return sorted(mapped)


print("\n[INFO] Mapping raw labels to canonical classes...")
df_meta["canonical_labels"] = df_meta["label_names"].apply(map_raw_to_canonical)

# Stats on canonical label usage
all_canon = [lab for row in df_meta["canonical_labels"] for lab in row]
canon_counts = Counter(all_canon)
print("\n[INFO] Canonical label counts (top 30):")
for lab, cnt in canon_counts.most_common(30):
    print(f"  {lab}: {cnt}")

print(f"\n[INFO] Unique canonical labels observed: {len(canon_counts)}")

# Ensure OTHER_CANONICAL included if used
if OTHER_CANONICAL in canon_counts:
    canonical_labels_set.add(OTHER_CANONICAL)

# -----------------------
# Build label_to_id / id_to_label
# -----------------------

# Only keep canonical labels that actually appear
used_labels = sorted(canon_counts.keys())

label_to_id = {lab: i for i, lab in enumerate(used_labels)}
id_to_label = {i: lab for lab, i in label_to_id.items()}

print("\n[INFO] Final canonical label mapping:")
for lab, idx in list(label_to_id.items())[:30]:
    print(f"  {idx}: {lab}")
if len(label_to_id) > 30:
    print(f"  ... ({len(label_to_id)} total labels)")

with open(LABEL_TO_ID_PATH, "w", encoding="utf-8") as f:
    json.dump(label_to_id, f, indent=2)
with open(ID_TO_LABEL_PATH, "w", encoding="utf-8") as f:
    json.dump(id_to_label, f, indent=2)

print(f"\n[INFO] Wrote label_to_id.json -> {LABEL_TO_ID_PATH}")
print(f"[INFO] Wrote id_to_label.json -> {ID_TO_LABEL_PATH}")

# Convert canonical_labels -> label_ids
df_meta["label_ids"] = df_meta["canonical_labels"].apply(
    lambda labs: [label_to_id[lab] for lab in labs if lab in label_to_id]
)

# Drop rows that somehow ended up empty (should be rare)
before = len(df_meta)
df_meta = df_meta[df_meta["label_ids"].apply(len) > 0].copy()
print(f"\n[INFO] Dropped rows with empty label_ids: {before} -> {len(df_meta)}")

# -----------------------
# Match feature files
# -----------------------

if not FEATURES_DIR.exists():
    raise FileNotFoundError(f"Features dir not found: {FEATURES_DIR}")

feature_files = list(FEATURES_DIR.glob("*.npy"))
print(f"\n[INFO] Found {len(feature_files)} feature files.")

# Map stem -> path
feat_map = {p.stem: p for p in feature_files}

print("\n[INFO] Example feature stems:")
print(pd.Series(list(feat_map.keys())[:10], name="ytid"))

# Ensure ytid column present
if "ytid" not in df_meta.columns:
    raise KeyError("metadata.jsonl must contain 'ytid' column matching feature filenames.")

# Coerce ytid to string for matching
df_meta["ytid"] = df_meta["ytid"].astype(str)

def get_feature_path(ytid):
    p = feat_map.get(str(ytid))
    return str(p) if p is not None else None

df_meta["feature_path"] = df_meta["ytid"].apply(get_feature_path)

matched = df_meta["feature_path"].notna().sum()
total   = len(df_meta)
print(f"\n[INFO] Rows with matching features: {matched}/{total}")

df_meta = df_meta[df_meta["feature_path"].notna()].copy()

# -----------------------
# Final index + save
# -----------------------

df_index = df_meta[["ytid", "canonical_labels", "label_ids", "feature_path"]].copy()
df_index = df_index.rename(columns={"canonical_labels": "label_names"})

print("\n[INFO] Final preview:")
print(df_index.head(10))

# Save CSV + Parquet
df_index.to_csv(DATA_INDEX_CSV, index=False)
try:
    df_index.to_parquet(DATA_INDEX_PARQ, index=False)
    print(f"\n[INFO] Wrote CSV    -> {DATA_INDEX_CSV}")
    print(f"[INFO] Wrote Parquet -> {DATA_INDEX_PARQ}")
except Exception as e:
    print(f"\n[WARN] Failed to write Parquet ({e}). CSV still saved at {DATA_INDEX_CSV}.")

print("\n[INFO] DONE.")
