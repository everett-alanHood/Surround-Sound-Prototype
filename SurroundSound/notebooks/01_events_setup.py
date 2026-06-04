"""
01_events_setup.py

Prepare FSD50K-based EVENT data for multi-label classification.

Steps:
- Load metadata.jsonl (from events_filter.py).
- Load taxonomy_events.json (canonical event classes + parents).
- Map raw FSD50K labels -> canonical multi-label set (including parent classes).
- Match rows to precomputed .npy log-mel features.
- Save:
    - data/events/label_to_id.json
    - data/events/id_to_label.json
    - data/events/data_index.csv
    - data/events/data_index.parquet
"""

import ast
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "data"
EVENTS_ROOT  = DATA_ROOT / "events"

METADATA_PATH    = EVENTS_ROOT / "metadata.jsonl"
TAXONOMY_PATH    = EVENTS_ROOT / "taxonomy_events.json"
FEATURES_DIR     = EVENTS_ROOT / "processed" / "features"
LABEL_TO_ID_PATH = EVENTS_ROOT / "label_to_id.json"
ID_TO_LABEL_PATH = EVENTS_ROOT / "id_to_label.json"
DATA_INDEX_CSV   = EVENTS_ROOT / "data_index.csv"
DATA_INDEX_PARQ  = EVENTS_ROOT / "data_index.parquet"

OTHER_CANONICAL  = "Other Event"


# ── Helpers ───────────────────────────────────────────────────────────────────

def ensure_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        # Handle semicolon-separated label strings from events_filter.py
        if ";" in x:
            return [p.strip() for p in x.split(";") if p.strip()]
        try:
            v = ast.literal_eval(x)
            if isinstance(v, list):
                return v
        except Exception:
            return [x]
    return []


# ── Taxonomy loading ──────────────────────────────────────────────────────────

def load_taxonomy(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing taxonomy_events.json at {path}\n"
            "This file defines the canonical event class hierarchy.\n"
            "Make sure it exists under data/events/taxonomy_events.json."
        )
    with path.open("r", encoding="utf-8") as f:
        taxonomy = json.load(f)
    print(f"[INFO] Loaded taxonomy with {len(taxonomy)} canonical entries.")
    return taxonomy


def build_raw_to_canonical(taxonomy: dict):
    raw_to_canonical = {}
    canonical_labels_set = set()
    parent_labels_set = set()

    for canon, info in taxonomy.items():
        canonical_labels_set.add(canon)
        for p in (info.get("parents") or []):
            parent_labels_set.add(p)
        for raw_label in (info.get("specific") or []):
            raw_to_canonical.setdefault(raw_label, set()).add(canon)

    canonical_labels_set.update(parent_labels_set)
    print(f"[INFO] Canonical label count (incl. parents): {len(canonical_labels_set)}")
    return raw_to_canonical, canonical_labels_set


# ── Label mapping ─────────────────────────────────────────────────────────────

def map_raw_to_canonical(raw_labels: list, raw_to_canonical: dict, taxonomy: dict) -> list:
    mapped = set()
    for raw in raw_labels:
        parts = [p.strip() for p in str(raw).split(";") if p.strip()]
        for part in parts:
            if part in raw_to_canonical:
                direct = raw_to_canonical[part]
                mapped.update(direct)
                for canon in direct:
                    mapped.update(taxonomy.get(canon, {}).get("parents") or [])
    if not mapped:
        mapped.add(OTHER_CANONICAL)
    return sorted(mapped)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"[INFO] PROJECT_ROOT  = {PROJECT_ROOT}")
    print(f"[INFO] EVENTS_ROOT   = {EVENTS_ROOT}")
    print(f"[INFO] METADATA      = {METADATA_PATH}")
    print(f"[INFO] TAXONOMY      = {TAXONOMY_PATH}")
    print(f"[INFO] FEATURES_DIR  = {FEATURES_DIR}")

    # Load taxonomy
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    raw_to_canonical, _ = build_raw_to_canonical(taxonomy)

    # Load metadata
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata.jsonl: {METADATA_PATH}\nRun events_filter.py first.")

    print(f"\n[INFO] Loading metadata: {METADATA_PATH}")
    df = pd.read_json(METADATA_PATH, lines=True)
    print(f"[INFO] Metadata shape: {df.shape}")

    if "status" in df.columns:
        before = len(df)
        df = df[df["status"] == "ok"].copy()
        print(f"[INFO] Filtered status=='ok': {before} -> {len(df)}")

    if "label_names" not in df.columns:
        raise KeyError("metadata.jsonl must contain a 'label_names' column.")

    df["label_names"] = df["label_names"].apply(ensure_list)

    before = len(df)
    df = df[df["label_names"].apply(len) > 0].copy()
    print(f"[INFO] Dropped rows with no labels: {before} -> {len(df)}")

    # Map to canonical labels
    print("\n[INFO] Mapping raw labels to canonical classes...")
    df["canonical_labels"] = df["label_names"].apply(
        lambda x: map_raw_to_canonical(x, raw_to_canonical, taxonomy)
    )

    all_canon = [lab for row in df["canonical_labels"] for lab in row]
    canon_counts = Counter(all_canon)
    print(f"[INFO] Unique canonical labels observed: {len(canon_counts)}")
    print("\n[INFO] Top 20 canonical labels:")
    for lab, cnt in canon_counts.most_common(20):
        print(f"  {cnt:6d}  {lab}")

    if OTHER_CANONICAL in canon_counts:
        print(f"[INFO] '{OTHER_CANONICAL}' used for {canon_counts[OTHER_CANONICAL]} rows.")

    # Build label maps — only labels that appear
    used_labels  = sorted(canon_counts.keys())
    label_to_id  = {lab: i for i, lab in enumerate(used_labels)}
    id_to_label  = {i: lab for lab, i in label_to_id.items()}
    num_classes  = len(label_to_id)
    print(f"\n[INFO] Total canonical classes: {num_classes}")

    with LABEL_TO_ID_PATH.open("w", encoding="utf-8") as f:
        json.dump(label_to_id, f, indent=2)
    with ID_TO_LABEL_PATH.open("w", encoding="utf-8") as f:
        json.dump(id_to_label, f, indent=2)
    print(f"[INFO] Wrote {LABEL_TO_ID_PATH}")
    print(f"[INFO] Wrote {ID_TO_LABEL_PATH}")

    # Convert canonical labels to IDs
    df["label_ids"] = df["canonical_labels"].apply(
        lambda labs: [label_to_id[l] for l in labs if l in label_to_id]
    )
    before = len(df)
    df = df[df["label_ids"].apply(len) > 0].copy()
    print(f"[INFO] Dropped rows with empty label_ids: {before} -> {len(df)}")

    # Match feature files
    if not FEATURES_DIR.exists():
        raise FileNotFoundError(f"Features dir not found: {FEATURES_DIR}\nRun events_preprocess.py first.")

    feat_map = {p.stem: str(p) for p in FEATURES_DIR.glob("*.npy")}
    print(f"\n[INFO] Feature files found: {len(feat_map)}")

    if "ytid" not in df.columns:
        raise KeyError("metadata.jsonl must contain 'ytid' column.")

    df["ytid"] = df["ytid"].astype(str)
    df["feature_path"] = df["ytid"].map(feat_map)

    before = len(df)
    df = df[df["feature_path"].notna()].copy()
    print(f"[INFO] Rows with matching features: {len(df)}/{before}")

    if len(df) == 0:
        raise RuntimeError(
            "No rows matched feature files. "
            "Check that events_preprocess.py has been run and ytid values match."
        )

    # Final index
    df_index = df[["ytid", "canonical_labels", "label_ids", "feature_path"]].copy()
    df_index = df_index.rename(columns={"canonical_labels": "label_names"})

    print("\n[INFO] Sample rows:")
    print(df_index.head(5).to_string(index=False))

    print(f"\n[INFO] Final class counts (top 20):")
    final_counts = Counter(l for row in df_index["label_names"] for l in row)
    for lab, cnt in final_counts.most_common(20):
        print(f"  {cnt:6d}  {lab}")

    df_index.to_csv(DATA_INDEX_CSV, index=False)
    print(f"\n[INFO] Wrote CSV     -> {DATA_INDEX_CSV}")

    try:
        df_index.to_parquet(DATA_INDEX_PARQ, index=False)
        print(f"[INFO] Wrote Parquet -> {DATA_INDEX_PARQ}")
    except Exception as e:
        print(f"[WARN] Could not write Parquet ({e}). CSV saved at {DATA_INDEX_CSV}.")

    print("\n[INFO] DONE.")


if __name__ == "__main__":
    main()