"""
events_filter.py (FSD50K version)

Builds a metadata.jsonl for events using FSD50K ground truth CSVs.
Does NOT require preprocessed features to exist first.

Output:
  data/events/metadata.jsonl

Each line looks like:
{
  "ytid": "<clip_id>",
  "start": 0.0,
  "end": 10.0,
  "label_names": "Alarm;Bell",
  "status": "ok",
  "source": "FSD50K"
}

Usage:
  python scripts/events_filter.py
  python scripts/events_filter.py --fsd-root data/events/FSD50K --out data/events/metadata.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Build events metadata.jsonl from FSD50K ground truth.")
    ap.add_argument("--fsd-root", type=Path,
                    default=None,
                    help="FSD50K root dir (default: auto-detected from script location)")
    ap.add_argument("--out", type=Path,
                    default=None,
                    help="Output metadata.jsonl path (default: data/events/metadata.jsonl)")
    return ap.parse_args()


# ── Paths ─────────────────────────────────────────────────────────────────────

def resolve_paths(args):
    project_root = Path(__file__).resolve().parent.parent
    data_root    = project_root / "data"
    events_root  = data_root / "events"

    fsd_root  = args.fsd_root or events_root / "FSD50K"
    out_path  = args.out      or events_root / "metadata.jsonl"

    print(f"[INFO] PROJECT_ROOT = {project_root}")
    print(f"[INFO] FSD_ROOT     = {fsd_root}")
    print(f"[INFO] OUTPUT       = {out_path}")

    return fsd_root, out_path, events_root


# ── Load ground truth ─────────────────────────────────────────────────────────

def load_ground_truth(fsd_root: Path) -> dict:
    """
    Load dev.csv and eval.csv from FSD50K ground truth.
    Returns: clip_id -> sorted list of label names
    """
    gt_dir = fsd_root / "FSD50K.ground_truth"

    dfs = []
    for split in ("dev", "eval"):
        csv_path = gt_dir / f"{split}.csv"
        if csv_path.exists():
            print(f"[INFO] Loading {split}.csv: {csv_path}")
            df = pd.read_csv(csv_path)
            df["__split__"] = split
            dfs.append(df)
        else:
            print(f"[WARN] {split}.csv not found at {csv_path}")

    if not dfs:
        raise FileNotFoundError(
            f"No ground truth CSVs found under {gt_dir}. "
            "Expected dev.csv and/or eval.csv."
        )

    df_gt = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Combined GT shape: {df_gt.shape}")

    if "fname" not in df_gt.columns or "labels" not in df_gt.columns:
        raise RuntimeError("Expected columns 'fname' and 'labels' in ground truth CSVs.")

    # Normalize clip_id — fname may or may not have .wav extension
    df_gt["clip_id"] = (
        df_gt["fname"].astype(str)
        .str.strip()
        .str.replace(r"\.wav$", "", regex=True)
    )

    def parse_labels(s):
        if not isinstance(s, str):
            return []
        return [p.strip() for p in s.split(",") if p.strip()]

    df_gt["label_list"] = df_gt["labels"].apply(parse_labels)

    # clip_id -> sorted unique label names
    clip_to_labels = (
        df_gt.groupby("clip_id")["label_list"]
        .apply(lambda rows: sorted({lab for lst in rows for lab in lst}))
        .to_dict()
    )

    print(f"[INFO] Unique clip_ids in GT: {len(clip_to_labels)}")
    return clip_to_labels


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args        = parse_args()
    fsd_root, out_path, events_root = resolve_paths(args)

    events_root.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    clip_to_labels = load_ground_truth(fsd_root)

    counts = Counter()

    print(f"\n[INFO] Writing metadata -> {out_path}")
    with out_path.open("w", encoding="utf-8") as f_out:
        for clip_id, labels in tqdm(clip_to_labels.items(), desc="Writing", unit="clip"):
            if not labels:
                counts["no_labels"] += 1
                continue

            record = {
                "ytid":        clip_id,
                "start":       0.0,
                "end":         10.0,
                "label_names": ";".join(labels),
                "status":      "ok",
                "source":      "FSD50K",
            }
            f_out.write(json.dumps(record) + "\n")
            counts["written"] += 1

    print(f"\n[SUMMARY]")
    print(f"  Written:    {counts['written']}")
    print(f"  No labels:  {counts['no_labels']}")
    print(f"  Output:     {out_path.resolve()}")
    print("[INFO] DONE.")


if __name__ == "__main__":
    main()