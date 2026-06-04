"""
Build an Events manifest from FSD50K.

Reads:
  data/events/FSD50K/FSD50K.ground_truth/dev.csv
  data/events/FSD50K/FSD50K.ground_truth/eval.csv
  data/events/FSD50K/FSD50K.ground_truth/vocabulary.csv

Writes:
  data/events/manifests/fsd50k_events_manifest.csv

Columns:
  clip_id, split, audio_path, label_ids, label_names
"""

import csv
from pathlib import Path
import argparse
from collections import Counter

from tqdm import tqdm


def load_vocabulary(vocab_path: Path):
    """
    Load FSD50K vocabulary.csv.

    FSD50K format (no header, 3 columns):
        idx, label_name, audioset_mid
    Returns:
        mid_to_name: audioset_mid -> label_name
    """
    mid_to_name = {}
    with vocab_path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or len(row) < 3:
                continue
            _, label_name, audioset_mid = row[0], row[1].strip(), row[2].strip()
            if audioset_mid:
                mid_to_name[audioset_mid] = label_name
    return mid_to_name


def load_split(split_name: str, gt_path: Path, audio_dir: Path):
    """
    Load all rows from dev.csv / eval.csv.
    Returns (records, n_missing) where records have audio_path verified.
    """
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    rows = []
    with gt_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    records  = []
    n_missing = 0

    for row in tqdm(rows, desc=f"  {split_name}", unit="clip", leave=False):
        fname = row.get("fname", "").strip()
        if not fname:
            continue

        audio_path = audio_dir / f"{fname}.wav"
        if not audio_path.exists():
            n_missing += 1
            continue

        labels_raw = row.get("labels", "")
        label_ids  = [t.strip() for t in labels_raw.split(",") if t.strip()]

        records.append({
            "clip_id":    fname,
            "split":      split_name,
            "audio_path": str(audio_path),
            "label_ids":  label_ids,
        })

    return records, n_missing


def main():
    ap = argparse.ArgumentParser(
        description="Build FSD50K Events manifest (paths + labels)."
    )
    ap.add_argument("--root", type=Path, default=Path("data/events/FSD50K"),
                    help="FSD50K root directory (default: data/events/FSD50K)")
    ap.add_argument("--out",  type=Path, default=Path("data/events/manifests/fsd50k_events_manifest.csv"),
                    help="Output manifest CSV path")
    args = ap.parse_args()

    root          = args.root
    gt_dir        = root / "FSD50K.ground_truth"
    dev_audio_dir = root / "FSD50K.dev_audio"
    eval_audio_dir = root / "FSD50K.eval_audio"

    for p, label in [
        (gt_dir / "vocabulary.csv", "vocabulary.csv"),
        (gt_dir / "dev.csv",        "dev.csv"),
        (gt_dir / "eval.csv",       "eval.csv"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {label} at {p}")

    mid_to_name = load_vocabulary(gt_dir / "vocabulary.csv")
    print(f"[INFO] Loaded {len(mid_to_name)} labels from vocabulary.csv")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    total      = 0
    per_split  = Counter()
    n_missing  = Counter()
    label_counts = Counter()

    print("[INFO] Loading splits...")
    dev_records,  dev_missing  = load_split("dev",  gt_dir / "dev.csv",  dev_audio_dir)
    eval_records, eval_missing = load_split("eval", gt_dir / "eval.csv", eval_audio_dir)

    n_missing["dev"]  = dev_missing
    n_missing["eval"] = eval_missing

    with args.out.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["clip_id", "split", "audio_path", "label_ids", "label_names"])

        for rec in tqdm(dev_records + eval_records, desc="Writing manifest", unit="clip"):
            label_ids   = rec["label_ids"]
            label_names = [mid_to_name.get(mid, mid) for mid in label_ids]
            writer.writerow([
                rec["clip_id"],
                rec["split"],
                rec["audio_path"],
                ";".join(label_ids),
                ";".join(label_names),
            ])
            total += 1
            per_split[rec["split"]] += 1
            for mid in label_ids:
                label_counts[mid] += 1

    print("\n[SUMMARY]")
    print(f"  Manifest: {args.out.resolve()}")
    print(f"  Total clips written: {total}")
    for split, n in per_split.items():
        print(f"    {split}: {n}  (skipped {n_missing[split]} missing audio)")
    print(f"  Unique label_ids: {len(label_counts)}")
    print(f"  Total missing audio: {sum(n_missing.values())}")


if __name__ == "__main__":
    main()