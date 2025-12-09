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


def load_vocabulary(vocab_path: Path):
    """
    Load FSD50K vocabulary.csv.

    FSD50K format (no header, 3 columns):
        idx, label_name, audioset_mid
    Return:
        mid_to_name:  audioset_mid -> label_name (string)
    """
    mid_to_name = {}

    with vocab_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            # Skip empty / malformed rows
            if not row or len(row) < 3:
                continue

            idx, label_name, audioset_mid = row[0], row[1], row[2]

            label_name = str(label_name).strip()
            audioset_mid = str(audioset_mid).strip()

            if not audioset_mid:
                continue

            mid_to_name[audioset_mid] = label_name

    return mid_to_name



def iter_split(split_name: str, gt_path: Path, audio_dir: Path):
    """
    Iterate over rows in dev.csv / eval.csv.

    Expected columns:
      fname  -> e.g. "64890"
      labels -> e.g. "/m/01xqw,/m/02zsn"
    """
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    with gt_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("fname")
            labels_raw = row.get("labels", "")

            if not fname:
                continue

            clip_id = fname.strip()
            audio_path = audio_dir / f"{clip_id}.wav"
            if not audio_path.exists():
                # Some FSD50K distros place audio elsewhere; warn and skip
                print(f"[WARN] Missing audio for {split_name} clip {clip_id}: {audio_path}")
                continue

            label_ids = []
            if labels_raw:
                # Split on comma, FSD50K labels are comma-separated MIDs
                label_ids = [t.strip() for t in labels_raw.split(",") if t.strip()]

            yield {
                "clip_id": clip_id,
                "split": split_name,
                "audio_path": str(audio_path),
                "label_ids": label_ids,
            }


def main():
    ap = argparse.ArgumentParser(
        description="Build FSD50K Events manifest (paths + labels)."
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("data/events/FSD50K"),
        help="FSD50K root directory (default: data/events/FSD50K)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/events/manifests/fsd50k_events_manifest.csv"),
        help="Output manifest CSV path",
    )
    args = ap.parse_args()

    root = args.root
    gt_dir = root / "FSD50K.ground_truth"
    dev_audio_dir = root / "FSD50K.dev_audio"
    eval_audio_dir = root / "FSD50K.eval_audio"

    vocab_path = gt_dir / "vocabulary.csv"
    dev_gt = gt_dir / "dev.csv"
    eval_gt = gt_dir / "eval.csv"

    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing vocabulary.csv at {vocab_path}")
    if not dev_gt.exists():
        raise FileNotFoundError(f"Missing dev.csv at {dev_gt}")
    if not eval_gt.exists():
        raise FileNotFoundError(f"Missing eval.csv at {eval_gt}")

    mid_to_name = load_vocabulary(vocab_path)
    print(f"[INFO] Loaded {len(mid_to_name)} label entries from vocabulary.csv")

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    per_split = Counter()
    label_counts = Counter()

    with out_path.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["clip_id", "split", "audio_path", "label_ids", "label_names"])

        # DEV split
        for rec in iter_split("dev", dev_gt, dev_audio_dir):
            label_ids = rec["label_ids"]
            label_names = [mid_to_name.get(mid, mid) for mid in label_ids]
            writer.writerow([
                rec["clip_id"],
                rec["split"],
                rec["audio_path"],
                ";".join(label_ids),
                ";".join(label_names),
            ])
            total += 1
            per_split["dev"] += 1
            for mid in label_ids:
                label_counts[mid] += 1

        # EVAL split
        for rec in iter_split("eval", eval_gt, eval_audio_dir):
            label_ids = rec["label_ids"]
            label_names = [mid_to_name.get(mid, mid) for mid in label_ids]
            writer.writerow([
                rec["clip_id"],
                rec["split"],
                rec["audio_path"],
                ";".join(label_ids),
                ";".join(label_names),
            ])
            total += 1
            per_split["eval"] += 1
            for mid in label_ids:
                label_counts[mid] += 1

    print("\n[SUMMARY]")
    print(f"  Manifest: {out_path.resolve()}")
    print(f"  Total clips written: {total}")
    for split, n in per_split.items():
        print(f"    {split}: {n}")
    print(f"  Unique label_ids: {len(label_counts)}")


if __name__ == "__main__":
    main()
