"""
Filter AudioSet CSVs down to ENVIRONMENT segments using an allow list of labels.

Features:
- Robust to CSV/TSV variants and commas inside the labels field.
- Reads ontology.json to map label NAMES <-> IDs (case-insensitive).
- Keeps any row that has *at least one* allowed label ID.
- Optional: drop rows that co-occur with deny labels (e.g., Speech, Music) via --strict.
- Optional: cap number of rows per allowed label via --max_per_label.
- Writes a manifest CSV: ytid, start, end, label_ids, label_names
"""

import json
import csv
import argparse
import collections
from pathlib import Path

# ---------------------------------------------------------------------
# Label sets (case-insensitive)
# ---------------------------------------------------------------------

ALLOWED_LABELS = {
    # Scenes / locations
    "Outside, rural or natural", "Outside, urban or manmade", "Crowd",
    "Inside, small room", "Inside, large room or hall", "Inside, public space",
    "Field recording",

    # Weather / geophony
    "Wind", "Rain", "Rain on surface", "Thunder", "Thunderstorm",
    "Waves, surf", "Ocean", "Stream", "Waterfall", "Raindrop", "Rustling leaves",
    "Water",

    # Wildlife / biophony
    "Bird vocalization, bird call, bird song",
    "Bee, wasp, etc.", "Wild animals", "Livestock, farm animals, working animals",

    # Broad transport ambience
    "Traffic noise, roadway noise", "Rail transport", "Railroad car, train wagon", "Train",
    "Motor vehicle (road)", "Car passing by", "Subway, metro, underground",
    "Aircraft", "Fixed-wing aircraft, airplane",
    "Boat, Water vehicle", "Rowboat, canoe, kayak", "Sailboat, sailing ship",

    # Extra ambience / noise
    "Environmental noise", "Silence",
    "Wind noise (microphone)",    # teaches robustness
    "Idling", "Skidding",
    "Hubbub, speech noise, speech babble",
}

# things probably *don’t* want mixed into environment, when --strict is used
DENY_LABELS = {
    "Speech",
    "Conversation",
    "Narration, monologue",
    "Music",
    "Background music",
    "Singing",
    "Rapping",
    "Musical instrument",
}

# ---------------------------------------------------------------------
# Ontology + label parsing
# ---------------------------------------------------------------------

def load_ontology(ontology_path: str):
    """Return dicts: id->name and lowercased name->id from AudioSet ontology.json."""
    p = Path(ontology_path)
    if not p.exists():
        raise FileNotFoundError(f"Ontology not found: {p}")
    txt = p.read_text(encoding="utf-8", errors="ignore").lstrip("\ufeff").strip()
    if not txt:
        raise ValueError(f"Ontology is empty: {p}")
    if txt[:1] in "<":
        raise ValueError(f"Ontology looks like HTML, not JSON: {p}")
    data = json.loads(txt)
    id2name = {
        node["id"]: (node.get("name") or node.get("display_name") or node["id"])
        for node in data
    }
    name2id = {name.lower(): _id for _id, name in id2name.items()}
    return id2name, name2id

def parse_label_ids(raw: str):
    """AudioSet CSV lists positive_labels as a comma-separated string of label IDs (quoted)."""
    s = str(raw).strip().strip('"')
    return [t.strip() for t in s.split(",") if t.strip()]

# ---------------------------------------------------------------------
# Robust CSV / TSV reader
# ---------------------------------------------------------------------

def iter_segments_file(path: Path):
    """
    Robust reader for AudioSet segment CSV/TSV variants.
    Yields tuples: (ytid:str, start:float, end:float, labels:str)

    - Skips blank lines and headers/comments.
    - Supports comma CSV and tab TSV.
    - If CSV: split at most 3 commas so the remainder is labels.
    - If TSV: join columns 4..N as labels with commas.
    """
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            low = line.lower()
            if low.startswith("# ytid") or low.startswith("ytid,") or low.startswith("ytid\t"):
                continue  # header
            if line.startswith("#"):
                continue  # comment

            if "\t" in line and line.count(",") < 3:
                parts = line.split("\t")
                if len(parts) < 4:
                    continue
                ytid, start, end = parts[0], parts[1], parts[2]
                labels = ",".join(p.strip() for p in parts[3:] if p.strip())
            else:
                parts = line.split(",", 3)
                if len(parts) < 4:
                    continue
                ytid, start, end, labels = parts[0], parts[1], parts[2], parts[3]

            labels = labels.strip().strip('"')
            try:
                yield ytid.strip(), float(start), float(end), labels
            except ValueError:
                continue  # skip malformed numeric row

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Filter AudioSet segments by an allowlist of label names (environment-focused)."
    )
    ap.add_argument(
        "--csv_dir", required=True,
        help="Dir with balanced_train_segments.csv, eval_segments.csv, (optional) unbalanced_train_segments.csv"
    )
    ap.add_argument(
        "--ontology", required=True,
        help="Path to ontology.json (AudioSet)"
    )
    ap.add_argument(
        "--out", default="data/manifests/environment_segments.csv",
        help="Output manifest CSV path"
    )
    ap.add_argument(
        "--include_unbalanced", action="store_true",
        help="Also include unbalanced_train_segments.csv"
    )
    ap.add_argument(
        "--strict", action="store_true",
        help="Drop rows that co-occur with deny labels (Speech/Music/etc.)"
    )
    ap.add_argument(
        "--max_per_label", type=int, default=None,
        help="If set, limit number of rows per allowed label ID to at most this many (across all splits)."
    )
    args = ap.parse_args()

    csv_dir = Path(args.csv_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Map our allowed/deny names -> IDs via the ontology
    id2name, name2id = load_ontology(args.ontology)

    allow_ids = {name2id[n.lower()] for n in ALLOWED_LABELS if n.lower() in name2id}
    deny_ids = {name2id[n.lower()] for n in DENY_LABELS if n.lower() in name2id}

    missing_allow = [n for n in ALLOWED_LABELS if n.lower() not in name2id]
    missing_deny = [n for n in DENY_LABELS if n.lower() not in name2id]

    if missing_allow:
        print(f"[WARN] {len(missing_allow)} allowed names not found in ontology:")
        for n in sorted(missing_allow):
            print(f"  - {n}")
    if missing_deny:
        print(f"[WARN] {len(missing_deny)} deny names not found in ontology:")
        for n in sorted(missing_deny):
            print(f"  - {n}")

    # Decide which splits to scan
    splits = ["balanced_train_segments.csv", "eval_segments.csv"]
    if args.include_unbalanced:
        splits.append("unbalanced_train_segments.csv")

    total_in = 0
    total_out = 0
    per_label_counts = collections.Counter()  # counts per allowed label ID

    with out_path.open("w", newline="", encoding="utf-8") as f_out:
        w = csv.writer(f_out)
        # Header: keep both IDs and names for clarity
        w.writerow(["ytid", "start", "end", "label_ids", "label_names"])

        for split in splits:
            p = csv_dir / split
            if not p.exists():
                print(f"[INFO] Missing split, skipping: {p}")
                continue

            print(f"[INFO] Reading {p} ...")
            kept = 0

            for ytid, start, end, labs in iter_segments_file(p):
                total_in += 1
                ids = set(parse_label_ids(labs))
                if not ids:
                    continue

                # At least one allowed label ID must be present
                allowed_in_row = ids & allow_ids
                if not allowed_in_row:
                    continue

                # Optional strict mode: drop if any deny label present
                if args.strict and (ids & deny_ids):
                    continue

                # Optional cap per allowed label ID
                if args.max_per_label is not None:
                    # If *all* allowed labels on this row already hit cap, skip
                    if all(per_label_counts[a_id] >= args.max_per_label for a_id in allowed_in_row):
                        continue

                # Accept the row
                names = [id2name.get(_id, _id) for _id in sorted(ids)]
                w.writerow([ytid, start, end, ";".join(sorted(ids)), ";".join(names)])
                kept += 1
                total_out += 1

                # Update counts for all allowed labels in this row
                if args.max_per_label is not None:
                    for a_id in allowed_in_row:
                        if per_label_counts[a_id] < args.max_per_label:
                            per_label_counts[a_id] += 1

            print(f"[INFO] Kept {kept} rows from {split}")

    print("\n[SUMMARY]")
    print(f"  Allowed label IDs: {len(allow_ids)}")
    print(f"  Deny label IDs:    {len(deny_ids)}")
    print(f"  Total rows scanned: {total_in}")
    print(f"  Total rows written: {total_out}")
    if args.max_per_label is not None:
        nonzero = {k: v for k, v in per_label_counts.items() if v > 0}
        print(f"  Per-label counts (nonzero, capped at {args.max_per_label}): {len(nonzero)} labels")

    print(f"  Output manifest: {out_path.resolve()}")

if __name__ == "__main__":
    main()
