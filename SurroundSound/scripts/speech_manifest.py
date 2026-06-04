#!/usr/bin/env python3
"""
Build a unified speech manifest from Common Voice + CHiME-6.

Labels:
  single_speaker  — Common Voice clips (one speaker per clip)
  multi_speaker   — CHiME-6 segments with >= 2 concurrent speakers
  no_speech       — CHiME-6 silence gaps between speaker turns

Output:
  data/speech/manifests/speech_manifest.csv
  Columns: clip_id, audio_path, label, source, duration

Usage:
  python scripts/speech_manifest.py --root data/speech
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple


# ── Config ────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Build unified speech manifest.")
    ap.add_argument("--root", type=Path, default=Path("data/speech"),
                    help="Speech data root (default: data/speech)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output manifest path (default: <root>/manifests/speech_manifest.csv)")
    ap.add_argument("--cv-max-clips", type=int, default=50000,
                    help="Max Common Voice clips to include (default: 50000)")
    ap.add_argument("--chime-segment-sec", type=float, default=10.0,
                    help="Segment length in seconds for CHiME-6 (default: 10.0)")
    ap.add_argument("--min-silence-sec", type=float, default=2.0,
                    help="Min silence gap to count as no_speech (default: 2.0)")
    return ap.parse_args()


# ── Common Voice ──────────────────────────────────────────────────────────────

def load_common_voice(cv_root: Path, max_clips: int) -> List[dict]:
    """
    Load validated Common Voice clips from train.tsv.
    Each clip is labeled single_speaker.
    """
    # CV extracts to a versioned subdirectory — find it
    tsv_candidates = list(cv_root.rglob("train.tsv"))
    if not tsv_candidates:
        print(f"[WARN] No train.tsv found under {cv_root}. Skipping Common Voice.")
        return []

    tsv_path = tsv_candidates[0]
    clips_dir = tsv_path.parent / "clips"
    print(f"[INFO] Loading Common Voice from: {tsv_path}")

    records = []
    with tsv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i >= max_clips:
                break
            audio_path = clips_dir / row["path"]
            if not audio_path.exists():
                continue
            records.append({
                "clip_id":    f"cv_{Path(row['path']).stem}",
                "audio_path": str(audio_path),
                "label":      "single_speaker",
                "source":     "common_voice",
                "duration":   None,  # filled during preprocess
            })

    print(f"[INFO] Common Voice clips loaded: {len(records)}")
    return records


# ── CHiME-6 ───────────────────────────────────────────────────────────────────

def _parse_chime6_time(t) -> Optional[float]:
    """
    Parse CHiME-6 timestamp.
    Can be:
      - float/int: seconds directly
      - str "HH:MM:SS.ss": convert to seconds
      - dict {channel: timestamp_str}: pick first available channel
    """
    if isinstance(t, (int, float)):
        return float(t)
    if isinstance(t, str):
        try:
            parts = t.strip().split(":")
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            return float(t)
        except Exception:
            return None
    if isinstance(t, dict):
        for k in ("U06", "U01", "U02", "U03", "U04", "U05"):
            if k in t:
                return _parse_chime6_time(t[k])
        if t:
            return _parse_chime6_time(next(iter(t.values())))
    return None


def load_chime6_transcriptions(trans_dir: Path) -> dict:
    """
    Load CHiME-6 JSON transcription files.
    Returns: session_id -> list of {speaker, start, end}
    """
    sessions   = {}
    json_files = sorted(trans_dir.rglob("*.json"))
    print(f"[INFO] Found {len(json_files)} CHiME-6 transcription files.")
    if json_files:
        print(f"[INFO] Example: {json_files[0]}")

    for jf in json_files:
        # Use session_id from first entry if available, else filename stem
        try:
            with jf.open(encoding="utf-8") as f:
                data = json.load(f)
            turns = []
            session_id = jf.stem
            for entry in data:
                # Override session_id from data if present
                if "session_id" in entry:
                    session_id = entry["session_id"]
                spk = entry.get("speaker", "UNK")
                s   = _parse_chime6_time(entry.get("start_time"))
                e   = _parse_chime6_time(entry.get("end_time"))
                if s is not None and e is not None and e > s:
                    turns.append({"speaker": spk, "start": s, "end": e})

            if turns:
                if session_id not in sessions:
                    sessions[session_id] = []
                sessions[session_id].extend(turns)
        except Exception as ex:
            print(f"[WARN] Failed to parse {jf.name}: {ex}")

    # Sort all turns by start time
    for sid in sessions:
        sessions[sid] = sorted(sessions[sid], key=lambda x: x["start"])

    return sessions


def find_chime6_audio(chime_root: Path, session_id: str) -> Optional[Path]:
    """Find the best single-channel audio file for a CHiME-6 session."""
    # Prefer U06 binaural, fall back to any available channel
    for split in ("train", "dev", "eval"):
        split_dir = chime_root / split
        if not split_dir.exists():
            continue
        # Try common naming patterns
        for channel in ("U06", "U01", "U02", "U03", "U04", "U05"):
            candidates = list(split_dir.rglob(f"{session_id}*{channel}*.wav"))
            if candidates:
                return candidates[0]
        # Fallback: any wav for this session
        candidates = list(split_dir.rglob(f"{session_id}*.wav"))
        if candidates:
            return candidates[0]
    return None


def segment_chime6_session(
    session_id: str,
    turns: list,
    audio_path: Path,
    segment_sec: float,
    min_silence_sec: float,
    chime_root: Path,
) -> List[dict]:
    """
    Segment a CHiME-6 session into fixed-length windows.
    Label each window based on how many speakers are active:
      - 0 speakers active for >= min_silence_sec → no_speech
      - 1 speaker active → single_speaker (not used — keeping binary multi/no for CHiME)
      - >= 2 speakers active → multi_speaker
    """
    if not turns:
        return []

    session_end = max(t["end"] for t in turns)
    records = []
    t = 0.0
    seg_idx = 0

    while t + segment_sec <= session_end:
        seg_start = t
        seg_end   = t + segment_sec

        # Count distinct active speakers in this window
        active_speakers = set()
        for turn in turns:
            # Turn overlaps with segment
            if turn["end"] > seg_start and turn["start"] < seg_end:
                active_speakers.add(turn["speaker"])

        n_speakers = len(active_speakers)

        if n_speakers == 0:
            label = "no_speech"
        elif n_speakers == 1:
            label = "single_speaker"
        else:
            label = "multi_speaker"

        records.append({
            "clip_id":    f"chime6_{session_id}_{seg_idx:04d}",
            "audio_path": str(audio_path),
            "label":      label,
            "source":     "chime6",
            "duration":   segment_sec,
            "seg_start":  seg_start,
            "seg_end":    seg_end,
        })

        t += segment_sec
        seg_idx += 1

    return records


def load_chime6(chime_root: Path, segment_sec: float, min_silence_sec: float) -> List[dict]:
    trans_dir = chime_root / "transcriptions"
    # Handle double-nested extraction (tar may extract to transcriptions/transcriptions/)
    if (trans_dir / "transcriptions").exists():
        trans_dir = trans_dir / "transcriptions"
    if not trans_dir.exists():
        print(f"[WARN] CHiME-6 transcriptions not found at {trans_dir}. Skipping.")
        return []
    print(f"[INFO] CHiME-6 transcriptions dir: {trans_dir}")

    sessions = load_chime6_transcriptions(trans_dir)
    all_records = []

    for session_id, turns in sessions.items():
        audio_path = find_chime6_audio(chime_root, session_id)
        if audio_path is None:
            print(f"[WARN] No audio found for CHiME-6 session {session_id}, skipping.")
            continue

        records = segment_chime6_session(
            session_id, turns, audio_path,
            segment_sec, min_silence_sec, chime_root,
        )
        all_records.extend(records)

    print(f"[INFO] CHiME-6 segments generated: {len(all_records)}")
    return all_records


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    root = args.root
    out_path = args.out or root / "manifests" / "speech_manifest.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cv_root    = root / "CommonVoice"
    chime_root = root / "CHiME6"

    all_records = []

    # Common Voice
    if cv_root.exists():
        cv_records = load_common_voice(cv_root, max_clips=args.cv_max_clips)
        all_records.extend(cv_records)
    else:
        print(f"[WARN] Common Voice not found at {cv_root}")

    # CHiME-6
    if chime_root.exists():
        chime_records = load_chime6(
            chime_root,
            segment_sec=args.chime_segment_sec,
            min_silence_sec=args.min_silence_sec,
        )
        all_records.extend(chime_records)
    else:
        print(f"[WARN] CHiME-6 not found at {chime_root}")

    if not all_records:
        raise RuntimeError("No records found. Check that datasets are downloaded and extracted.")

    # Write manifest
    fieldnames = ["clip_id", "audio_path", "label", "source", "duration", "seg_start", "seg_end"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)

    # Summary
    counts = Counter(r["label"] for r in all_records)
    sources = Counter(r["source"] for r in all_records)

    print(f"\n[SUMMARY]")
    print(f"  Output: {out_path.resolve()}")
    print(f"  Total clips: {len(all_records)}")
    print(f"  By label:")
    for label, n in sorted(counts.items()):
        print(f"    {n:8d}  {label}")
    print(f"  By source:")
    for src, n in sorted(sources.items()):
        print(f"    {n:8d}  {src}")


if __name__ == "__main__":
    main()