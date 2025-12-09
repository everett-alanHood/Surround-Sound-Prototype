#!/usr/bin/env python3
"""
Download and unpack the FSD50K dataset for the Events head.

- Downloads multi-part dev/eval audio zips + metadata zips from Zenodo.
- Verifies MD5 checksums.
- Merges split zips (z01, z02, ..., .zip) into a single unsplit .zip.
- Extracts archives using Python's zipfile (no external zip/unzip needed).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import zipfile

import requests
from tqdm import tqdm


# Config for remote files

@dataclass
class RemoteFile:
    filename: str
    url: str
    md5: str


REMOTES: Dict[str, List[RemoteFile]] = {
    "FSD50K.dev_audio": [
        RemoteFile(
            filename="FSD50K.dev_audio.zip",
            url="https://zenodo.org/record/4060432/files/FSD50K.dev_audio.zip?download=1",
            md5="c480d119b8f7a7e32fdb58f3ea4d6c5a",
        ),
        RemoteFile(
            filename="FSD50K.dev_audio.z01",
            url="https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z01?download=1",
            md5="faa7cf4cc076fc34a44a479a5ed862a3",
        ),
        RemoteFile(
            filename="FSD50K.dev_audio.z02",
            url="https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z02?download=1",
            md5="8f9b66153e68571164fb1315d00bc7bc",
        ),
        RemoteFile(
            filename="FSD50K.dev_audio.z03",
            url="https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z03?download=1",
            md5="1196ef47d267a993d30fa98af54b7159",
        ),
        RemoteFile(
            filename="FSD50K.dev_audio.z04",
            url="https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z04?download=1",
            md5="d088ac4e11ba53daf9f7574c11cccac9",
        ),
        RemoteFile(
            filename="FSD50K.dev_audio.z05",
            url="https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z05?download=1",
            md5="81356521aa159accd3c35de22da28c7f",
        ),
    ],
    "FSD50K.eval_audio": [
        RemoteFile(
            filename="FSD50K.eval_audio.zip",
            url="https://zenodo.org/record/4060432/files/FSD50K.eval_audio.zip?download=1",
            md5="6fa47636c3a3ad5c7dfeba99f2637982",
        ),
        RemoteFile(
            filename="FSD50K.eval_audio.z01",
            url="https://zenodo.org/record/4060432/files/FSD50K.eval_audio.z01?download=1",
            md5="3090670eaeecc013ca1ff84fe4442aeb",
        ),
    ],
    "ground_truth": [
        RemoteFile(
            filename="FSD50K.ground_truth.zip",
            url="https://zenodo.org/record/4060432/files/FSD50K.ground_truth.zip?download=1",
            md5="ca27382c195e37d2269c4c866dd73485",
        )
    ],
    "metadata": [
        RemoteFile(
            filename="FSD50K.metadata.zip",
            url="https://zenodo.org/record/4060432/files/FSD50K.metadata.zip?download=1",
            md5="b9ea0c829a411c1d42adb9da539ed237",
        )
    ],
    "documentation": [
        RemoteFile(
            filename="FSD50K.doc.zip",
            url="https://zenodo.org/record/4060432/files/FSD50K.doc.zip?download=1",
            md5="3516162b82dc2945d3e7feba0904e800",
        )
    ],
}


@dataclass
class Config:
    root: Path
    skip_audio: bool
    skip_meta: bool
    skip_extract: bool


def parse_args() -> Config:
    ap = argparse.ArgumentParser(
        description="Download and unpack the FSD50K dataset for Events."
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("data/events/FSD50K"),
        help="Root directory where FSD50K will live (default: data/events/FSD50K)",
    )
    ap.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip downloading dev/eval audio archives.",
    )
    ap.add_argument(
        "--skip-meta",
        action="store_true",
        help="Skip downloading ground_truth/metadata/doc archives.",
    )
    ap.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extraction/unsplitting; only download files.",
    )
    args = ap.parse_args()

    return Config(
        root=args.root,
        skip_audio=args.skip_audio,
        skip_meta=args.skip_meta,
        skip_extract=args.skip_extract,
    )


# Helpers/download functions

def md5(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, expected_md5: Optional[str] = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"[SKIP] {dest.name} already exists")
        if expected_md5:
            actual = md5(dest)
            if actual != expected_md5:
                raise RuntimeError(
                    f"MD5 mismatch for existing file {dest} "
                    f"(expected {expected_md5}, got {actual})"
                )
        return

    print(f"[DOWNLOAD] {dest.name}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) or None
        with tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest.name,
        ) as pbar:
            with dest.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if not chunk:
                        continue
                    f.write(chunk)
                    pbar.update(len(chunk))

    if expected_md5:
        actual = md5(dest)
        if actual != expected_md5:
            raise RuntimeError(
                f"MD5 mismatch for {dest} after download "
                f"(expected {expected_md5}, got {actual})"
            )


def download_group(name: str, files: List[RemoteFile], root: Path) -> None:
    print(f"\n=== Download group: {name} ===")
    for rf in files:
        dest = root / rf.filename
        download_file(rf.url, dest, rf.md5)


def merge_split_zip(parts: List[Path], output_zip: Path) -> None:
    """
    Concatenate split zip parts (z01, z02, ..., .zip) into a single unsplit zip.
    """
    if output_zip.exists():
        print(f"[SKIP] {output_zip.name} already exists (merged)")
        return

    print(f"[MERGE] Creating {output_zip.name} from {len(parts)} parts")
    with output_zip.open("wb") as out_f:
        for p in parts:
            print(f"  + {p.name}")
            with p.open("rb") as in_f:
                for chunk in iter(lambda: in_f.read(1 << 20), b""):
                    out_f.write(chunk)


def extract_zip(zip_path: Path, dest: Path) -> None:
    """
    Extract a zip archive using Python's zipfile.
    """
    if not zip_path.exists():
        print(f"[WARN] Missing archive, cannot extract: {zip_path}")
        return
    print(f"[EXTRACT] {zip_path.name} -> {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)


# Extract all archives

def extract_all(cfg: Config) -> None:
    root = cfg.root
    print(f"\n=== Extracting archives under {root} ===")

    # 1) Dev audio (split zip)
    if not cfg.skip_audio:
        dev_files = REMOTES["FSD50K.dev_audio"]
        dev_paths = [root / rf.filename for rf in dev_files]
        if all(p.exists() for p in dev_paths):
            # Sort parts so z01..z05 come before .zip
            dev_parts_sorted = sorted(dev_paths, key=lambda p: p.suffix)
            dev_unsplit = root / "FSD50K.dev_audio.unsplit.zip"
            merge_split_zip(dev_parts_sorted, dev_unsplit)
            extract_zip(dev_unsplit, root)
        else:
            print("[WARN] Some dev audio parts missing; skipping dev audio extraction")

        # 2) Eval audio (split zip)
        eval_files = REMOTES["FSD50K.eval_audio"]
        eval_paths = [root / rf.filename for rf in eval_files]
        if all(p.exists() for p in eval_paths):
            eval_parts_sorted = sorted(eval_paths, key=lambda p: p.suffix)
            eval_unsplit = root / "FSD50K.eval_audio.unsplit.zip"
            merge_split_zip(eval_parts_sorted, eval_unsplit)
            extract_zip(eval_unsplit, root)
        else:
            print("[WARN] Some eval audio parts missing; skipping eval audio extraction")
    else:
        print("[INFO] Skipping audio extraction (--skip-audio)")

    # 3) Metadata (simple zips)
    if not cfg.skip_meta:
        for key in ("ground_truth", "metadata", "documentation"):
            for rf in REMOTES[key]:
                zip_path = root / rf.filename
                extract_zip(zip_path, root)
    else:
        print("[INFO] Skipping metadata/doc extraction (--skip-meta)")

    print("\n[INFO] Extraction finished. You should see directories like:")
    print(f"  {root}/FSD50K.dev_audio")
    print(f"  {root}/FSD50K.eval_audio")
    print(f"  {root}/FSD50K.ground_truth")
    print(f"  {root}/FSD50K.metadata")
    print(f"  {root}/FSD50K.doc")


# Main function

def main() -> None:
    cfg = parse_args()
    root = cfg.root
    root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] FSD50K root: {root}")

    # Downloads (skip files already downloaded)
    if not cfg.skip_audio:
        download_group("FSD50K.dev_audio", REMOTES["FSD50K.dev_audio"], root)
        download_group("FSD50K.eval_audio", REMOTES["FSD50K.eval_audio"], root)
    else:
        print("[INFO] Skipping audio downloads (--skip-audio)")

    if not cfg.skip_meta:
        for group in ("ground_truth", "metadata", "documentation"):
            download_group(group, REMOTES[group], root)
    else:
        print("[INFO] Skipping metadata/doc downloads (--skip-meta)")

    if not cfg.skip_extract:
        extract_all(cfg)
    else:
        print("[INFO] Skipping extraction (--skip-extract)")

    print("\n[DONE] FSD50K download script finished.")


if __name__ == "__main__":
    main()
