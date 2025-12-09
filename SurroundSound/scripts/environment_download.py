"""
Download 10-second AudioSet environment clips as 16 kHz mono WAVs.

Workflow per row (ytid,start,end):
  1) Try segmented download with yt-dlp (--download-sections), unless --no_segments is used.
  2) If that fails, download full audio then trim with ffmpeg.
  3) Transcode to WAV (16 kHz mono).
  4) Append a provenance record to metadata.jsonl.

Manifest format (CSV):
  Required columns: ytid, start, end
  Optional columns: label_ids, label_names (will be copied into metadata)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional


# Configuration dataclass

@dataclass
class Config:
    manifest: Path
    outdir: Path
    logdir: Path
    meta: Path
    retries: int = 2
    sleep: float = 1.0           # seconds between attempts
    limit: Optional[int] = None  # stop after N rows (for testing)
    no_segments: bool = False    # if True, skip segmented_download and always use full-download-then-trim


def parse_args() -> Config:
    ap = argparse.ArgumentParser(
        description="Download environment clips from an AudioSet manifest."
    )
    ap.add_argument(
        "--manifest",
        default="data/manifests/environment_segments.csv",
        type=Path,
        help="CSV with columns: ytid,start,end,(label_ids,label_names)",
    )
    ap.add_argument(
        "--outdir",
        default="data/environment/raw",
        type=Path,
        help="Output directory for WAV files",
    )
    ap.add_argument(
        "--logdir",
        default="data/environment/logs",
        type=Path,
        help="Directory for per-clip logs from yt-dlp/ffmpeg",
    )
    ap.add_argument(
        "--meta",
        default="data/environment/metadata.jsonl",
        type=Path,
        help="Path to JSONL provenance file (appended)",
    )
    ap.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Number of retries per clip (in addition to first attempt)",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Sleep between attempts (seconds)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N rows (debug)",
    )
    ap.add_argument(
        "--no_segments",
        action="store_true",
        help="Disable yt-dlp --download-sections; always full-download then trim.",
    )
    args = ap.parse_args()

    cfg = Config(
        manifest=args.manifest,
        outdir=args.outdir,
        logdir=args.logdir,
        meta=args.meta,
        retries=args.retries,
        sleep=args.sleep,
        limit=args.limit,
        no_segments=args.no_segments,
    )
    return cfg


# Utilities

def ensure_dirs(cfg: Config) -> None:
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    cfg.logdir.mkdir(parents=True, exist_ok=True)
    cfg.meta.parent.mkdir(parents=True, exist_ok=True)
    cfg.meta.touch(exist_ok=True)


def md5(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: Iterable[str], log_file: Path) -> int:
    """Run a command, append stdout/stderr to log_file, return exit code."""
    with log_file.open("a") as lf:
        lf.write(
            f"\n== {time.strftime('%Y-%m-%d %H:%M:%S')} :: {' '.join(map(str, cmd))} ==\n"
        )
        return subprocess.call(list(cmd), stdout=lf, stderr=lf)


def transcode_to_wav(src: Path, dst_wav: Path, log_file: Path) -> bool:
    """ffmpeg -> 16 kHz mono WAV."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(dst_wav),
    ]
    rc = run(cmd, log_file)
    return rc == 0 and dst_wav.exists()


def segmented_download(
    url: str, start: str, end: str, tmp_base: Path, log_file: Path
) -> Optional[Path]:
    """
    Attempt direct segment download via yt-dlp --download-sections.
    Returns the extracted audio file path if successful, else None.
    """
    cmd = [
        "yt-dlp",
        "-f",
        "bestaudio/best",
        "--no-warnings",
        "--quiet",
        "--download-sections",
        f"*{start}-{end}",
        "-o",
        f"{tmp_base}.%(ext)s",
        url,
    ]
    rc = run(cmd, log_file)
    if rc != 0:
        return None
    
    matches = list(tmp_base.parent.glob(f"{tmp_base.name}.*"))
    return matches[0] if matches else None


def full_download_then_trim(
    url: str, start: str, end: str, tmp_base: Path, log_file: Path
) -> Optional[Path]:
    """
    Fallback: download full audio, then trim with ffmpeg.
    Returns path to trimmed audio file if successful, else None.
    """
    full_tmp = tmp_base.with_suffix(".full.tmp")

    # 1) full download
    rc = run(
        ["yt-dlp", "-f", "bestaudio/best", "-o", f"{full_tmp}.%(ext)s", url],
        log_file,
    )
    if rc != 0:
        return None
    full_files = list(full_tmp.parent.glob(f"{full_tmp.name}.*"))
    if not full_files:
        return None
    src = full_files[0]

    # 2) trim to [start, end]
    trimmed = tmp_base.with_suffix(".trim.wav")
    rc = run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            start,
            "-to",
            end,
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(trimmed),
        ],
        log_file,
    )

    # cleanup full file
    try:
        src.unlink(missing_ok=True)  # type: ignore[arg-type]
    except TypeError:
        if src.exists():
            src.unlink()

    return trimmed if rc == 0 and trimmed.exists() else None


def write_metadata(meta_path: Path, record: Dict) -> None:
    with meta_path.open("a") as f:
        f.write(json.dumps(record) + "\n")


# Core processing

def process_row(row: Dict[str, str], cfg: Config) -> None:
    """
    Process a single manifest row.
    Expected keys: ytid, start, end (strings). Optional: label_ids, label_names.
    """
    ytid = row["ytid"].strip()
    start = str(row["start"]).strip()
    end = str(row["end"]).strip()

    stem = f"{ytid}_{start}_{end}"
    wav_path = cfg.outdir / f"{stem}.wav"
    log_file = cfg.logdir / f"{stem}.log"

    if wav_path.exists():
        # Already present; still record provenance if missing
        write_metadata(
            cfg.meta,
            {
                "ytid": ytid,
                "start": float(start),
                "end": float(end),
                "url": f"https://www.youtube.com/watch?v={ytid}",
                "path": str(wav_path),
                "status": "exists",
                "md5": md5(wav_path),
                "ts": time.time(),
                "label_ids": row.get("label_ids"),
                "label_names": row.get("label_names"),
            },
        )
        return

    url = f"https://www.youtube.com/watch?v={ytid}"
    tmp_base = cfg.outdir / f"{stem}.tmp"

    ok = False
    attempts = cfg.retries + 1  # first try + N retries

    for attempt in range(1, attempts + 1):
        with log_file.open("a") as lf:
            lf.write(f"\n== Attempt {attempt}/{attempts} ==\n")

        seg = None

        # 1) segmented download path (unless disabled)
        if not cfg.no_segments:
            seg = segmented_download(url, start, end, tmp_base, log_file)
            if seg is not None:
                ok = transcode_to_wav(seg, wav_path, log_file)
                # cleanup segmented source
                try:
                    seg.unlink(missing_ok=True)  # type: ignore[arg-type]
                except TypeError:
                    if seg.exists():
                        seg.unlink()
                if ok:
                    break  # success, done with attempts

        # 2) fallback: full download then trim
        trimmed = full_download_then_trim(url, start, end, tmp_base, log_file)
        if trimmed is not None:
            # If trimmed is already WAV @16k mono, just rename/move
            trimmed.replace(wav_path)
            ok = wav_path.exists()
            if ok:
                break

        # wait before retry
        time.sleep(cfg.sleep)

    # provenance record
    record = {
        "ytid": ytid,
        "start": float(start),
        "end": float(end),
        "url": url,
        "path": str(wav_path),
        "status": "ok" if ok else "fail",
        "md5": md5(wav_path) if ok else None,
        "ts": time.time(),
        "label_ids": row.get("label_ids"),
        "label_names": row.get("label_names"),
    }
    write_metadata(cfg.meta, record)


def read_manifest(path: Path) -> Iterable[Dict[str, str]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        required = {"ytid", "start", "end"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing required columns: {missing}")
        for row in reader:
            yield row


# Main execution

def main() -> None:
    cfg = parse_args()
    ensure_dirs(cfg)

    # Ctrl-C: exit loop but keep already-written metadata
    interrupted = {"flag": False}

    def _sigint(_sig, _frm):
        interrupted["flag"] = True
        print(
            "\n[INFO] Interrupted by user. Finishing current item and exiting...",
            flush=True,
        )

    signal.signal(signal.SIGINT, _sigint)

    count = 0
    print(f"[INFO] Reading manifest: {cfg.manifest}")
    for row in read_manifest(cfg.manifest):
        if interrupted["flag"]:
            break
        process_row(row, cfg)
        count += 1
        if cfg.limit and count >= cfg.limit:
            break
        if count % 25 == 0:
            print(f"[INFO] Processed {count} rows...")

    print(f"[DONE] Processed {count} rows. Metadata → {cfg.meta}")


if __name__ == "__main__":
    main()
