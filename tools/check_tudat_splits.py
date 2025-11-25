#!/usr/bin/env python3
"""
Check TU-DAT split files for overlap and missing videos.

Usage:
python tools/check_tudat_splits.py \
  --video_root data/TU-DAT \
  --train_split data/TU-DAT/train_split_unet.txt \
  --test_split data/TU-DAT/test_split.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate TU-DAT split files.")
    p.add_argument("--video_root", default="data/TU-DAT", help="TU-DAT root directory")
    p.add_argument("--train_split", required=True, help="Path to train split file")
    p.add_argument("--test_split", required=True, help="Path to test split file")
    p.add_argument("--exts", nargs="+", default=[".mp4", ".mov", ".avi"], help="Allowed video extensions")
    return p.parse_args()


def load_split(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def find_video(video_root: Path, entry: str, exts: Sequence[str]) -> Path | None:
    parts = entry.split("/")
    if not parts:
        return None
    exts_norm = {e.lower() for e in exts}
    if parts[0] == "Rash-Driving":
        base = video_root / "Rash-Driving"
        rel = Path("/".join(parts[1:]))
    else:
        base = video_root / "Final_videos"
        rel = Path(entry)
    for ext in exts_norm:
        cand = (base / rel).with_suffix(ext)
        if cand.exists():
            return cand
    return None


def validate_split(entries: Iterable[str], video_root: Path, exts: Sequence[str]) -> Tuple[int, List[str]]:
    missing: List[str] = []
    count = 0
    for e in entries:
        if find_video(video_root, e, exts) is None:
            missing.append(e)
        else:
            count += 1
    return count, missing


def main() -> None:
    args = parse_args()
    video_root = Path(args.video_root)
    train_entries = load_split(Path(args.train_split))
    test_entries = load_split(Path(args.test_split))

    train_set = set(train_entries)
    test_set = set(test_entries)
    overlap = train_set & test_set

    print(f"Train entries: {len(train_set)}")
    print(f"Test entries: {len(test_set)}")
    print(f"Overlap: {len(overlap)}")
    if overlap:
        print(f"Overlap samples (first 10): {sorted(list(overlap))[:10]}")

    train_valid, train_missing = validate_split(train_set, video_root, args.exts)
    test_valid, test_missing = validate_split(test_set, video_root, args.exts)

    print(f"Train valid: {train_valid}, missing: {len(train_missing)}")
    if train_missing:
        print(f"Missing train samples (first 10): {train_missing[:10]}")
    print(f"Test valid: {test_valid}, missing: {len(test_missing)}")
    if test_missing:
        print(f"Missing test samples (first 10): {test_missing[:10]}")


if __name__ == "__main__":
    main()
