#!/usr/bin/env python3
"""
Update split files with frame counts measured from extracted frames.

Keeps the original line order (video_name,label,frame_count).
Useful when 원본 영상을 지우고 프레임만 남은 상태에서 frame_count를 새로 세고 싶을 때.
"""

import argparse
import glob
import os
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_root", required=True,
                    help="Root to frames/<training|testing>. e.g., data/ucf-crime/frames/training")
    ap.add_argument("--split_in", required=True,
                    help="Input split txt (video_name,label,old_count). Order is preserved.")
    ap.add_argument("--split_out", default=None,
                    help="Output split txt. If not set, writes split_in + '.updated'.")
    ap.add_argument("--exts", nargs="+", default=["jpg", "png"],
                    help="Frame extensions to count.")
    return ap.parse_args()


def find_frames(frames_root: Path, video_name: str, exts: List[str]) -> List[str]:
    # search for nested directory video_name
    patterns = []
    patterns += [str(frames_root / "*" / video_name / f"*.*")]
    patterns += [str(frames_root / "*" / f"{video_name}*.*")]
    candidates: List[str] = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))
    if not candidates:
        for root, _, files in os.walk(frames_root):
            if os.path.basename(root) == video_name:
                for f in files:
                    candidates.append(os.path.join(root, f))
    frames = [p for p in candidates if p.lower().endswith(tuple(f".{e.lower()}" for e in exts))]
    return sorted(frames, key=lambda x: (os.path.dirname(x), x))


def main():
    args = parse_args()
    frames_root = Path(args.frames_root)
    split_in = Path(args.split_in)
    split_out = Path(args.split_out) if args.split_out else split_in.with_suffix(split_in.suffix + ".updated")

    lines_out = []
    missing = []

    with split_in.open("r") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    for ln in lines:
        parts = ln.split(",")
        if len(parts) < 2:
            continue
        video_name, label = parts[0], parts[1]
        frames = find_frames(frames_root, video_name, args.exts)
        if not frames:
            missing.append(video_name)
            continue
        lines_out.append(f"{video_name},{label},{len(frames)}")

    with split_out.open("w") as fh:
        fh.write("\n".join(lines_out))

    print(f"[summary] wrote {len(lines_out)} entries to {split_out}")
    if missing:
        print(f"[warn] missing frames for {len(missing)} videos (showing up to 10): {missing[:10]}")


if __name__ == "__main__":
    main()
