#!/usr/bin/env python3
"""
Extract anomaly (and normal) video frames for UCF-Crime and regenerate split files with updated frame counts.

Assumptions:
- Anomaly videos: <data_root>/data/Anomaly/<class>/<video_name>.<ext>
- Normal testing videos: <data_root>/data/Normal_Testing/<video_name>.<ext>
- Split files: video_name,label,orig_frame_count
- Output frames: JPEGs named <frame_idx>.jpg starting at 0
- Output structure:
    <frames_out>/training/<class>/<video_name>/<frame_idx>.jpg
    <frames_out>/testing/<class>/<video_name>/<frame_idx>.jpg
"""

import argparse
import glob
import os
import subprocess
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Root path containing data/Anomaly, Normal_Testing, split files.")
    ap.add_argument("--video_dir", default="data/Anomaly", help="Relative path from data_root to anomaly videos.")
    ap.add_argument("--normal_dir", default="data/Normal_Testing", help="Relative path from data_root to normal testing videos.")
    ap.add_argument("--frames_out", default="frames", help="Relative path from data_root to write extracted frames.")
    ap.add_argument("--train_split", default="Anomaly_Train.txt", help="Train split file (relative to data_root).")
    ap.add_argument("--test_split", default="Anomaly_Test.txt", help="Test split file (relative to data_root).")
    ap.add_argument("--train_split_out", default="Anomaly_Train_ver2.txt", help="Output train split with new frame counts.")
    ap.add_argument("--test_split_out", default="Anomaly_Test_ver2.txt", help="Output test split with new frame counts.")
    ap.add_argument("--ffmpeg_path", default="ffmpeg", help="ffmpeg executable.")
    ap.add_argument("--jpeg_quality", type=int, default=2,
                    help="ffmpeg -qscale:v value for JPEG (lower is better quality; 2~5 is common).")
    ap.add_argument("--overwrite", action="store_true", help="Re-extract even if frames already exist.")
    ap.add_argument("--train_only", action="store_true", help="Process only train split.")
    ap.add_argument("--test_only", action="store_true", help="Process only test split.")
    return ap.parse_args()


def find_video_path(video_dir: str, video_name: str, class_required: bool = True) -> Tuple[str, str]:
    exts = [".mp4", ".avi", ".mkv", ".mov"]
    for ext in exts:
        pattern = os.path.join(video_dir, "*", video_name + ext)
        matches = glob.glob(pattern)
        if matches:
            path = matches[0]
            cls = os.path.basename(os.path.dirname(path))
            return path, cls
    if not class_required:
        for ext in exts:
            pattern = os.path.join(video_dir, video_name + ext)
            matches = glob.glob(pattern)
            if matches:
                path = matches[0]
                cls = "Normal"
                return path, cls
    raise FileNotFoundError(f"Video {video_name} not found under {video_dir}")


def extract_frames(ffmpeg: str, src: str, dst_dir: str, overwrite: bool, jpeg_quality: int) -> int:
    os.makedirs(dst_dir, exist_ok=True)
    existing = glob.glob(os.path.join(dst_dir, "*.jpg"))
    if existing and not overwrite:
        return len(existing)

    out_pattern = os.path.join(dst_dir, "%d.jpg")
    cmd = [
        ffmpeg, "-y",
        "-i", src,
        "-start_number", "0",
        "-vsync", "0",
        "-qscale:v", str(jpeg_quality),
        out_pattern,
    ]
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frames = glob.glob(os.path.join(dst_dir, "*.jpg"))
    return len(frames)


def process_split(split_path: str, split_tag: str, args: argparse.Namespace) -> List[str]:
    entries = []
    base_dir = os.path.join(args.data_root, args.video_dir)
    normal_dir = os.path.join(args.data_root, args.normal_dir)
    frames_root = os.path.join(args.data_root, args.frames_out, split_tag)

    with open(split_path, "r") as fh:
        lines = [l.strip() for l in fh if l.strip()]

    for line in lines:
        parts = line.split(",")
        if len(parts) < 2:
            continue
        video_name, label = parts[0], parts[1]
        try:
            if video_name.lower().startswith("normal"):
                src_path, cls = find_video_path(normal_dir, video_name, class_required=False)
            else:
                src_path, cls = find_video_path(base_dir, video_name)
        except FileNotFoundError as e:
            print(f"[missing] {e}")
            continue

        dst_dir = os.path.join(frames_root, cls, video_name)
        frame_count = extract_frames(args.ffmpeg_path, src_path, dst_dir, args.overwrite, args.jpeg_quality)
        if frame_count == 0:
            print(f"[warn] {video_name}: no frames extracted")
            continue
        entries.append(f"{video_name},{label},{frame_count}")
        print(f"[done] {video_name}: {frame_count} frames -> {dst_dir}")

    return entries


def main():
    args = parse_args()
    if args.train_only and args.test_only:
        raise ValueError("Choose only one of --train_only or --test_only (or neither to run both).")
    train_path = os.path.join(args.data_root, args.train_split)
    test_path = os.path.join(args.data_root, args.test_split)

    train_out = os.path.join(args.data_root, args.train_split_out)
    test_out = os.path.join(args.data_root, args.test_split_out)

    train_entries: List[str] = []
    test_entries: List[str] = []

    if not args.test_only:
        train_entries = process_split(train_path, "training", args)
        with open(train_out, "w") as fh:
            fh.write("\n".join(train_entries))

    if not args.train_only:
        test_entries = process_split(test_path, "testing", args)
        with open(test_out, "w") as fh:
            fh.write("\n".join(test_entries))

    print(f"[summary] train videos: {len(train_entries)}, test videos: {len(test_entries)}")
    if train_entries:
        print(f"[summary] train split saved to {train_out}")
    if test_entries:
        print(f"[summary] test split saved to {test_out}")


if __name__ == "__main__":
    main()
