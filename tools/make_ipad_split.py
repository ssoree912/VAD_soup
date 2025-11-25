#!/usr/bin/env python3
"""
Build split files for the IPAD dataset with a small portion of anomalous
videos moved into LANP training while keeping UNet training normal-only.

Outputs:
  - lanp_train_split.txt : normals from training + selected anomalous test videos
  - unet_train_split.txt : normals from training only
  - test_split.txt       : remaining test videos (normals + held-out anomalies)

Each line format matches the existing split convention:
    <relative_video_path>,<label>,<frame_len>
where <relative_video_path> is relative to the IPAD root, label is 0 (normal)
or 1 (anomalous), and frame_len is the number of frames.
"""

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate train/test splits for IPAD.")
    parser.add_argument(
        "--ipad-root",
        type=Path,
        default=Path("data/IPAD"),
        help="Root directory of the IPAD dataset.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/IPAD"),
        help="Directory to write the split txt files.",
    )
    parser.add_argument(
        "--train-anom-ratio",
        type=float,
        default=0.3,
        help="Fraction of anomalous test videos per scenario to move into LANP train.",
    )
    parser.add_argument(
        "--min-train-anom",
        type=int,
        default=1,
        help="Minimum number of anomalous videos to move per scenario (clipped by availability).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling anomalous training videos.",
    )
    return parser.parse_args()


def load_test_labels(label_dir: Path) -> List[Tuple[str, int, bool]]:
    """
    Returns a list of (video_id, frame_len, is_anom).
    video_id matches the folder name under testing/frames (zero-padded to width>=2).
    """
    if not label_dir.is_dir():
        return []
    items = []
    for npy_path in sorted(label_dir.glob("*.npy")):
        vid_raw = npy_path.stem  # e.g., '001'
        video_id = f"{int(vid_raw):02d}"
        label_arr = np.load(npy_path)
        frame_len = int(label_arr.shape[0])
        is_anom = bool(np.any(label_arr))
        items.append((video_id, frame_len, is_anom))
    return items


def count_frames(video_dir: Path) -> int:
    return len(
        [
            f
            for f in os.listdir(video_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    lanp_train: List[str] = []
    unet_train: List[str] = []
    test_split: List[str] = []
    summary: Dict[str, Dict[str, int]] = {}

    scenarios = sorted(
        d for d in os.listdir(args.ipad_root) if (args.ipad_root / d).is_dir()
    )
    for scenario in scenarios:
        sc_root = args.ipad_root / scenario
        train_frames_root = sc_root / "training" / "frames"
        test_frames_root = sc_root / "testing" / "frames"
        label_root = sc_root / "test_label"

        # Skip non-scenario folders (e.g., features/)
        if not train_frames_root.is_dir() or not test_frames_root.is_dir():
            continue

        # Normal training videos for both models.
        train_vids = sorted(
            d for d in os.listdir(train_frames_root) if (train_frames_root / d).is_dir()
        )
        for vid in train_vids:
            v_path = f"{scenario}/training/frames/{vid}"
            frame_len = count_frames(train_frames_root / vid)
            line = f"{v_path},0,{frame_len}\n"
            lanp_train.append(line)
            unet_train.append(line)

        # Test videos: split into anomaly/non-anomaly, then sample anomalies for LANP train.
        test_label_info = load_test_labels(label_root)
        anom_videos = [item for item in test_label_info if item[2]]
        normal_videos = [item for item in test_label_info if not item[2]]

        rng.shuffle(anom_videos)
        if len(anom_videos) == 0:
            n_train_anom = 0
        else:
            n_train_anom = max(
                args.min_train_anom, int(round(len(anom_videos) * args.train_anom_ratio))
            )
            n_train_anom = min(n_train_anom, len(anom_videos))
        train_anom = anom_videos[:n_train_anom]
        test_anom = anom_videos[n_train_anom:]

        for video_id, frame_len, _ in train_anom:
            v_path = f"{scenario}/testing/frames/{video_id}"
            lanp_train.append(f"{v_path},1,{frame_len}\n")

        for video_id, frame_len, is_anom in normal_videos + test_anom:
            v_path = f"{scenario}/testing/frames/{video_id}"
            label = 1 if is_anom else 0
            test_split.append(f"{v_path},{label},{frame_len}\n")

        summary[scenario] = {
            "train_normals": len(train_vids),
            "train_anoms": len(train_anom),
            "test_normals": len(normal_videos),
            "test_anoms": len(test_anom),
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "lanp_train_split.txt").write_text("".join(lanp_train))
    (args.out_dir / "unet_train_split.txt").write_text("".join(unet_train))
    (args.out_dir / "test_split.txt").write_text("".join(test_split))

    print("Wrote splits to", args.out_dir.resolve())
    for scenario, info in summary.items():
        print(
            f"{scenario}: LANP train (norm/anom) = {info['train_normals']}/{info['train_anoms']}, "
            f"test (norm/anom) = {info['test_normals']}/{info['test_anoms']}"
        )


if __name__ == "__main__":
    main()
