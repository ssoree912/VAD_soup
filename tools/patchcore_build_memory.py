#!/usr/bin/env python3
"""Build patch-core style memory bank from normal frames."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
from typing import List

import cv2
import numpy as np
import torch
from tqdm import tqdm

from tools.patchcore_backbone import ResNetFeatureExtractor, feat_to_patches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construct memory bank from normal frames.")
    parser.add_argument("--frames_root", required=True, help="Root of training frames (normal).")
    parser.add_argument("--output_path", required=True, help="Where to save memory npy.")
    parser.add_argument("--max_patches", type=int, default=50000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    extractor = ResNetFeatureExtractor(device=args.device).eval()

    frames_root = Path(args.frames_root)
    video_dirs = sorted([p for p in frames_root.iterdir() if p.is_dir()])

    memory: List[np.ndarray] = []

    for vid_dir in tqdm(video_dirs, desc="Videos"):
        frame_paths = sorted(list(vid_dir.glob("*.jpg")) + list(vid_dir.glob("*.png")))
        for frame_path in frame_paths:
            img = cv2.imread(str(frame_path))
            if img is None:
                continue
            with torch.no_grad():
                feat = extractor(img)
                patches = feat_to_patches(feat).cpu().numpy()
            for patch in patches:
                if len(memory) < args.max_patches:
                    memory.append(patch)
                else:
                    idx = rng.randint(0, len(memory) - 1)
                    if rng.random() < args.max_patches / float(len(memory) + 1):
                        memory[idx] = patch

    if not memory:
        raise RuntimeError("No patches collected.")
    memory_np = np.stack(memory, axis=0).astype(np.float32)
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_path, memory_np)
    print("saved memory:", args.output_path, "shape:", memory_np.shape)


if __name__ == "__main__":
    main()
