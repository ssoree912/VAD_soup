#!/usr/bin/env python3
"""
Inspect top-scoring LANP frames for a given video and report GT presence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def load_gt_masks(gt_root: Path, video: str) -> np.ndarray:
    """
    Load frame-level GT masks (Shanghaitech test_frame_mask format).
    Returns a 0/1 array of shape (num_frames,).
    """
    path = gt_root / f"{video}.npy"
    arr = np.load(path, allow_pickle=True)
    masks = np.asarray(arr)
    if masks.ndim < 3:
        raise ValueError(f"Unexpected GT mask shape {masks.shape} for {video}")
    gt = (masks.sum(axis=(1, 2)) > 0).astype(np.int32)
    return gt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanp-scores", required=True, help="Path to LANP score dict npy/npz.")
    ap.add_argument("--video-name", required=True, help="Target video id (e.g., 01_0177).")
    ap.add_argument("--gt-root", required=True, help="Root dir containing <video>.npy GT masks.")
    ap.add_argument("--top-k", type=int, default=20, help="How many top frames to list.")
    args = ap.parse_args()

    scores_dict = np.load(args.lanp_scores, allow_pickle=True)
    if isinstance(scores_dict, np.lib.npyio.NpzFile):
        if "data" in scores_dict.files:
            scores_dict = scores_dict["data"].item()
        else:
            scores_dict = {k: scores_dict[k] for k in scores_dict.files}
    elif isinstance(scores_dict, np.ndarray) and scores_dict.dtype == object:
        scores_dict = scores_dict.item()

    if not isinstance(scores_dict, dict):
        raise ValueError("LANP scores must be a dict-like npy/npz with video keys.")
    if args.video_name not in scores_dict:
        raise KeyError(f"Video {args.video_name} not found in LANP scores.")

    scores = np.asarray(scores_dict[args.video_name], dtype=np.float32).reshape(-1)
    gt = load_gt_masks(Path(args.gt_root), args.video_name)
    if len(scores) != len(gt):
        raise ValueError(f"scores({len(scores)}) vs gt({len(gt)}) length mismatch for {args.video_name}")

    idxs = np.argsort(scores)[::-1][: args.top_k]

    print(f"=== {args.video_name} top-{args.top_k} frames by LANP score ===")
    for rank, i in enumerate(idxs, start=1):
        print(f"#{rank:02d}: frame {i:04d}  score={scores[i]:.6f}  gt={gt[i]}")


if __name__ == "__main__":
    main()
