#!/usr/bin/env python3
"""Quick utility to inspect frame/pixel GT masks for a video."""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path


def load_array(path: Path):
    arr = np.load(path, allow_pickle=True)
    return np.asarray(arr)


def main():
    ap = argparse.ArgumentParser(description="Inspect ShanghaiTech GT masks for one video")
    ap.add_argument("--video", required=True, help="Video id (e.g., 08_0179)")
    ap.add_argument("--frame_mask_root", default="data/shanghaitech/testing/test_frame_mask", help="Dir/npz for frame-level GT")
    ap.add_argument("--pixel_mask_root", default="data/shanghaitech/testing/test_pixel_mask", help="Dir/npz for pixel-level GT")
    args = ap.parse_args()

    vid = args.video
    frame_root = Path(args.frame_mask_root)
    pixel_root = Path(args.pixel_mask_root)

    # Frame-level
    frame_path = frame_root / f"{vid}.npy"
    frame_arr = None
    if frame_path.exists():
        frame_arr = load_array(frame_path)
    elif frame_root.is_file():
        payload = np.load(frame_root, allow_pickle=True)
        if isinstance(payload, np.lib.npyio.NpzFile) and vid in payload.files:
            frame_arr = np.asarray(payload[vid])
    if frame_arr is not None:
        frame_arr = np.asarray(frame_arr)
        if frame_arr.ndim > 1:
            frame_arr = (frame_arr > 0).any(axis=tuple(range(1, frame_arr.ndim)))
        frame_arr = frame_arr.astype(np.uint8).reshape(-1)
        total = len(frame_arr)
        positives = int(np.count_nonzero(frame_arr))
        print(f"[frame] {vid}: length={total}, positives={positives}")
        if positives > 0:
            pos_idxs = np.flatnonzero(frame_arr)
            print(f"         first positive frames: {pos_idxs[:10]}")
    else:
        print(f"[frame] {vid}: not found under {frame_root}")

    # Pixel-level
    pixel_path = pixel_root / f"{vid}.npy"
    pixel_arr = None
    if pixel_path.exists():
        pixel_arr = load_array(pixel_path)
    elif pixel_root.is_file():
        payload = np.load(pixel_root, allow_pickle=True)
        if isinstance(payload, np.lib.npyio.NpzFile) and vid in payload.files:
            pixel_arr = np.asarray(payload[vid])
    if pixel_arr is not None:
        pixel_arr = (np.asarray(pixel_arr) > 0)
        T = pixel_arr.shape[0]
        per_frame_pos = [int(np.count_nonzero(pixel_arr[t])) for t in range(T)]
        frames_with_gt = [t for t, cnt in enumerate(per_frame_pos) if cnt > 0]
        print(f"[pixel] {vid}: frames={T}, frames_with_gt={len(frames_with_gt)}")
        if frames_with_gt:
            print(f"         first GT frames: {frames_with_gt[:10]}")
    else:
        print(f"[pixel] {vid}: not found under {pixel_root}")

if __name__ == "__main__":
    main()
