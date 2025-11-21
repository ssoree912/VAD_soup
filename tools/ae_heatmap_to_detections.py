#!/usr/bin/env python3
"""Convert AE heatmaps into blob detections compatible with LANP pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Threshold AE heatmaps, extract blobs, and save detections.npy per video."
    )
    parser.add_argument("--frames_root", required=True, help="Root with <video> frame folders.")
    parser.add_argument("--heatmaps_root", required=True, help="Root with AE maps: <video>/<frame>_err.npy")
    parser.add_argument("--output_root", required=True, help="Where <video>/detections.npy will be written.")
    parser.add_argument("--videos", nargs="*", default=None, help="Optional subset of video ids.")
    parser.add_argument(
        "--err_percentile",
        type=float,
        default=95.0,
        help="Per-frame percentile threshold (e.g., 95 keeps top 5% pixels).",
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=30,
        help="Minimum blob area (in heatmap px) to keep as a detection.",
    )
    parser.add_argument(
        "--morph_kernel",
        type=int,
        default=5,
        help="Kernel size for morphological open/close (0 to disable).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="If set, keep only top-K blobs per frame by score.",
    )
    parser.add_argument(
        "--score_agg",
        choices=["mean", "max"],
        default="mean",
        help="Aggregate error inside blob for detection score.",
    )
    parser.add_argument(
        "--class_id",
        type=int,
        default=0,
        help="Class id assigned to every blob (compatibility with downstream code).",
    )
    return parser.parse_args()


def load_frames(video_dir: Path) -> List[Path]:
    return sorted(list(video_dir.glob("*.jpg")) + list(video_dir.glob("*.png")))


def load_err_map(hmap_dir: Path, stem: str) -> np.ndarray | None:
    path = hmap_dir / f"{stem}_err.npy"
    if not path.exists():
        return None
    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.squeeze()
    return arr


def err_to_mask(err: np.ndarray, percentile: float) -> np.ndarray:
    err = np.nan_to_num(err, nan=0.0, posinf=0.0, neginf=0.0)
    thr = np.percentile(err, percentile)
    if thr <= 0:
        thr = float(err.mean()) + 1e-6
    return (err >= thr).astype(np.uint8)


def mask_to_boxes_scores(
    mask: np.ndarray,
    err: np.ndarray,
    min_area: int,
    score_agg: str,
    morph_kernel: int = 0,
    top_k: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    mask_u8 = (mask > 0).astype(np.uint8)
    if morph_kernel and morph_kernel > 0:
        k = np.ones((morph_kernel, morph_kernel), np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k)
    mask_u8 = np.ascontiguousarray(mask_u8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    scores = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area:
            continue
        x1, y1, x2, y2 = x, y, x + w, y + h
        boxes.append([x1, y1, x2, y2])
        patch = err[y1:y2, x1:x2].reshape(-1)
        if patch.size == 0:
            scores.append(0.0)
        else:
            scores.append(float(patch.mean() if score_agg == "mean" else patch.max()))
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    boxes_arr = np.asarray(boxes, dtype=np.float32)
    scores_arr = np.asarray(scores, dtype=np.float32)

    if top_k is not None and top_k > 0 and scores_arr.shape[0] > top_k:
        order = np.argsort(scores_arr)[-top_k:]
        boxes_arr = boxes_arr[order]
        scores_arr = scores_arr[order]

    return boxes_arr, scores_arr


def rescale_boxes(boxes: np.ndarray, src_hw: Tuple[int, int], dst_hw: Tuple[int, int]) -> np.ndarray:
    if boxes.size == 0:
        return boxes.astype(np.float32)
    sy = dst_hw[0] / max(float(src_hw[0]), 1.0)
    sx = dst_hw[1] / max(float(src_hw[1]), 1.0)
    scaled = boxes.copy().astype(np.float32)
    scaled[:, [0, 2]] *= sx
    scaled[:, [1, 3]] *= sy
    return scaled


def main() -> None:
    args = parse_args()

    frames_root = Path(args.frames_root)
    heatmaps_root = Path(args.heatmaps_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    video_dirs = sorted([p for p in frames_root.iterdir() if p.is_dir()])
    if args.videos:
        allowed = set(args.videos)
        video_dirs = [p for p in video_dirs if p.name in allowed]

    for video_dir in tqdm(video_dirs, desc="Videos"):
        vid = video_dir.name
        hm_dir = heatmaps_root / vid
        if not hm_dir.exists():
            print(f"[warn] missing heatmap dir for {vid}, skipping")
            continue
        frames = load_frames(video_dir)
        if not frames:
            print(f"[warn] no frames found for {vid}")
            continue

        frame_files: List[str] = []
        frame_indices: List[int] = []
        boxes_seq: List[np.ndarray] = []
        scores_seq: List[np.ndarray] = []
        classes_seq: List[np.ndarray] = []

        for idx, frame_path in enumerate(frames):
            frame_files.append(frame_path.name)
            frame_indices.append(idx)

            err = load_err_map(hm_dir, frame_path.stem)
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
            classes = np.zeros((0,), dtype=np.int64)

            if err is not None:
                mask = err_to_mask(err, args.err_percentile)
                boxes_ae, scores_ae = mask_to_boxes_scores(
                    mask,
                    err,
                    args.min_area,
                    args.score_agg,
                    args.morph_kernel,
                    args.top_k,
                )
                frame_img = cv2.imread(str(frame_path))
                if frame_img is None:
                    frame_h, frame_w = err.shape[:2]
                else:
                    frame_h, frame_w = frame_img.shape[:2]
                boxes = rescale_boxes(boxes_ae, err.shape[:2], (frame_h, frame_w))
                scores = scores_ae
                classes = np.full((boxes.shape[0],), int(args.class_id), dtype=np.int64)

            boxes_seq.append(boxes)
            scores_seq.append(scores)
            classes_seq.append(classes)

        payload: Dict[str, object] = {
            "frame_files": np.array(frame_files, dtype=object),
            "frame_indices": np.array(frame_indices, dtype=np.int32),
            "boxes": np.array(boxes_seq, dtype=object),
            "scores": np.array(scores_seq, dtype=object),
            "classes": np.array(classes_seq, dtype=object),
            "num_frames": len(frames),
        }

        out_dir = output_root / vid
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "detections.npy"
        np.save(out_path, payload, allow_pickle=True)
        print(f"[{vid}] detections saved to {out_path}")


if __name__ == "__main__":
    main()
