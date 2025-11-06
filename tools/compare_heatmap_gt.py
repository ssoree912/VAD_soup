#!/usr/bin/env python3

"""Compare occlusion heatmaps against ShanghaiTech test pixel masks.

Given a saved heatmap (typically `grid_fused_heatmap.npy`) and the ground-truth
pixel mask for the corresponding video, this script thresholds the heatmap with
Top-p pooling and reports IoU / precision / recall metrics.
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate occlusion heatmaps with pixel-level GT masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video-name", required=True, help="Video identifier (e.g. 01_0015).")
    parser.add_argument("--segment-index", type=int, required=True, help="Segment index used for the heatmap.")
    parser.add_argument("--heatmap-root", default="visualizations/heatmaps",
                        help="Root directory where heatmaps are stored.")
    parser.add_argument("--heatmap-tag", default="grid_fused",
                        help="Prefix of the heatmap file before '_heatmap.npy'.")
    parser.add_argument("--mask-root", default="data/shanghaitech/testing/test_frame_mask",
                        help="Directory containing per-frame GT masks (.npy).")
    parser.add_argument("--segment-len", type=int, default=16,
                        help="Temporal length of each segment.")
    parser.add_argument("--segment-stride", type=int, default=None,
                        help="Stride between segment starts; defaults to segment length.")
    parser.add_argument("--top-p", type=float, default=0.05,
                        help="Fraction of heatmap pixels to keep as positives.")
    parser.add_argument("--frame-mode", choices=["center", "union"], default="center",
                        help="Which GT frames to compare against: center frame or union over the snippet.")
    parser.add_argument("--verbose", action="store_true", help="Print extra diagnostics.")
    return parser.parse_args()


def load_heatmap(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Heatmap file not found: {path}")
    heatmap = np.load(path)
    if heatmap.ndim != 2:
        raise ValueError(f"Expected 2D heatmap, got shape {heatmap.shape}")
    return heatmap.astype(np.float32)


def load_gt_masks(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"GT mask file not found: {path}")
    masks = np.load(path)
    if masks.ndim != 3:
        raise ValueError(f"Expected GT mask with shape (frames, H, W), got {masks.shape}")
    return masks.astype(np.uint8)


def select_gt(snippet: np.ndarray, mode: str, default_index: int) -> np.ndarray:
    if mode == "center":
        idx = min(max(default_index, 0), snippet.shape[0] - 1)
        return snippet[idx] > 0
    if mode == "union":
        return (snippet > 0).any(axis=0)
    raise ValueError(f"Unsupported frame mode: {mode}")


def threshold_heatmap(heatmap: np.ndarray, top_p: float) -> Tuple[np.ndarray, float]:
    top_p = float(top_p)
    if not 0.0 < top_p < 1.0:
        raise ValueError(f"top-p must be in (0, 1); received {top_p}")
    flat = heatmap.flatten()
    kth = max(1, int(round(len(flat) * top_p)))
    kth = min(kth, len(flat))
    thresh = np.partition(flat, -kth)[-kth]
    mask = heatmap >= thresh
    return mask, float(thresh)


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    pred_pixels = pred.sum()
    gt_pixels = gt.sum()

    iou = intersection / union if union else 0.0
    precision = intersection / pred_pixels if pred_pixels else 0.0
    recall = intersection / gt_pixels if gt_pixels else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return float(iou), float(precision), float(recall), float(f1)


def main() -> None:
    args = parse_args()

    segment_stride = args.segment_stride or args.segment_len
    if segment_stride <= 0:
        raise ValueError("segment stride must be positive.")

    heatmap_path = (Path(args.heatmap_root)
                    / args.video_name
                    / f"seg{args.segment_index:03d}"
                    / f"{args.heatmap_tag}_heatmap.npy")
    gt_path = Path(args.mask_root) / f"{args.video_name}.npy"

    heatmap = load_heatmap(heatmap_path)
    gt_masks = load_gt_masks(gt_path)

    start = args.segment_index * segment_stride
    end = start + args.segment_len
    if start >= gt_masks.shape[0]:
        raise IndexError(f"Segment start {start} exceeds GT mask length {gt_masks.shape[0]} for video {args.video_name}.")

    snippet = gt_masks[start:min(end, gt_masks.shape[0])]
    if snippet.shape[0] < args.segment_len and args.verbose:
        print(f"[warn] snippet truncated to {snippet.shape[0]} frames (GT shorter than expected).")

    gt_mask = select_gt(snippet, args.frame_mode, args.segment_len // 2)
    pred_mask, threshold = threshold_heatmap(heatmap, args.top_p)

    iou, precision, recall, f1 = compute_metrics(pred_mask, gt_mask)

    print(f"video: {args.video_name}")
    print(f"segment: {args.segment_index}")
    print(f"heatmap: {heatmap_path}")
    print(f"gt_mask: {gt_path}")
    print(f"top_p: {args.top_p} (threshold={threshold:.6f})")
    print(f"metrics: IoU={iou:.4f}  Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}")

    if args.verbose:
        gt_pixels = gt_mask.sum()
        pred_pixels = pred_mask.sum()
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        print(f"[info] GT positive pixels: {gt_pixels}")
        print(f"[info] Pred positive pixels: {pred_pixels}")
        print(f"[info] Intersection: {intersection}")
        print(f"[info] Union: {np.logical_or(pred_mask, gt_mask).sum()}")


if __name__ == "__main__":
    main()
