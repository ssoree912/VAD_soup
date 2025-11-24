#!/usr/bin/env python3
"""
Frame-level ROC/PR AUC for Attn U-Net error maps against ShanghaiTech frame masks.

Assumes error maps are saved per video as <frames_root>/<video>/<frame>_err.npy
 (produced by run_attn_unet_on_lanp_frames.py) and GT frame masks are in
 data/shanghaitech/testing/test_frame_mask/<video>.npy (0/1 per frame).

Frame score is aggregated from each error map (mean/max/percentile) and then
min-max normalized per video or globally, following the paper's frame-score
computation style.
unnet 을 이용한 프레임 수준 ROC/PR AUC 평가
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate Attn U-Net error maps with frame-level ROC/PR AUC.")
    p.add_argument("--errmaps_root", required=True, help="Root with per-video error maps (<video>/<frame>_err.npy).")
    p.add_argument("--gt_frame_masks", required=True, help="GT frame mask root (e.g., data/.../test_frame_mask).")
    p.add_argument("--agg", choices=["mean", "max", "p95"], default="mean",
                   help="How to aggregate each error map into a frame score.")
    p.add_argument("--norm", choices=["video", "global", "none"], default="video",
                   help="Min-max normalization scope for frame scores.")
    p.add_argument("--verbose", action="store_true", help="Print per-video AUCs.")
    return p.parse_args()


def load_gt_frames(root: Path, vid: str) -> np.ndarray:
    path = root / f"{vid}.npy"
    if not path.exists():
        raise FileNotFoundError(f"GT frame mask not found: {path}")
    masks = np.load(path, allow_pickle=True)
    masks = np.asarray(masks)
    if masks.ndim == 1:
        # already frame-level 0/1 labels
        return masks.astype(np.uint8).reshape(-1)
    if masks.ndim >= 3:
        return (masks.sum(axis=(1, 2)) > 0).astype(np.uint8)
    raise ValueError(f"Unsupported GT shape {masks.shape} for {vid}")


def agg_score(err: np.ndarray, mode: str) -> float:
    if mode == "mean":
        return float(err.mean())
    if mode == "max":
        return float(err.max())
    if mode == "p95":
        return float(np.percentile(err, 95))
    raise ValueError(f"Unsupported agg: {mode}")


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def collect_scores(err_root: Path, gt_root: Path, agg: str, norm: str) -> Tuple[List[float], List[int]]:
    all_scores: List[float] = []
    all_labels: List[int] = []
    per_video: Dict[str, Tuple[List[float], List[int]]] = {}

    for vid_dir in sorted(err_root.iterdir()):
        if not vid_dir.is_dir():
            continue
        vid = vid_dir.name
        try:
            gt = load_gt_frames(gt_root, vid)
        except FileNotFoundError:
            continue
        scores: List[float] = []
        labels: List[int] = []
        err_paths = sorted(vid_dir.glob("*_err.npy"))
        if not err_paths:
            continue
        for ep in err_paths:
            stem = ep.stem  # e.g., 000123_err
            frame_idx = None
            if stem.endswith("_err"):
                # take prefix before _err
                maybe = stem[: -len("_err")]
                if maybe.isdigit():
                    frame_idx = int(maybe)
            if frame_idx is None:
                parts = stem.split("_")
                for p in parts[::-1]:
                    if p.isdigit():
                        frame_idx = int(p)
                        break
            err = np.load(ep).astype(np.float32)
            if err.ndim == 3:
                err = err.squeeze()
            score = agg_score(err, agg)
            if frame_idx is None or frame_idx >= len(gt):
                continue
            scores.append(score)
            labels.append(int(gt[frame_idx]))

        if not scores:
            continue
        scores_arr = np.asarray(scores, dtype=np.float32)
        if norm == "video":
            scores_arr = minmax(scores_arr)
        per_video[vid] = (scores_arr.tolist(), labels)

    # global min-max if requested
    if norm == "global":
        all_raw = np.concatenate([np.asarray(v[0], dtype=np.float32) for v in per_video.values()])
        all_norm = minmax(all_raw)
        offset = 0
        for vid, (sc, lab) in per_video.items():
            n = len(sc)
            per_video[vid] = (all_norm[offset:offset + n].tolist(), lab)
            offset += n

    # collect
    for sc, lab in per_video.values():
        all_scores.extend(sc)
        all_labels.extend(lab)

    return all_scores, all_labels


def main() -> None:
    args = parse_args()
    err_root = Path(args.errmaps_root)
    gt_root = Path(args.gt_frame_masks)

    scores, labels = collect_scores(err_root, gt_root, args.agg, args.norm)
    scores_arr = np.asarray(scores, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.uint8)

    if labels_arr.size == 0:
        raise RuntimeError("No scores collected; check paths.")
    print(f"[info] frames collected: {len(scores_arr)}  positives: {labels_arr.sum()}  negatives: {len(labels_arr)-labels_arr.sum()}")

    if np.unique(labels_arr).size < 2:
        print("[warn] Only one class present; AUC undefined.")
        return

    roc = roc_auc_score(labels_arr, scores_arr)
    pr = average_precision_score(labels_arr, scores_arr)
    print(f"[frame] ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}  (agg={args.agg}, norm={args.norm})")


if __name__ == "__main__":
    main()
