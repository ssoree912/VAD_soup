#!/usr/bin/env python3
"""
Convert AE heatmaps into detections.npy with debugging:
- thresholding + morphology
- min_area + top-K filtering
- per-frame stats print
- optional debug overlay (frame + heatmap + boxes)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("AE heatmap -> detections.npy (with debug)")
    p.add_argument("--frames_root", required=True,
                   help="Root with per-video frame folders (e.g. data/shanghaitech/testing/frames)")
    p.add_argument("--heatmaps_root", required=True,
                   help="Root with per-video AE error maps (<video>/<frame>_err.npy)")
    p.add_argument("--output_root", required=True,
                   help="Where to save detections (per video)/detections.npy")

    # blob 파라미터
    p.add_argument("--err_percentile", type=float, default=95.0,
                   help="Per-frame percentile on err map for thresholding (e.g., 95, 98, 99)")
    p.add_argument("--min_area", type=int, default=30,
                   help="Minimum blob area in pixels (in heatmap resolution)")
    p.add_argument("--morph_kernel", type=int, default=0,
                   help="Morphology kernel size (0이면 morphology 비활성)")
    p.add_argument("--top_k", type=int, default=0,
                   help="Per-frame keep at most top-K blobs by score (0이면 전체 유지)")
    p.add_argument("--score_agg", choices=["mean", "max"], default="mean",
                   help="How to aggregate err inside each box")
    p.add_argument("--class_id", type=int, default=0,
                   help="Fixed class id for all detections")

    # 디버깅 옵션
    p.add_argument("--debug_root", type=str, default=None,
                   help="If set, save overlay images (frame + heatmap + boxes) here")
    p.add_argument("--debug_every", type=int, default=30,
                   help="Save debug image every N frames (per video)")
    return p.parse_args()


def mask_to_boxes_and_scores(
    err: np.ndarray,
    thr: float,
    min_area: int,
    score_agg: str,
    morph_kernel: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """err(H,W) -> (N,4) boxes, (N,) scores."""
    # threshold
    mask = (err >= thr).astype(np.uint8)

    # morphology (옵션)
    if morph_kernel > 0:
        k = np.ones((morph_kernel, morph_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    if mask.sum() == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    mask = np.ascontiguousarray(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = err.shape
    boxes: List[List[float]] = []
    scores: List[float] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < min_area:
            continue
        x2 = min(x + w, W)
        y2 = min(y + h, H)
        patch = err[y:y2, x:x2]
        if patch.size == 0:
            continue
        if score_agg == "max":
            s = float(patch.max())
        else:
            s = float(patch.mean())
        boxes.append([x, y, x2, y2])
        scores.append(s)

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    return np.asarray(boxes, dtype=np.float32), np.asarray(scores, dtype=np.float32)


def save_debug_overlay(
    out_path: Path,
    frame_bgr: np.ndarray,
    err: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    thr: float,
    nonzero_ratio: float,
):
    """원본 프레임 + heatmap + 박스 시각화."""
    h, w = frame_bgr.shape[:2]
    # err를 frame 크기로 resize
    err_norm = err.copy()
    err_norm = err_norm - err_norm.min()
    if err_norm.max() > 0:
        err_norm = err_norm / err_norm.max()
    err_up = cv2.resize(err_norm, (w, h), interpolation=cv2.INTER_LINEAR)
    err_col = cv2.applyColorMap((err_up * 255).astype(np.uint8), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(frame_bgr, 0.6, err_col, 0.4, 0.0)

    # 박스 그리기 (노란색)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
        if i < len(scores):
            cv2.putText(
                overlay,
                f"{scores[i]:.2f}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # 텍스트 정보
    txt1 = f"thr={thr:.4f}"
    txt2 = f"mask_ratio={nonzero_ratio*100:.2f}%  boxes={len(boxes)}"
    cv2.putText(overlay, txt1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(overlay, txt2, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


def main() -> None:
    args = parse_args()

    frames_root = Path(args.frames_root)
    heatmaps_root = Path(args.heatmaps_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    debug_root = Path(args.debug_root) if args.debug_root else None

    video_dirs = sorted([p for p in frames_root.iterdir() if p.is_dir()])

    for video_dir in tqdm(video_dirs, desc="Videos"):
        vid = video_dir.name
        heat_dir = heatmaps_root / vid
        if not heat_dir.exists():
            print(f"[warn] heatmaps for video {vid} not found at {heat_dir}")
            continue

        frame_paths = sorted(list(video_dir.glob("*.jpg")) + list(video_dir.glob("*.png")))
        if not frame_paths:
            print(f"[warn] no frames found for {vid}")
            continue

        num_frames = len(frame_paths)
        boxes_seq: List[np.ndarray] = [np.zeros((0, 4), dtype=np.float32) for _ in range(num_frames)]
        scores_seq: List[np.ndarray] = [np.zeros((0,), dtype=np.float32) for _ in range(num_frames)]
        classes_seq: List[np.ndarray] = [np.zeros((0,), dtype=np.int64) for _ in range(num_frames)]

        frames_with_dets = 0
        total_boxes = 0

        for fi, fp in enumerate(tqdm(frame_paths, desc=f"{vid}", leave=False)):
            stem = fp.stem
            err_path = heat_dir / f"{stem}_err.npy"
            if not err_path.exists():
                # AE 안 돌린 프레임일 수 있음
                continue

            err = np.load(err_path)
            if err.ndim == 3:
                err = err.squeeze()
            err = np.asarray(err, dtype=np.float32)

            frame_bgr = cv2.imread(str(fp))
            if frame_bgr is not None:
                frame_h, frame_w = frame_bgr.shape[:2]
            else:
                frame_h, frame_w = err.shape[:2]

            # per-frame percentile
            thr = float(np.percentile(err, args.err_percentile))
            mask = (err >= thr)
            nonzero_ratio = float(mask.mean())

            boxes, scores = mask_to_boxes_and_scores(
                err,
                thr,
                min_area=args.min_area,
                score_agg=args.score_agg,
                morph_kernel=args.morph_kernel,
            )

            # top-K 필터
            if args.top_k > 0 and boxes.shape[0] > args.top_k:
                order = np.argsort(scores)
                keep_idx = order[-args.top_k :]
                boxes = boxes[keep_idx]
                scores = scores[keep_idx]

            boxes_frame = boxes
            if boxes.shape[0] > 0:
                sx = frame_w / float(err.shape[1])
                sy = frame_h / float(err.shape[0])
                boxes_frame = boxes.copy()
                boxes_frame[:, [0, 2]] *= sx
                boxes_frame[:, [1, 3]] *= sy

                frames_with_dets += 1
                total_boxes += boxes.shape[0]
                boxes_seq[fi] = boxes_frame
                scores_seq[fi] = scores
                classes_seq[fi] = np.full((boxes.shape[0],), args.class_id, dtype=np.int64)

            # per-frame 디버그 로그
            print(
                f"[{vid} frame {fi}] err_mean={err.mean():.4f} "
                f"err_max={err.max():.4f} thr({args.err_percentile}%)={thr:.4f} "
                f"mask_ratio={nonzero_ratio*100:.2f}% boxes={boxes.shape[0]}"
            )

            # 디버그 overlay 저장
            if debug_root is not None and (fi % args.debug_every == 0):
                if frame_bgr is not None:
                    dbg_path = debug_root / vid / f"{fi:06d}_debug.jpg"
                    save_debug_overlay(
                        dbg_path,
                        frame_bgr,
                        err,
                        boxes_frame,
                        scores,
                        thr,
                        nonzero_ratio,
                    )

        print(
            f"[{vid}] num_frames={num_frames}, frames_with_dets={frames_with_dets}, "
            f"total_boxes={total_boxes}"
        )

        # detections.npy 저장 (LANP 파이프라인 포맷 맞추기)
        out_video_dir = output_root / vid
        out_video_dir.mkdir(parents=True, exist_ok=True)

        frame_files = np.array([fp.name for fp in frame_paths], dtype=object)
        frame_indices = np.arange(num_frames, dtype=np.int32)

        payload: Dict[str, np.ndarray] = {
            "boxes": np.array(boxes_seq, dtype=object),
            "scores": np.array(scores_seq, dtype=object),
            "classes": np.array(classes_seq, dtype=object),
            "frame_files": frame_files,
            "frame_indices": frame_indices,
            "num_frames": np.array(num_frames, dtype=np.int32),
        }
        np.save(out_video_dir / "detections.npy", payload, allow_pickle=True)
        print(f"[save] {vid} -> {out_video_dir / 'detections.npy'}")


if __name__ == "__main__":
    main()
