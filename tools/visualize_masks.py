#!/usr/bin/env python3
"""Overlay SAM2 masks, detection boxes, and score diagnostics onto frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score


def load_detection_payload(path: Path) -> Dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, np.ndarray) and payload.dtype == object:
        return payload.item()
    raise ValueError(f"Unsupported detection payload: {type(payload)}")


def load_frame_dict(path: Optional[Path]) -> Dict[str, np.ndarray]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    payload = np.load(path, allow_pickle=True)
    data = None
    if isinstance(payload, np.lib.npyio.NpzFile):
        if "frame_scores" in payload.files:
            arr = payload["frame_scores"]
            if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
                data = arr.flat[0]
        elif "data" in payload.files:
            arr = payload["data"]
            if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
                maybe_dict = arr.flat[0]
                if isinstance(maybe_dict, dict) and "frame_scores" in maybe_dict:
                    data = maybe_dict["frame_scores"]
    elif isinstance(payload, np.ndarray) and payload.dtype == object:
        data = payload.item()
    if data is None:
        raise ValueError(f"Could not parse frame_scores from {path}")
    return data


def load_frame_labels(path: Optional[Path], video: str) -> Optional[np.ndarray]:
    if path is None:
        return None
    if path.is_dir():
        video_path = path / f"{video}.npy"
        if video_path.exists():
            return np.load(video_path)
        return None
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.lib.npyio.NpzFile):
        if video in payload.files:
            return payload[video]
        if "data" in payload.files:
            arr = payload["data"]
            if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
                maybe_dict = arr.flat[0]
                if isinstance(maybe_dict, dict) and video in maybe_dict:
                    return np.asarray(maybe_dict[video])
    if isinstance(payload, np.ndarray) and payload.dtype == object:
        data = payload.item()
        if isinstance(data, dict) and video in data:
            return np.asarray(data[video])
    return None


def mask_path_for(frame_name: str, masks_dir: Path) -> Optional[Path]:
    base = Path(frame_name).stem
    candidates = [masks_dir / f"{base}_mask.png", masks_dir / f"{base}_mask.jpg", masks_dir / f"{base}.png"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def overlay_mask(frame: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float) -> np.ndarray:
    overlay = frame.copy()
    mask_bool = mask > 0
    overlay[mask_bool] = ((1 - alpha) * overlay[mask_bool] + alpha * np.array(color, dtype=np.float32)).astype(np.uint8)
    return overlay


def draw_boxes(frame: np.ndarray, boxes: np.ndarray, scores: np.ndarray, color=(0, 255, 0)):
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if scores is not None and idx < len(scores):
            text = f"{scores[idx]:.2f}"
            cv2.putText(frame, text, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def annotate(frame: np.ndarray, text_lines: List[str]):
    y = 20
    for line in text_lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18


def parse_args():
    parser = argparse.ArgumentParser(description="Overlay SAM2 masks and detection scores onto frames")
    parser.add_argument("--video", required=True, help="Video identifier (e.g., 01_0014)")
    parser.add_argument("--frames_root", required=True, help="Root directory containing <video> frame folders")
    parser.add_argument("--masks_root", required=True, help="Root directory containing SAM2 masks/<video>")
    parser.add_argument("--detections", required=True, help="Path to detections.npy for the video")
    parser.add_argument("--roi_scores", required=True, help="NPZ with frame_scores")
    parser.add_argument("--frame_scores", default=None, help="Optional npy/npz with LANP frame scores")
    parser.add_argument("--gt_labels", default=None, help="Optional directory/npz with frame-level GT labels")
    parser.add_argument("--output_dir", required=True, help="Directory to save overlay images")
    parser.add_argument("--alpha", type=float, default=0.5, help="Mask overlay alpha")
    parser.add_argument("--max_frames", type=int, default=None, help="Limit number of frames to export")
    parser.add_argument("--video_output", default=None, help="Optional mp4 path to save a video montage")
    return parser.parse_args()


def main():
    args = parse_args()
    video = args.video
    frames_dir = Path(args.frames_root) / video
    masks_dir = Path(args.masks_root) / video
    output_dir = Path(args.output_dir) / video
    output_dir.mkdir(parents=True, exist_ok=True)

    det_payload = load_detection_payload(Path(args.detections))
    frame_scores = load_frame_dict(Path(args.roi_scores))
    roi_seq = frame_scores.get(video, [])
    if not roi_seq:
        print(f"[warn] ROI frame scores missing for {video}")
    lanp_scores = None
    if args.frame_scores:
        lanp_dict = load_frame_dict(Path(args.frame_scores))
        lanp_scores = lanp_dict.get(video)
    gt_labels = load_frame_labels(Path(args.gt_labels), video) if args.gt_labels else None

    frame_files = det_payload.get("frame_files")
    frame_indices = det_payload.get("frame_indices")
    boxes_seq = det_payload.get("boxes")
    scores_seq = det_payload.get("scores")
    total = len(frame_indices)

    video_writer = None
    if args.video_output:
        args.video_output = str(Path(args.video_output))

    preds = []
    labels = []

    for idx in range(total):
        if args.max_frames is not None and idx >= args.max_frames:
            break
        frame_number = int(frame_indices[idx])
        frame_name = str(frame_files[idx])
        frame_path = frames_dir / frame_name
        if not frame_path.exists():
            print(f"[skip] frame missing: {frame_path}")
            continue
        frame = cv2.imread(str(frame_path))
        mask_path = mask_path_for(frame_name, masks_dir)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path else None
        overlay = frame.copy()
        if mask is not None:
            overlay = overlay_mask(overlay, mask, (0, 0, 255), args.alpha)
        boxes = np.asarray(boxes_seq[idx]) if len(boxes_seq) > idx else np.zeros((0, 4))
        roi_scores_frame = roi_seq[idx] if idx < len(roi_seq) else np.zeros((len(boxes),))
        draw_boxes(overlay, boxes, roi_scores_frame)

        score_line = f"ROI max: {np.max(roi_scores_frame) if len(roi_scores_frame) else 0:.2f}"
        if lanp_scores is not None and idx < len(lanp_scores):
            score_line += f" | LANP score: {lanp_scores[idx]:.2f}"
            preds.append(float(lanp_scores[idx]))
            if gt_labels is not None and idx < len(gt_labels):
                labels.append(int(gt_labels[idx]))
        text_lines = [f"Video: {video}", f"Frame: {frame_number}", score_line, f"Detections: {len(boxes)}"]
        annotate(overlay, text_lines)

        out_path = output_dir / f"{frame_number:06d}_overlay.jpg"
        cv2.imwrite(str(out_path), overlay)

        if args.video_output:
            if video_writer is None:
                h, w = overlay.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(args.video_output, fourcc, 15, (w, h))
            video_writer.write(overlay)

    if video_writer is not None:
        video_writer.release()
        print(f"[info] Saved video to {args.video_output}")

    if preds and labels and len(set(labels)) > 1:
        auc = roc_auc_score(labels, preds)
        print(f"[metrics] Frame ROC-AUC for {video}: {auc * 100:.2f}% based on {len(labels)} frames")
    else:
        print("[metrics] Not enough labeled frames to compute ROC-AUC")


if __name__ == "__main__":
    main()
