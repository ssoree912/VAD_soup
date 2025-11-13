#!/usr/bin/env python3
"""Build robust prompts from ROI scores and detections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from post.robust_filter import RobustFilterConfig, robust_filter


def load_detection_payload(path: Path) -> Dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, np.ndarray) and payload.dtype == object:
        return payload.item()
    raise ValueError(f"Unsupported detection payload: {type(payload)}")


def load_roi_frame_scores(path: Path) -> Dict[str, List[np.ndarray]]:
    payload = np.load(path, allow_pickle=True)
    data = None
    if isinstance(payload, np.lib.npyio.NpzFile):
        if "frame_scores" in payload.files:
            arr = payload["frame_scores"]
            if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
                data = arr.flat[0]
            else:
                print(f"[debug] frame_scores key present but unexpected format: type={type(arr)}, dtype={getattr(arr, 'dtype', None)}")
        elif "data" in payload.files:
            arr = payload["data"]
            if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
                maybe_dict = arr.flat[0]
                if isinstance(maybe_dict, dict) and "frame_scores" in maybe_dict:
                    data = maybe_dict["frame_scores"]
                else:
                    print("[debug] 'data' key found but frame_scores missing inside object")
            else:
                print(f"[debug] data key present but unexpected payload type: {type(arr)}")
    elif isinstance(payload, np.ndarray) and payload.dtype == object:
        data = payload.item()
    if data is None:
        raise ValueError(f"Could not find frame_scores in {path}")
    if not isinstance(data, dict):
        raise ValueError(f"frame_scores payload is not a dict (got {type(data)})")
    print(f"[debug] Loaded ROI frame scores for {len(data)} videos from {path}")
    return data


def center_xyxy(box: np.ndarray) -> List[float]:
    return [float(0.5 * (box[0] + box[2])), float(0.5 * (box[1] + box[3]))]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate robust prompts for SAM2")
    parser.add_argument("--detections_root", required=True, help="Directory with per-video detections.npy")
    parser.add_argument("--roi_scores", required=True, help="NPZ containing frame_scores")
    parser.add_argument("--frames_root", required=True, help="Root containing video frame folders")
    parser.add_argument("--output_root", required=True, help="Directory to store prompts per video")
    parser.add_argument("--videos", nargs="*", default=None, help="Optional subset of videos")
    parser.add_argument("--score_threshold", type=float, default=0.5, help="ROI score threshold tau")
    parser.add_argument("--window", type=int, default=5, help="Robust filter temporal window k")
    parser.add_argument("--iou_threshold", type=float, default=0.3, help="IoU threshold h")
    parser.add_argument("--min_hits", type=int, default=3, help="Min matches m")
    return parser.parse_args()


def main():
    args = parse_args()
    detections_root = Path(args.detections_root)
    frames_root = Path(args.frames_root)
    output_root = Path(args.output_root)
    frame_scores = load_roi_frame_scores(Path(args.roi_scores))
    det_files = sorted(detections_root.rglob("detections.npy"))
    if args.videos:
        allowed = set(args.videos)
        det_files = [f for f in det_files if f.parent.name in allowed]
    if not det_files:
        print(f"[warn] No detections found under {detections_root}; nothing to do")
    config = RobustFilterConfig(window=args.window, iou_threshold=args.iou_threshold, min_hits=args.min_hits)
    for det_file in det_files:
        video = det_file.parent.name
        if video not in frame_scores:
            print(f"[warn] Skipping {video}: ROI frame scores not found")
            continue
        payload = load_detection_payload(det_file)
        boxes_seq = [np.asarray(b).reshape(-1, 4) for b in payload["boxes"]]
        roi_seq = frame_scores[video]
        if len(boxes_seq) != len(roi_seq):
            print(f"[debug] Frame count mismatch for {video}: detections={len(boxes_seq)}, roi_scores={len(roi_seq)}")
        keep_masks = robust_filter(boxes_seq, roi_seq, args.score_threshold, config)
        frame_files = payload["frame_files"]
        frame_indices = payload["frame_indices"]
        prompts = []
        for frame_idx, (boxes, mask) in enumerate(zip(boxes_seq, keep_masks)):
            if boxes.size == 0:
                continue
            kept = np.asarray(mask).astype(bool)
            if not np.any(kept):
                continue
            scores = roi_seq[frame_idx]
            classes = payload["classes"][frame_idx]
            file_key = str(frame_files[frame_idx])
            abs_idx = int(frame_indices[frame_idx])
            for det_idx, keep_flag in enumerate(kept):
                if not keep_flag:
                    continue
                bbox = boxes[det_idx].tolist()
                prompts.append(
                    {
                        "frame_key": file_key,
                        "frame_index": abs_idx,
                        "bbox": bbox,
                        "center": center_xyxy(boxes[det_idx]),
                        "score": float(scores[det_idx]) if det_idx < len(scores) else float(args.score_threshold),
                        "class_id": int(classes[det_idx]) if det_idx < len(classes) else -1,
                    }
                )
        if not prompts:
            continue
        frames_dir = (frames_root / video).resolve()
        video_out = output_root / video
        video_out.mkdir(parents=True, exist_ok=True)
        payload_out = {"video": video, "frames_dir": str(frames_dir), "prompts": prompts}
        out_path = video_out / "robust_prompts.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(payload_out, fh, indent=2)
        print(f"[prompts] {video}: {len(prompts)} boxes saved -> {out_path}")


if __name__ == "__main__":
    main()
