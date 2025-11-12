#!/usr/bin/env python3
"""Extract ROI features and anomaly scores from gated detections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

from lanp import FrameFeatureBackbone
from lanp.train import reduce_roi_scores
from features import roi_feature_vectors
from features.roi_scorer import ROIScorer


def load_detection_payload(path: Path) -> Dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.dtype == object:
        return payload.item()
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unexpected detection container: {type(payload)}")


def load_memory(path: Path, device: torch.device) -> torch.Tensor:
    payload = np.load(path, allow_pickle=True)
    arr = None
    if isinstance(payload, np.lib.npyio.NpzFile):
        for key in ("normal_memory", "memory", "arr_0"):
            if key in payload.files:
                candidate = payload[key]
                if isinstance(candidate, np.ndarray):
                    arr = candidate
                    break
    elif isinstance(payload, np.ndarray):
        arr = payload
    if arr is None:
        raise ValueError(f"Unable to parse memory tensor from {path}")
    memory = torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(device)
    return memory


def resolve_frame_path(video_dir: Path, frame_key: str, frame_idx: int) -> Path:
    candidate = video_dir / frame_key
    if candidate.exists():
        return candidate
    alt = video_dir / f"{frame_idx:06d}.jpg"
    if alt.exists():
        return alt
    png_alt = alt.with_suffix(".png")
    if png_alt.exists():
        return png_alt
    raise FileNotFoundError(f"Frame not found for key={frame_key} idx={frame_idx} in {video_dir}")


def save_features(path: Path, payload: Dict[str, List[np.ndarray]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, payload, allow_pickle=True)


def save_scores(path: Path, frame_scores: Dict[str, List[np.ndarray]], snippet_scores: Dict[str, np.ndarray]):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frame_scores=np.array([frame_scores], dtype=object),
        snippet_scores=np.array([snippet_scores], dtype=object),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Compute ROI features and scores")
    parser.add_argument("--detections_root", required=True, help="Path to detection outputs (*/<video>/detections.npy)")
    parser.add_argument("--frames_root", required=True, help="Root directory containing frame folders per video")
    parser.add_argument("--memory_path", required=True, help="Path to saved LANP normal memory npy/npz")
    parser.add_argument("--output_features", required=True, help="Path to save ROI feature dictionary")
    parser.add_argument("--output_scores", required=True, help="Path to save ROI score npz")
    parser.add_argument("--seg_len", type=int, required=True, help="Segment length for snippet reduction")
    parser.add_argument("--pool_size", type=int, nargs=2, default=[7, 7], help="ROIAlign output size (H W)")
    parser.add_argument("--pooling", choices=["avg", "max"], default="avg")
    parser.add_argument("--score_reducer", choices=["max", "mean"], default="max", help="Reducer for snippet scores")
    parser.add_argument("--device", default=None, help="Device for feature backbone/memory")
    parser.add_argument("--backbone_arch", default="resnet50", help="2D backbone architecture (default: resnet50)")
    parser.add_argument("--videos", nargs="*", default=None, help="Optional subset of videos to process")
    parser.add_argument("--summary", type=str, default=None, help="Optional JSON summary output path")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = FrameFeatureBackbone(arch=args.backbone_arch, pretrained=True, device=device)
    memory = load_memory(Path(args.memory_path), device)
    feature_dim = int(memory.shape[1]) if memory.ndim == 2 and memory.shape[1] > 0 else 2048
    scorer = ROIScorer(memory)
    detections_root = Path(args.detections_root)
    frames_root = Path(args.frames_root)
    det_files = sorted(detections_root.rglob("detections.npy"))
    if args.videos:
        allowed = set(args.videos)
        det_files = [f for f in det_files if f.parent.name in allowed]
    features_payload: Dict[str, List[np.ndarray]] = {}
    frame_scores_map: Dict[str, List[np.ndarray]] = {}
    summary = {}
    pool_size = tuple(args.pool_size)
    for det_file in tqdm(det_files, desc="ROI"):
        video = det_file.parent.name
        payload = load_detection_payload(det_file)
        boxes_seq = payload["boxes"]
        frame_files = payload["frame_files"]
        frame_indices = payload["frame_indices"]
        video_dir = frames_root / video
        num_frames = int(payload.get("num_frames", len(frame_indices)))
        per_frame_features: List[np.ndarray] = [np.zeros((0, feature_dim), dtype=np.float32) for _ in range(num_frames)]
        per_frame_scores: List[np.ndarray] = [np.zeros((0,), dtype=np.float32) for _ in range(num_frames)]
        for idx in range(min(len(frame_indices), num_frames)):
            boxes = np.asarray(boxes_seq[idx])
            if boxes.size == 0:
                continue
            frame_path = resolve_frame_path(video_dir, str(frame_files[idx]), int(frame_indices[idx]))
            image_bgr = cv2.imread(str(frame_path))
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            feature_map = backbone.extract_feature_map(image_rgb)
            spatial_scale = feature_map.shape[-1] / max(image_rgb.shape[1], 1)
            torch_boxes = torch.from_numpy(boxes.astype(np.float32))
            pooled = roi_feature_vectors(
                feature_map.unsqueeze(0),
                [torch_boxes],
                output_size=pool_size,
                spatial_scale=spatial_scale,
                pooling=args.pooling,
            )
            feats_np = pooled.cpu().numpy().astype(np.float32)
            per_frame_features[int(frame_indices[idx])] = feats_np
            roi_scores = scorer.score(pooled).detach().cpu().numpy().astype(np.float32)
            per_frame_scores[int(frame_indices[idx])] = roi_scores
        features_payload[video] = per_frame_features
        frame_scores_map[video] = per_frame_scores
        summary[video] = {
            "frames_with_detections": int(sum(1 for scores in per_frame_scores if len(scores))),
            "total_frames": num_frames,
        }
    snippet_scores = reduce_roi_scores(frame_scores_map, args.seg_len, reducer=args.score_reducer)
    save_features(Path(args.output_features), features_payload)
    save_scores(Path(args.output_scores), frame_scores_map, snippet_scores)
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
