#!/usr/bin/env python3
"""Run YOLO on gated frames and store per-frame detections."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import numpy as np
from tqdm import tqdm

from detector import YOLODetector


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_gated_frames(path: Path) -> Dict[str, Set[int]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return {k: {int(v) for v in values} for k, values in payload.items()}
    mapping: Dict[str, Set[int]] = defaultdict(set)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            video, frame_idx = parts[0], int(parts[1])
            mapping[video].add(frame_idx)
    return mapping


def iter_frames(video_dir: Path) -> List[Path]:
    return sorted([p for p in video_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def detect_video(
    detector: YOLODetector,
    video_dir: Path,
    gated_indices: Optional[Set[int]],
    output_file: Path,
    overwrite: bool,
):
    if output_file.exists() and not overwrite:
        return 0, 0
    frame_files = iter_frames(video_dir)
    if not frame_files:
        return 0, 0
    boxes_list: List[np.ndarray] = []
    scores_list: List[np.ndarray] = []
    classes_list: List[np.ndarray] = []
    processed_flags: List[bool] = []
    frame_indices: List[int] = []
    frame_names: List[str] = []
    processed = 0
    gated_lookup = gated_indices or set()
    for idx, frame_path in enumerate(frame_files):
        frame_indices.append(idx)
        frame_names.append(frame_path.name)
        should_process = (gated_indices is None) or (idx in gated_lookup)
        processed_flags.append(should_process)
        if should_process:
            boxes, classes, scores = detector.detect_frame(frame_path)
            processed += 1
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            classes = np.zeros((0,), dtype=np.int32)
            scores = np.zeros((0,), dtype=np.float32)
        boxes_list.append(boxes)
        classes_list.append(classes)
        scores_list.append(scores)
    payload = {
        "video": video_dir.name,
        "frame_indices": np.asarray(frame_indices, dtype=np.int32),
        "frame_files": np.asarray(frame_names, dtype=object),
        "boxes": np.asarray(boxes_list, dtype=object),
        "scores": np.asarray(scores_list, dtype=object),
        "classes": np.asarray(classes_list, dtype=object),
        "processed": np.asarray(processed_flags, dtype=bool),
        "num_frames": len(frame_files),
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, payload, allow_pickle=True)
    return len(frame_files), processed


def build_argparser():
    parser = argparse.ArgumentParser(description="YOLO detection on gated frames")
    parser.add_argument("--data_root", required=True, help="Dataset root (e.g., ./data/shanghaitech)")
    parser.add_argument("--split", default="test", help="Dataset split (default: test)")
    parser.add_argument("--frames_dir", default="frames", help="Relative frames subdir (default: frames)")
    parser.add_argument("--output_root", required=True, help="Output root for detections")
    parser.add_argument("--gated_frames", default=None, help="Optional txt/json with 'video frame_idx' for gating")
    parser.add_argument("--videos", nargs="*", default=None, help="Optional subset of video names")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing detections")
    parser.add_argument("--model", default="yolov5s", help="YOLOv5 model name (default: yolov5s)")
    parser.add_argument("--weights", default=None, help="Custom YOLO weights path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", default="auto", help="Device override (auto/cpu/cuda:0 ...)")
    parser.add_argument("--filter_classes", nargs="*", default=("person",), help="Class names to keep")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    frames_root = Path(args.data_root) / args.split / args.frames_dir
    if not frames_root.exists():
        raise FileNotFoundError(frames_root)
    output_root = Path(args.output_root) / args.split
    gated_map = parse_gated_frames(Path(args.gated_frames)) if args.gated_frames else None
    detector = YOLODetector(
        model_name=args.model,
        weights=args.weights,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        imgsz=args.imgsz,
        filter_classes=args.filter_classes,
    )
    videos = args.videos or sorted([p.name for p in frames_root.iterdir() if p.is_dir()])
    total_frames = 0
    total_processed = 0
    for video in tqdm(videos, desc="YOLO"):
        video_dir = frames_root / video
        if not video_dir.exists():
            continue
        out_file = output_root / video / "detections.npy"
        frames, processed = detect_video(
            detector,
            video_dir,
            gated_map.get(video) if gated_map else None,
            out_file,
            args.overwrite,
        )
        total_frames += frames
        total_processed += processed
    print(f"[yolo] processed frames: {total_processed}/{total_frames}")


if __name__ == "__main__":
    main()
