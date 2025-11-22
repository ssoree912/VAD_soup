#!/usr/bin/env python3
"""Run YOLO-World only on anomaly heatmaps and save LANP-compatible detections."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from detector import YOLOWorldDetector  # 네가 만든 YOLOWorldDetector


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Heatmap-gated YOLO-World detections.")
    parser.add_argument("--frames_root", required=True, help="Root with <video> frame folders.")
    parser.add_argument("--heatmaps_root", required=True, help="Root with <video>/<frame>_err.npy heatmaps.")
    parser.add_argument("--output_root", required=True, help="Where <video>/detections.npy will be saved.")
    parser.add_argument("--videos", nargs="*", default=None, help="Optional subset of video ids.")
    parser.add_argument(
        "--split_file",
        default=None,
        help="Optional split file (e.g., test_split.txt). If provided, videos are read from here unless --videos is set.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config (e.g., config_sh.yaml). If provided, uses testing_split inside the config when --videos/--split_file are not set.",
    )

    parser.add_argument("--weights", default="yolov8l-world.pt")
    parser.add_argument(
        "--classes",
        nargs="*",
        default=[
            "person",
            "crowd",
            "person running",
            "person falling",
            "person biking",
            "person skating",
            "abandoned object",
            "suspicious bag",
        ],
        help="YOLO-World textual prompts.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)

    # 프레임 게이트
    parser.add_argument("--frame_gate_value", type=float, default=None,
                        help="Absolute threshold on heatmap.max.")
    parser.add_argument(
        "--frame_gate_percentile",
        type=float,
        default=None,
        help="Percentile over heatmap maxima to derive gate threshold (dataset-wide).",
    )

    # 박스 필터링
    parser.add_argument("--box_heat_threshold", type=float, default=None,
                        help="Absolute threshold on box heat score.")
    parser.add_argument(
        "--box_heat_percentile",
        type=float,
        default=95.0,
        help="Per-frame percentile of heatmap used as box heat threshold (ignored if box_heat_threshold is set).",
    )
    parser.add_argument("--top_k", type=int, default=3, help="Per frame keep at most K boxes after filtering.")
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def iter_frames(video_dir: Path) -> List[Path]:
    """Return sorted list of frame paths inside a video directory.

    Uses glob on common image extensions to be robust to directory contents.
    """
    frames = sorted(
        list(video_dir.glob("*.jpg"))
        + list(video_dir.glob("*.jpeg"))
        + list(video_dir.glob("*.png"))
        + list(video_dir.glob("*.bmp"))
    )
    return frames


def collect_heatmap_maxima(root: Path, videos: Sequence[str]) -> List[float]:
    maxima: List[float] = []
    for vid in videos:
        vid_dir = root / vid
        if not vid_dir.exists():
            continue
        for err_path in vid_dir.glob("*_err.npy"):
            arr = np.load(err_path)
            maxima.append(float(arr.max()))
    return maxima


def box_heat_scores(boxes: np.ndarray, heatmap: np.ndarray, frame_shape: Tuple[int, int]) -> np.ndarray:
    """각 bbox 안에서 heatmap 평균값을 계산."""
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)

    frame_h, frame_w = frame_shape
    heat_h, heat_w = heatmap.shape

    sx = heat_w / max(frame_w, 1)
    sy = heat_h / max(frame_h, 1)

    scores = np.zeros((boxes.shape[0],), dtype=np.float32)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(float)

        hx1 = int(np.clip(np.floor(x1 * sx), 0, heat_w - 1))
        hx2 = int(np.clip(np.ceil(x2 * sx), hx1 + 1, heat_w))
        hy1 = int(np.clip(np.floor(y1 * sy), 0, heat_h - 1))
        hy2 = int(np.clip(np.ceil(y2 * sy), hy1 + 1, heat_h))

        patch = heatmap[hy1:hy2, hx1:hx2]
        if patch.size == 0:
            scores[i] = 0.0
        else:
            scores[i] = float(patch.mean())

    return scores


def filter_boxes(
    boxes: np.ndarray,
    classes: np.ndarray,
    scores: np.ndarray,
    heat_scores: np.ndarray,
    box_threshold: Optional[float],
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """heat_scores와 threshold, top_k 기준으로 bbox 필터링."""
    if boxes.size == 0:
        return boxes, classes, scores, heat_scores

    mask = np.ones(len(boxes), dtype=bool)
    if box_threshold is not None:
        mask &= heat_scores >= box_threshold

    boxes = boxes[mask]
    classes = classes[mask]
    scores = scores[mask]
    heat_scores = heat_scores[mask]

    if top_k > 0 and boxes.shape[0] > top_k:
        order = np.argsort(heat_scores)[-top_k:]  # 상위 top_k만
        boxes = boxes[order]
        classes = classes[order]
        scores = scores[order]
        heat_scores = heat_scores[order]

    return boxes, classes, scores, heat_scores


def main() -> None:
    args = parse_args()
    frames_root = Path(args.frames_root)
    heatmaps_root = Path(args.heatmaps_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.videos:
        videos = args.videos
    elif args.split_file:
        split_path = Path(args.split_file)
        split_ids = []
        with split_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vid = line.split(",")[0]
                split_ids.append(vid)
        videos = split_ids
    elif args.config:
        cfg = yaml.safe_load(Path(args.config).read_text())
        split = cfg.get("testing_split")
        if split is None:
            raise ValueError(f"--config provided but no 'testing_split' key found in {args.config}")
        split_path = Path(split)
        split_ids = []
        with split_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vid = line.split(",")[0]
                split_ids.append(vid)
        videos = split_ids
    else:
        videos = sorted([p.name for p in frames_root.iterdir() if p.is_dir()])

    # 프레임 게이트값: 명시 값 > percentile 순
    frame_gate = args.frame_gate_value
    if frame_gate is None and args.frame_gate_percentile is not None:
        max_vals = collect_heatmap_maxima(heatmaps_root, videos)
        if max_vals:
            frame_gate = float(np.percentile(max_vals, args.frame_gate_percentile))
            print(f"[gate] frame threshold (p{args.frame_gate_percentile}): {frame_gate:.4f}")

    detector = YOLOWorldDetector(
        weights=args.weights,
        classes=args.classes,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )

    total_detected = 0
    total_frames = 0
    skipped_existing = 0
    missing_heatmaps = 0
    empty_frame_dirs = 0

    for vid in tqdm(videos, desc="Videos"):
        video_dir = frames_root / vid
        heat_dir = heatmaps_root / vid
        if not video_dir.exists() or not heat_dir.exists():
            if not heat_dir.exists():
                missing_heatmaps += 1
            continue

        out_vid_dir = output_root / vid
        out_file = out_vid_dir / "detections.npy"
        if out_file.exists() and not args.overwrite:
            skipped_existing += 1
            continue

        frame_paths = iter_frames(video_dir)
        num_frames = len(frame_paths)
        if not frame_paths:
            print(f"[warn] no frames found for video {vid} at {video_dir}")
            empty_frame_dirs += 1
            continue

        boxes_seq: List[np.ndarray] = []
        scores_seq: List[np.ndarray] = []
        classes_seq: List[np.ndarray] = []
        heat_scores_seq: List[np.ndarray] = []
        processed_flags: List[bool] = []

        for frame_path in frame_paths:
            total_frames += 1
            err_path = heat_dir / f"{frame_path.stem}_err.npy"

            if not err_path.exists():
                boxes_seq.append(np.zeros((0, 4), dtype=np.float32))
                scores_seq.append(np.zeros((0,), dtype=np.float32))
                classes_seq.append(np.zeros((0,), dtype=np.int32))
                heat_scores_seq.append(np.zeros((0,), dtype=np.float32))
                processed_flags.append(False)
                continue

            heatmap = np.load(err_path).astype(np.float32)
            frame_bgr = cv2.imread(str(frame_path))

            if frame_bgr is None:
                boxes_seq.append(np.zeros((0, 4), dtype=np.float32))
                scores_seq.append(np.zeros((0,), dtype=np.float32))
                classes_seq.append(np.zeros((0,), dtype=np.int32))
                heat_scores_seq.append(np.zeros((0,), dtype=np.float32))
                processed_flags.append(False)
                continue

            heat_max = float(heatmap.max())
            if frame_gate is not None and heat_max < frame_gate:
                boxes_seq.append(np.zeros((0, 4), dtype=np.float32))
                scores_seq.append(np.zeros((0,), dtype=np.float32))
                classes_seq.append(np.zeros((0,), dtype=np.int32))
                heat_scores_seq.append(np.zeros((0,), dtype=np.float32))
                processed_flags.append(False)
                continue

            # YOLO-World detection
            boxes, classes, scores = detector.detect_image(frame_bgr)
            if boxes.size == 0:
                boxes_seq.append(boxes)
                scores_seq.append(scores)
                classes_seq.append(classes)
                heat_scores_seq.append(np.zeros((0,), dtype=np.float32))
                processed_flags.append(True)
                continue

            # 박스마다 heat score 계산
            heat_scores = box_heat_scores(boxes, heatmap, frame_bgr.shape[:2])

            # box threshold: 명시 값 > percentile 순
            if args.box_heat_threshold is not None:
                box_thr = args.box_heat_threshold
            elif args.box_heat_percentile is not None:
                box_thr = float(np.percentile(heatmap, args.box_heat_percentile))
            else:
                box_thr = None

            boxes, classes, scores, heat_scores = filter_boxes(
                boxes,
                classes,
                scores,
                heat_scores,
                box_thr,
                args.top_k,
            )

            boxes_seq.append(boxes)
            scores_seq.append(scores)
            classes_seq.append(classes)
            heat_scores_seq.append(heat_scores)
            processed_flags.append(True)
            total_detected += boxes.shape[0]

        out_vid_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "video": vid,
            "frame_indices": np.arange(num_frames, dtype=np.int32),
            "frame_files": np.array([p.name for p in frame_paths], dtype=object),
            "boxes": np.array(boxes_seq, dtype=object),
            "scores": np.array(scores_seq, dtype=object),
            "classes": np.array(classes_seq, dtype=object),
            "heat_scores": np.array(heat_scores_seq, dtype=object),
            "processed": np.array(processed_flags, dtype=bool),
            "num_frames": int(num_frames),
        }
        np.save(out_file, payload, allow_pickle=True)

    print(f"[yolo-world] boxes kept: {total_detected}, frames visited: {total_frames}")
    if skipped_existing:
        print(f"[info] skipped {skipped_existing} videos because detections.npy already exists (use --overwrite to recompute)")
    if missing_heatmaps:
        print(f"[info] skipped {missing_heatmaps} videos because heatmaps were missing under {heatmaps_root}")
    if empty_frame_dirs:
        print(f"[info] skipped {empty_frame_dirs} videos because no frame images were found")


if __name__ == "__main__":
    main()
