#!/usr/bin/env python3
"""Visualize YOLO-World detections guided by anomaly heatmaps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize heatmap-guided YOLO-World detections.")
    parser.add_argument("--frames_root", required=True, help="Root with <video> frame folders.")
    parser.add_argument("--heatmaps_root", required=False, help="(Optional) Root with <video>/<frame>_err.npy.")
    parser.add_argument("--detections_root", required=True, help="Root with <video>/detections.npy.")
    parser.add_argument("--video", required=True, help="Video id (folder name) to visualize.")
    parser.add_argument("--output_dir", required=True, help="Where to save visualized frames.")
    parser.add_argument("--draw_heatmap", action="store_true", help="Overlay heatmap as colored mask.")
    parser.add_argument("--alpha_heat", type=float, default=0.4, help="Heatmap overlay alpha.")
    parser.add_argument("--score_min", type=float, default=None, help="Optional min score for drawing boxes.")
    parser.add_argument("--max_frames", type=int, default=None, help="Visualize at most N frames.")
    return parser.parse_args()


def iter_frames(video_dir: Path):
    frames = sorted([p for p in video_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    return frames


def load_heatmap(heat_dir: Optional[Path], frame_stem: str) -> Optional[np.ndarray]:
    if heat_dir is None:
        return None
    err_path = heat_dir / f"{frame_stem}_err.npy"
    if not err_path.exists():
        return None
    return np.load(err_path).astype(np.float32)


def overlay_heatmap(
    image_bgr: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """히트맵을 BGR 이미지 위에 컬러맵으로 overlay."""
    h_img, w_img = image_bgr.shape[:2]
    h_hm, w_hm = heatmap.shape

    # heatmap을 0~255로 정규화
    hm = heatmap.copy()
    if hm.max() > hm.min():
        hm = (hm - hm.min()) / (hm.max() - hm.min())
    hm = (hm * 255).astype(np.uint8)

    # 이미지 크기에 맞게 resize
    hm_resized = cv2.resize(hm, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
    hm_color = cv2.applyColorMap(hm_resized, colormap)

    blended = cv2.addWeighted(image_bgr, 1.0, hm_color, alpha, 0)
    return blended


def main() -> None:
    args = parse_args()

    frames_root = Path(args.frames_root)
    heatmaps_root = Path(args.heatmaps_root) if args.heatmaps_root else None
    detections_root = Path(args.detections_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vid = args.video
    video_dir = frames_root / vid
    if not video_dir.exists():
        raise FileNotFoundError(f"Frames dir not found: {video_dir}")

    det_file = detections_root / vid / "detections.npy"
    if not det_file.exists():
        raise FileNotFoundError(f"detections.npy not found: {det_file}")

    payload = np.load(det_file, allow_pickle=True).item()
    boxes_seq = payload["boxes"]
    scores_seq = payload["scores"]
    classes_seq = payload["classes"]
    heat_scores_seq = payload.get("heat_scores", None)

    heat_dir = None
    if heatmaps_root is not None:
        heat_dir = heatmaps_root / vid

    frame_paths = iter_frames(video_dir)
    max_frames = args.max_frames if args.max_frames is not None else len(frame_paths)

    for idx, frame_path in enumerate(tqdm(frame_paths, desc="Viz frames")):
        if idx >= max_frames:
            break

        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            continue

        frame = frame_bgr.copy()

        # (옵션) heatmap overlay
        heatmap = None
        if args.draw_heatmap and heat_dir is not None:
            heatmap = load_heatmap(heat_dir, frame_path.stem)
            if heatmap is not None:
                frame = overlay_heatmap(frame, heatmap, alpha=args.alpha_heat)

        boxes = boxes_seq[idx]
        scores = scores_seq[idx]
        classes = classes_seq[idx]
        heat_scores = None
        if heat_scores_seq is not None and len(heat_scores_seq) > idx:
            heat_scores = heat_scores_seq[idx]

        # bbox 그리기
        if boxes is not None and boxes.size > 0:
            for j, box in enumerate(boxes):
                x1, y1, x2, y2 = box.astype(int)
                score = scores[j] if j < len(scores) else 0.0
                if args.score_min is not None and score < args.score_min:
                    continue

                # 색은 일단 고정 (원하면 heat_score 기반으로 바꿀 수 있음)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"{int(classes[j])}" if j < len(classes) else "cls?"
                if heat_scores is not None and j < len(heat_scores):
                    label = f"{label} | h:{heat_scores[j]:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        out_path = output_dir / frame_path.name
        cv2.imwrite(str(out_path), frame)

    print(f"[viz] saved frames to {output_dir}")


if __name__ == "__main__":
    main()