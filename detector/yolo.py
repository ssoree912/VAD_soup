from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


def _names_from_model(model) -> Tuple[Dict[int, str], Dict[str, int]]:
    names = model.names
    if isinstance(names, dict):
        idx2name = {int(k): str(v) for k, v in names.items()}
    else:
        idx2name = {i: str(v) for i, v in enumerate(names)}
    name2idx = {v.lower(): k for k, v in idx2name.items()}
    return idx2name, name2idx


class YOLODetector:
    """
    Thin wrapper around YOLOv5 Torch Hub models. The class is import-friendly and
    can be reused by training/inference scripts without duplicating CLI logic.
    """

    def __init__(
        self,
        model_name: str = "yolov5s",
        weights: Optional[str] = None,
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        device: Optional[str] = "auto",
        imgsz: int = 640,
        filter_classes: Optional[Sequence[str]] = ("person",),
    ):
        if device in (None, "auto"):
            device = 0 if torch.cuda.is_available() else "cpu"
        self.device = device
        if weights:
            self.model = torch.hub.load("ultralytics/yolov5", "custom", path=weights)
        else:
            self.model = torch.hub.load("ultralytics/yolov5:v6.2", model_name, pretrained=True, trust_repo=True)
        self.model.to(self.device)
        self.model.conf = float(conf_threshold)
        self.model.iou = float(iou_threshold)
        self.imgsz = int(imgsz)
        self.idx2name, self.name2idx = _names_from_model(self.model)

        self.filter_indices: Optional[set] = None
        if filter_classes:
            normalized = [c.lower() for c in filter_classes]
            self.filter_indices = {self.name2idx[c] for c in normalized if c in self.name2idx}

    def detect_image(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if image_bgr is None:
            return (np.zeros((0, 4), np.float32), np.zeros((0,), np.int32), np.zeros((0,), np.float32))
        with torch.no_grad():
            results = self.model(image_bgr, size=self.imgsz)
        if results is None or len(results.xyxy) == 0:
            return (np.zeros((0, 4), np.float32), np.zeros((0,), np.int32), np.zeros((0,), np.float32))
        pred = results.xyxy[0].cpu().numpy()
        if pred.size == 0:
            return (np.zeros((0, 4), np.float32), np.zeros((0,), np.int32), np.zeros((0,), np.float32))
        boxes = pred[:, :4].astype(np.float32)
        scores = pred[:, 4].astype(np.float32)
        classes = pred[:, 5].astype(np.int32)
        if self.filter_indices is not None:
            mask = np.isin(classes, np.fromiter(self.filter_indices, dtype=np.int32))
            boxes, scores, classes = boxes[mask], scores[mask], classes[mask]
        return boxes, classes, scores

    def detect_frame(self, frame_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        img = cv2.imread(str(frame_path))
        return self.detect_image(img)

    def process_video_frames(self, frames_dir: Path) -> Dict[int, Dict[str, np.ndarray]]:
        frames_dir = Path(frames_dir)
        frame_files = sorted([f for f in frames_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
        detections: Dict[int, Dict[str, np.ndarray]] = {}
        for idx, frame_file in enumerate(frame_files):
            try:
                frame_idx = int(frame_file.stem)
            except ValueError:
                frame_idx = idx
            boxes, classes, scores = self.detect_frame(frame_file)
            detections[frame_idx] = {"boxes": boxes, "classes": classes, "scores": scores}
        return detections

    def batched_detect(self, frames: Iterable[np.ndarray]):
        for frame in frames:
            yield self.detect_image(frame)

