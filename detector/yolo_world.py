from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLOWorld as _YOLOWorld
except Exception as exc:  # pragma: no cover
    _YOLOWorld = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _ensure_model():
    if _YOLOWorld is None:
        raise RuntimeError(
            "ultralytics YOLOWorld is not available. Install 'ultralytics>=8.0' to use YOLOWorldDetector'"
        ) from _IMPORT_ERROR


class YOLOWorldDetector:
    """Thin wrapper over Ultralytics YOLO-World API."""

    def __init__(
        self,
        weights: str = "yolov8l-world.pt",
        classes: Optional[Sequence[str]] = None,
        device: Optional[str] = None,
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 640,
    ) -> None:
        _ensure_model()
        self.model = _YOLOWorld(weights)
        if classes:
            self.model.set_classes(list(classes))
        if device:
            self.model.to(device)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)

    def detect_image(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if image_bgr is None:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.float32),
            )
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.model.predict(
            source=image_rgb,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )
        if not results:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.float32),
            )
        res = results[0]
        boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = res.boxes.conf.cpu().numpy().astype(np.float32)
        classes = res.boxes.cls.cpu().numpy().astype(np.int32)
        return boxes, classes, scores

    def detect_frame(self, frame_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = cv2.imread(str(frame_path))
        return self.detect_image(image)

    def batched_detect(self, frames: Iterable[np.ndarray]):
        for frame in frames:
            yield self.detect_image(frame)
