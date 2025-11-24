"""YOLO-World 래퍼 유틸리티.

ultralytics 기반 YOLO-World 모델을 불러와 텍스트 프롬프트를 설정하고
예측 결과를 numpy 포맷으로 반환한다. 라이브러리가 설치되지 않은 경우
명확한 에러 메시지를 던져서 사용자가 pip로 추가 설치할 수 있게 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

DEFAULT_PROMPTS: Tuple[str, ...] = (
    "person",
    "group of people",
    "car",
    "bus",
    "truck",
    "bicycle",
    "motorcycle",
    "bag",
    "box",
    "cart",
    "object",
    "unknown object",
)


def _import_yolo_world():
    """
    YOLO-World 모듈을 불러온다.

    - 우선 yoloworld 패키지를 찾고
    - 없으면 ultralytics 에 포함된 YOLOWorld 클래스를 찾는다.
    """
    try:
        from yoloworld import YOLOWorld  # type: ignore
        return YOLOWorld
    except Exception:
        try:
            from ultralytics import YOLOWorld  # type: ignore
            return YOLOWorld
        except Exception as exc:
            raise ImportError(
                "YOLO-World가 설치되어 있지 않습니다. "
                "pip install yoloworld ultralytics 등으로 모델을 설치해 주세요."
            ) from exc


@dataclass
class YOLOWorldOutputs:
    """YOLO-World 예측 결과를 numpy 포맷으로 묶어서 전달."""

    boxes: np.ndarray  # (N, 4) float32, xyxy (원본 해상도)
    scores: np.ndarray  # (N,) float32
    class_ids: np.ndarray  # (N,) int64, prompts 인덱스
    class_names: List[str]  # len=prompt 개수


class YOLOWorldDetector:
    """텍스트 프롬프트를 설정할 수 있는 YOLO-World 추론기."""

    def __init__(
        self,
        model_name: str = "yoloworld_l",
        prompts: Sequence[str] | None = None,
        device: str | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 200,
        imgsz: int = 640,
    ) -> None:
        YOLOWorldCls = _import_yolo_world()
        self.model = YOLOWorldCls(model_name)
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.imgsz = imgsz

        if device is not None and hasattr(self.model, "to"):
            self.model.to(device)  # type: ignore[attr-defined]

        if prompts is None:
            prompts = list(DEFAULT_PROMPTS)
        self.set_prompts(prompts)

    @staticmethod
    def _clean_prompts(prompts: Iterable[str]) -> List[str]:
        return [p.strip() for p in prompts if isinstance(p, str) and p.strip()]

    def set_prompts(self, prompts: Iterable[str]) -> None:
        """텍스트 클래스 목록을 설정."""
        cleaned = self._clean_prompts(prompts)
        if not cleaned:
            raise ValueError("프롬프트가 비어 있습니다.")
        self.prompts = cleaned
        if hasattr(self.model, "set_classes"):
            self.model.set_classes(cleaned)  # type: ignore[attr-defined]
        else:
            # ultralytics >= 8.2와 호환
            if hasattr(self.model, "set_classes_from_list"):
                self.model.set_classes_from_list(cleaned)  # type: ignore[attr-defined]
            else:
                raise AttributeError("모델에서 클래스를 설정하는 메서드를 찾을 수 없습니다.")

    def predict(self, image) -> YOLOWorldOutputs:
        """
        한 장의 이미지를 받아 YOLO-World 예측을 반환.

        Args:
            image: 경로, PIL.Image, numpy array 등 ultralytics가 허용하는 포맷.
        """
        results = self.model.predict(  # type: ignore[call-arg]
            image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            max_det=self.max_det,
            verbose=False,
        )
        if not results:
            return YOLOWorldOutputs(
                boxes=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
                class_ids=np.zeros((0,), dtype=np.int64),
                class_names=list(self.prompts),
            )

        res = results[0]
        boxes = res.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = res.boxes.conf.detach().cpu().numpy().astype(np.float32)
        class_ids = res.boxes.cls.detach().cpu().numpy().astype(np.int64)

        return YOLOWorldOutputs(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            class_names=list(self.prompts),
        )


def load_prompts(inline: str | None, file_path: str | None) -> List[str]:
    """
    프롬프트 문자열/파일을 읽어 정제된 목록을 반환.

    inline: "person,car" 같이 콤마로 구분된 문자열.
    file_path: 한 줄에 하나씩 프롬프트가 적힌 텍스트 파일 경로.
    둘 다 None이면 DEFAULT_PROMPTS를 반환.
    """
    prompts: List[str] = []
    if inline:
        prompts.extend([p.strip() for p in inline.split(",") if p.strip()])
    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            prompts.extend([line.strip() for line in f if line.strip()])
    if not prompts:
        prompts = list(DEFAULT_PROMPTS)
    return prompts
