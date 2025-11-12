from __future__ import annotations

from typing import Dict

import numpy as np


def _to_float_array(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float32).copy()


def fuse_scores(
    base_scores: Dict[str, np.ndarray],
    roi_scores: Dict[str, np.ndarray],
    method: str = "max",
    alpha: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    Fuse snippet-level frame scores with ROI-based anomaly values.

    Args:
        base_scores: Dict[video_name -> np.ndarray] baseline snippet scores.
        roi_scores: Dict[video_name -> np.ndarray] ROI snippet scores aligned to base.
        method: 'max' or 'weighted'.
        alpha: weight for ROI scores when method == 'weighted'.
    """
    method = (method or "max").lower()
    fused: Dict[str, np.ndarray] = {}
    for video, base in base_scores.items():
        base_arr = _to_float_array(base)
        roi_arr = roi_scores.get(video)
        if roi_arr is None or roi_arr.size == 0:
            fused[video] = base_arr
            continue
        roi_arr = _to_float_array(roi_arr)
        length = min(len(base_arr), len(roi_arr))
        if method == "weighted":
            lam = float(np.clip(alpha, 0.0, 1.0))
            mix = lam * roi_arr[:length] + (1.0 - lam) * base_arr[:length]
            fused_vec = base_arr.copy()
            fused_vec[:length] = mix
        else:
            fused_vec = base_arr.copy()
            fused_vec[:length] = np.maximum(base_arr[:length], roi_arr[:length])
        fused[video] = fused_vec
    return fused

