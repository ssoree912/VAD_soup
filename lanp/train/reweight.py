from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def _normalize(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    if max_v - min_v < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - min_v) / (max_v - min_v)).astype(np.float32)


def reduce_roi_scores(
    frame_scores: Dict[str, List[np.ndarray]],
    seg_len: int,
    reducer: str = "max",
) -> Dict[str, np.ndarray]:
    """
    Converts per-frame ROI scores into snippet-level arrays aligned with LANP features.
    """
    reducer = reducer.lower()
    if reducer not in {"max", "mean"}:
        raise ValueError(f"Unsupported reducer '{reducer}'.")
    snippet_scores: Dict[str, np.ndarray] = {}
    for video, frames in frame_scores.items():
        if not frames:
            snippet_scores[video] = np.zeros((0,), dtype=np.float32)
            continue
        frame_vals = np.array(
            [np.max(f) if len(f) else 0.0 for f in frames], dtype=np.float32
        )
        pad = (-len(frame_vals)) % seg_len
        if pad:
            frame_vals = np.pad(frame_vals, (0, pad), constant_values=0.0)
        reshaped = frame_vals.reshape(-1, seg_len)
        if reducer == "max":
            snippets = np.max(reshaped, axis=1)
        else:
            snippets = np.mean(reshaped, axis=1)
        snippet_scores[video] = snippets.astype(np.float32)
    return snippet_scores


def blend_scores(
    global_scores: Dict[str, np.ndarray],
    object_scores: Dict[str, np.ndarray],
    lam: float,
) -> Dict[str, np.ndarray]:
    lam = float(np.clip(lam, 0.0, 1.0))
    blended: Dict[str, np.ndarray] = {}
    for video, g_scores in global_scores.items():
        obj = object_scores.get(video)
        if obj is None or obj.size == 0:
            blended[video] = g_scores.astype(np.float32)
            continue
        length = min(len(g_scores), len(obj))
        g_norm = _normalize(g_scores[:length])
        o_norm = _normalize(obj[:length])
        blended_vec = lam * o_norm + (1.0 - lam) * g_norm
        blended[video] = blended_vec.astype(np.float32)
    return blended


def update_reweight(dataset, blended_scores: Dict[str, np.ndarray]):
    """
    Applies the blended anomaly distances to LANP's per-snippet re-weighting.
    """
    for video_name, info in dataset.video_info_dict.items():
        pseudo = np.asarray(info["pseudo_label"])
        scores = blended_scores.get(video_name)
        if scores is None or scores.size == 0:
            continue
        trimmed = scores[: pseudo.shape[0]]
        reweight = np.exp(-np.abs(trimmed - pseudo)).astype(np.float32)
        info["reweight"] = reweight

