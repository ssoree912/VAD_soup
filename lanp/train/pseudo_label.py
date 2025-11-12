from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def _percentile_threshold(values: np.ndarray, top_ratio: float) -> float:
    if values.size == 0:
        return float("inf")
    ratio = np.clip(top_ratio, 1e-4, 1.0)
    k = max(1, int(np.ceil(values.size * ratio)))
    kth = np.partition(values, values.size - k)[values.size - k]
    return float(kth)


def gate_frames_by_score(frame_scores: Dict[str, np.ndarray], top_percent: float) -> Dict[str, np.ndarray]:
    """
    Returns boolean masks selecting the top-P% frames per video.
    """
    ratio = np.clip(top_percent / 100.0, 0.0, 1.0)
    gated: Dict[str, np.ndarray] = {}
    for video, scores in frame_scores.items():
        arr = np.asarray(scores).reshape(-1)
        if arr.size == 0 or ratio == 0:
            gated[video] = np.zeros_like(arr, dtype=bool)
            continue
        thr = _percentile_threshold(arr, ratio)
        gated[video] = arr >= thr
    return gated


def gate_frames_by_boxes(
    box_scores: Dict[str, List[np.ndarray]],
    top_percent: float,
) -> Dict[str, np.ndarray]:
    """
    Marks frames that contain at least one detection inside the top-q% score range.
    """
    ratio = np.clip(top_percent / 100.0, 0.0, 1.0)
    gated: Dict[str, np.ndarray] = {}
    for video, per_frame_scores in box_scores.items():
        flat = np.concatenate([np.asarray(s).reshape(-1) for s in per_frame_scores if len(s)], axis=0) if per_frame_scores else np.zeros((0,))
        if flat.size == 0 or ratio == 0:
            gated[video] = np.zeros(len(per_frame_scores), dtype=bool)
            continue
        thr = _percentile_threshold(flat, ratio)
        mask = np.zeros(len(per_frame_scores), dtype=bool)
        for idx, scores in enumerate(per_frame_scores):
            if np.any(np.asarray(scores) >= thr):
                mask[idx] = True
        gated[video] = mask
    return gated


def merge_frame_masks(
    base: Dict[str, np.ndarray],
    addon: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    merged: Dict[str, np.ndarray] = {k: v.copy() for k, v in base.items()}
    if addon is None:
        return merged
    for video, mask in addon.items():
        if video not in merged:
            merged[video] = mask.copy()
            continue
        length = min(len(merged[video]), len(mask))
        merged[video][:length] = np.logical_and(merged[video][:length], mask[:length])
    return merged


def frames_to_snippets(mask: np.ndarray, seg_len: int) -> np.ndarray:
    if seg_len <= 0:
        raise ValueError("seg_len must be positive.")
    total_segments = int(np.ceil(len(mask) / seg_len))
    pad = total_segments * seg_len - len(mask)
    padded = np.pad(mask.astype(bool), (0, pad), constant_values=False)
    reshaped = padded.reshape(total_segments, seg_len)
    snippet_mask = np.any(reshaped, axis=1)
    return snippet_mask.astype(bool)


def reinforce_pseudo_labels(dataset, frame_mask: Dict[str, np.ndarray], seg_len: Optional[int] = None):
    """
    Keeps pseudo anomalies only when the gated frame mask marks at least one frame
    inside the snippet.
    """
    segment_len = seg_len if seg_len is not None else getattr(dataset, "seg_len", 1)
    for video_name, info in dataset.video_info_dict.items():
        pseudo = np.asarray(info["pseudo_label"])
        frames = frame_mask.get(video_name)
        if frames is None or not np.any(frames):
            info["pseudo_label"] = np.zeros_like(pseudo)
            continue
        snippet_mask = frames_to_snippets(frames, segment_len)
        snippet_mask = snippet_mask[: pseudo.shape[0]]
        info["pseudo_label"] = np.where(snippet_mask, pseudo, 0).astype(np.float32)
