from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class RobustFilterConfig:
    window: int = 5
    iou_threshold: float = 0.3
    min_hits: int = 1


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
    return float(inter / (area_a + area_b - inter))


def _greedy_match(prev_boxes: np.ndarray, curr_boxes: np.ndarray, iou_thr: float) -> List[int]:
    if prev_boxes.size == 0 or curr_boxes.size == 0:
        return [-1] * len(curr_boxes)
    assigned = [-1] * len(curr_boxes)
    used_prev = set()
    for ci, cbox in enumerate(curr_boxes):
        best_i, best_v = -1, 0.0
        for pi, pbox in enumerate(prev_boxes):
            if pi in used_prev:
                continue
            val = _iou_xyxy(pbox, cbox)
            if val > best_v:
                best_v, best_i = val, pi
        if best_v >= iou_thr:
            assigned[ci] = best_i
            used_prev.add(best_i)
    return assigned


def build_tracks(frames_boxes: Sequence[np.ndarray], iou_thr: float) -> List[List[int]]:
    per_frame_ids: List[List[int]] = []
    prev_map = {}
    next_id = 0
    for t, boxes in enumerate(frames_boxes):
        boxes = np.asarray(boxes).reshape(-1, 4)
        tids = [-1] * len(boxes)
        if t == 0:
            for i in range(len(boxes)):
                tids[i] = next_id
                next_id += 1
            prev_map = {i: tids[i] for i in range(len(boxes))}
            per_frame_ids.append(tids)
            continue
        match = _greedy_match(np.asarray(frames_boxes[t - 1]), boxes, iou_thr)
        new_map = {}
        for i, prev_idx in enumerate(match):
            if prev_idx >= 0 and prev_idx in prev_map:
                tids[i] = prev_map[prev_idx]
            else:
                tids[i] = next_id
                next_id += 1
            new_map[i] = tids[i]
        prev_map = new_map
        per_frame_ids.append(tids)
    return per_frame_ids


def _count_support(
    frames_boxes: Sequence[np.ndarray],
    per_frame_ids: Sequence[Sequence[int]],
    target_tid: int,
    ref_frame: int,
    ref_det: int,
    window: int,
    iou_thr: float,
    direction: int,
) -> int:
    ref_box = frames_boxes[ref_frame][ref_det]
    hits = 0
    for offset in range(1, window + 1):
        idx = ref_frame + direction * offset
        if idx < 0 or idx >= len(frames_boxes):
            break
        tids = per_frame_ids[idx]
        boxes = frames_boxes[idx]
        for det_idx, tid in enumerate(tids):
            if tid == target_tid and _iou_xyxy(boxes[det_idx], ref_box) >= iou_thr:
                hits += 1
                break
    return hits


def robust_filter(
    frames_boxes: Sequence[np.ndarray],
    scores: Sequence[np.ndarray],
    score_threshold: float,
    config: RobustFilterConfig,
) -> List[np.ndarray]:
    per_frame_ids = build_tracks(frames_boxes, config.iou_threshold)
    keep_masks: List[np.ndarray] = []
    for t, boxes in enumerate(frames_boxes):
        boxes = np.asarray(boxes).reshape(-1, 4)
        mask = np.zeros(len(boxes), dtype=bool)
        if len(boxes) == 0:
            keep_masks.append(mask)
            continue
        score_vec = np.asarray(scores[t]).reshape(-1)
        tids = per_frame_ids[t]
        for det_idx, score in enumerate(score_vec):
            if score < score_threshold:
                continue
            tid = tids[det_idx]
            if tid < 0:
                continue
            prev_hits = _count_support(frames_boxes, per_frame_ids, tid, t, det_idx, config.window, config.iou_threshold, -1)
            next_hits = _count_support(frames_boxes, per_frame_ids, tid, t, det_idx, config.window, config.iou_threshold, 1)
            if prev_hits >= config.min_hits or next_hits >= config.min_hits:
                mask[det_idx] = True
        keep_masks.append(mask)
    return keep_masks

