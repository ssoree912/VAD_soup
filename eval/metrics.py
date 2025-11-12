from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def frame_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    roc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    return {"frame_roc_auc": float(roc), "frame_ap": float(ap)}


def snippet_metrics(snippet_scores: np.ndarray, snippet_labels: np.ndarray) -> Dict[str, float]:
    snippet_scores = np.asarray(snippet_scores).reshape(-1)
    snippet_labels = np.asarray(snippet_labels).reshape(-1)
    roc = roc_auc_score(snippet_labels, snippet_scores)
    ap = average_precision_score(snippet_labels, snippet_scores)
    return {"snippet_roc_auc": float(roc), "snippet_ap": float(ap)}


def _flatten_masks(pred_maps: Sequence[np.ndarray], gt_maps: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    preds = np.concatenate([p.reshape(-1) for p in pred_maps])
    gts = np.concatenate([g.reshape(-1) for g in gt_maps])
    return preds, gts


def pixel_metrics(
    pred_maps: Sequence[np.ndarray],
    gt_maps: Sequence[np.ndarray],
    num_thresholds: int = 200,
) -> Dict[str, float]:
    preds, gts = _flatten_masks(pred_maps, gt_maps)
    auroc = roc_auc_score(gts, preds)
    ap = average_precision_score(gts, preds)

    thresholds = np.linspace(preds.min(), preds.max(), num=num_thresholds)
    pro_vals = []
    fpr_vals = []
    total_anom = float(np.count_nonzero(gts))
    total_norm = float(gts.size - total_anom)
    total_anom = max(total_anom, 1.0)
    total_norm = max(total_norm, 1.0)
    for thr in thresholds:
        pred_mask = preds >= thr
        tp = float(np.sum(pred_mask & (gts > 0)))
        fp = float(np.sum(pred_mask & (gts == 0)))
        pro_vals.append(tp / total_anom)
        fpr_vals.append(fp / total_norm)
    au_pro = float(np.trapz(pro_vals, fpr_vals))
    return {"pixel_auroc": float(auroc), "pixel_ap": float(ap), "pixel_aupro": au_pro}


def pixel_f1_at_threshold(pred_map: np.ndarray, gt_map: np.ndarray, threshold: float) -> float:
    pred_mask = pred_map >= threshold
    tp = np.sum(pred_mask & (gt_map > 0))
    fp = np.sum(pred_mask & (gt_map == 0))
    fn = np.sum(~pred_mask & (gt_map > 0))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def precision_recall_curve_from_maps(pred_maps: Sequence[np.ndarray], gt_maps: Sequence[np.ndarray]):
    preds, gts = _flatten_masks(pred_maps, gt_maps)
    precision, recall, thresh = precision_recall_curve(gts, preds)
    return precision, recall, thresh


def greedy_match(pred_boxes: np.ndarray, gt_boxes: np.ndarray, iou_thr: float) -> Tuple[int, int, int]:
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)
    gt_used = set()
    tp = 0
    for pbox in pred_boxes:
        best_iou = 0.0
        best_idx = -1
        for idx, gbox in enumerate(gt_boxes):
            if idx in gt_used:
                continue
            iou = _iou_xyxy(pbox, gbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_iou >= iou_thr and best_idx >= 0:
            tp += 1
            gt_used.add(best_idx)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


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


def compute_rbdc(
    predictions: Dict[str, Dict[int, np.ndarray]],
    ground_truth: Dict[str, Dict[int, np.ndarray]],
    iou_thr: float = 0.3,
) -> Dict[str, float]:
    tp = fp = fn = 0
    videos = set(predictions.keys()) | set(ground_truth.keys())
    for vid in videos:
        pred_frames = predictions.get(vid, {})
        gt_frames = ground_truth.get(vid, {})
        frames = set(pred_frames.keys()) | set(gt_frames.keys())
        for frame in frames:
            pred_boxes = pred_frames.get(frame, np.zeros((0, 4), dtype=np.float32))
            gt_boxes = gt_frames.get(frame, np.zeros((0, 4), dtype=np.float32))
            a, b, c = greedy_match(pred_boxes, gt_boxes, iou_thr)
            tp += a
            fp += b
            fn += c
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"rbdc_precision": precision, "rbdc_recall": recall}


@dataclass
class Track:
    frames: Sequence[int]
    boxes: Sequence[np.ndarray]


def compute_tbdc(
    predictions: Dict[str, Dict[int, np.ndarray]],
    gt_tracks: Dict[str, List[Track]],
    iou_thr: float = 0.3,
) -> float:
    if not gt_tracks:
        return 0.0
    ratios = []
    for vid, tracks in gt_tracks.items():
        pred_frames = predictions.get(vid, {})
        for track in tracks:
            matched = 0
            total = len(track.frames)
            total = max(total, 1)
            for frame, gt_box in zip(track.frames, track.boxes):
                preds = pred_frames.get(frame, np.zeros((0, 4), dtype=np.float32))
                if preds.size == 0:
                    continue
                for pbox in preds:
                    if _iou_xyxy(pbox, gt_box) >= iou_thr:
                        matched += 1
                        break
            ratios.append(matched / total)
    if not ratios:
        return 0.0
    return float(np.mean(ratios))

