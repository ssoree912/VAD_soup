#!/usr/bin/env python3
"""Overlay SAM2 masks, detection boxes, ROI scores, and evaluate vs GT (frame/object/pixel)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

# ---------------------------
# I/O helpers
# ---------------------------

def load_detection_payload(path: Path) -> Dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, np.ndarray) and payload.dtype == object:
        return payload.item()
    raise ValueError(f"Unsupported detection payload: {type(payload)}")

def load_frame_dict(path: Optional[Path]) -> Dict[str, np.ndarray]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    payload = np.load(path, allow_pickle=True)
    data = None
    if isinstance(payload, np.lib.npyio.NpzFile):
        if "frame_scores" in payload.files:
            arr = payload["frame_scores"]
            if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
                data = arr.flat[0]
        elif "data" in payload.files:
            arr = payload["data"]
            if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
                maybe_dict = arr.flat[0]
                if isinstance(maybe_dict, dict) and "frame_scores" in maybe_dict:
                    data = maybe_dict["frame_scores"]
    elif isinstance(payload, np.ndarray) and payload.dtype == object:
        data = payload.item()
    if data is None:
        raise ValueError(f"Could not parse frame_scores from {path}")
    return data

def load_frame_labels(path: Optional[Path], video: str) -> Optional[np.ndarray]:
    if path is None:
        return None
    if path.is_dir():
        video_path = path / f"{video}.npy"
        if video_path.exists():
            return np.load(video_path)
        return None
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.lib.npyio.NpzFile):
        if video in payload.files:
            return payload[video]
        if "data" in payload.files:
            arr = payload["data"]
            if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
                maybe_dict = arr.flat[0]
                if isinstance(maybe_dict, dict) and video in maybe_dict:
                    return np.asarray(maybe_dict[video])
    if isinstance(payload, np.ndarray) and payload.dtype == object:
        data = payload.item()
        if isinstance(data, dict) and video in data:
            return np.asarray(data[video])
    return None

def load_pixel_masks(path: Optional[Path], video: str) -> Optional[np.ndarray]:
    """Load GT pixel masks (T,H,W) from directory (video.npy) or npz key=video."""
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        p = path / f"{video}.npy"
        if not p.exists():
            return None
        arr = np.load(p, allow_pickle=True)
        return np.asarray(arr > 0, dtype=np.uint8)
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.lib.npyio.NpzFile):
        if video in payload.files:
            arr = np.asarray(payload[video])
            return np.asarray(arr > 0, dtype=np.uint8)
        if "data" in payload.files:
            arr = payload["data"]
            if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
                d = arr.flat[0]
                if isinstance(d, dict) and video in d:
                    arr = np.asarray(d[video])
                    return np.asarray(arr > 0, dtype=np.uint8)
    arr = np.asarray(payload)
    return np.asarray(arr > 0, dtype=np.uint8)

def mask_path_for(frame_name: str, masks_dir: Path) -> Optional[Path]:
    base = Path(frame_name).stem
    candidates = [masks_dir / f"{base}_mask.png", masks_dir / f"{base}_mask.jpg", masks_dir / f"{base}.png"]
    for c in candidates:
        if c.exists():
            return c
    return None

# ---------------------------
# Visualization
# ---------------------------

def overlay_mask(frame: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float) -> np.ndarray:
    overlay = frame.copy()
    mask_bool = mask > 0
    overlay[mask_bool] = ((1 - alpha) * overlay[mask_bool] + alpha * np.array(color, dtype=np.float32)).astype(np.uint8)
    return overlay

def draw_boxes(frame: np.ndarray, boxes: np.ndarray, scores: Optional[np.ndarray], color=(0, 255, 0), thickness=2):
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        if scores is not None and idx < len(scores):
            text = f"{scores[idx]:.2f}"
            cv2.putText(frame, text, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def annotate(frame: np.ndarray, text_lines: List[str]):
    y = 20
    for line in text_lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18

# ---------------------------
# GT pixel mask -> boxes & matching
# ---------------------------

def masks_to_boxes(mask: np.ndarray, min_area: int = 20) -> np.ndarray:
    """binary mask(H,W){0/1}->{N,4} xyxy boxes via connected components."""
    if mask is None or mask.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    mask = np.asarray(mask > 0, dtype=np.uint8)
    mask = np.ascontiguousarray(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area:
            boxes.append([x, y, x + w, y + h])
    return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)

def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between sets of boxes a:[Na,4], b:[Nb,4]."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:,0], a[:,1], a[:,2], a[:,3]
    bx1, by1, bx2, by2 = b[:,0], b[:,1], b[:,2], b[:,3]
    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    area_a = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    area_b = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)

def greedy_match(iou_mat: np.ndarray, thr: float = 0.3):
    """Greedy bipartite matching by IoU; returns TP, FP, FN counts and matched flags."""
    used_a, used_b = set(), set()
    tp = 0
    matches = []
    pairs = [(i, j) for i in range(iou_mat.shape[0]) for j in range(iou_mat.shape[1])]
    pairs.sort(key=lambda ij: -iou_mat[ij[0], ij[1]])
    for i, j in pairs:
        if i in used_a or j in used_b:
            continue
        if iou_mat[i, j] >= thr:
            tp += 1
            used_a.add(i); used_b.add(j)
            matches.append((i, j))
    fp = iou_mat.shape[0] - tp
    fn = iou_mat.shape[1] - tp
    return tp, fp, fn, set(matches)

# ---------------------------
# AP computation at fixed IoU
# ---------------------------

def compute_ap(preds: List[Tuple[float, int, np.ndarray]], gt_boxes_by_frame: Dict[int, np.ndarray], iou_thr: float):
    """
    preds: list of (score, frame_idx, box[4])
    gt_boxes_by_frame: frame_idx -> array[Ng,4]
    Returns: AP, precision[], recall[]
    """
    # sort predictions by descending score
    preds_sorted = sorted(preds, key=lambda x: -x[0])
    tp_flags = np.zeros(len(preds_sorted), dtype=np.float32)
    fp_flags = np.zeros(len(preds_sorted), dtype=np.float32)

    # mark GT as unmatched initially
    matched_gt = {fi: np.zeros(len(gt_boxes_by_frame.get(fi, [])), dtype=bool) for fi in gt_boxes_by_frame}

    total_gt = sum(len(v) for v in gt_boxes_by_frame.values())
    if total_gt == 0 or len(preds_sorted) == 0:
        return 0.0, np.array([0.0]), np.array([0.0])

    for k, (score, fi, box) in enumerate(preds_sorted):
        gt_boxes = gt_boxes_by_frame.get(fi, np.zeros((0, 4), dtype=np.float32))
        if gt_boxes.size == 0:
            fp_flags[k] = 1.0
            continue
        ious = box_iou(box.reshape(1, 4), gt_boxes).reshape(-1)
        j = int(np.argmax(ious)) if ious.size else -1
        if j >= 0 and ious[j] >= iou_thr and not matched_gt[fi][j]:
            tp_flags[k] = 1.0
            matched_gt[fi][j] = True
        else:
            fp_flags[k] = 1.0

    cum_tp = np.cumsum(tp_flags)
    cum_fp = np.cumsum(fp_flags)
    recall = cum_tp / max(total_gt, 1)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    # VOC-style AP (area under precision-recall curve)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap, precision, recall

# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Overlay + Evaluate ROI boxes/masks against GT")
    p.add_argument("--video", required=True, help="Video id, e.g., 01_0014")
    p.add_argument("--frames_root", required=True, help="Root containing <video> frame folder")
    p.add_argument("--masks_root", required=True, help="Predicted masks root (masks/<video>)")
    p.add_argument("--detections", required=True, help="Path to detections.npy for the video")
    p.add_argument("--roi_scores", required=True, help="NPZ with frame_scores dict (video->list[np.ndarray])")
    p.add_argument("--frame_scores", default=None, help="Optional npy/npz with LANP frame scores (video->1D)")
    p.add_argument("--gt_labels", default=None, help="Optional dir/npz with frame-level GT labels (test_frame_mask)")
    p.add_argument("--gt_pixel_masks", default=None, help="Optional dir/npz with pixel GT masks (test_pixel_mask)")
    p.add_argument("--output_dir", required=True, help="Where to save overlays")
    p.add_argument("--alpha", type=float, default=0.5, help="Mask overlay alpha")
    p.add_argument("--max_frames", type=int, default=None, help="Limit frames exported")
    p.add_argument("--video_output", default=None, help="Optional mp4 path to save montage")
    p.add_argument("--roi_tau", type=float, default=0.6, help="ROI score threshold for per-frame TP/FP/FN")
    p.add_argument("--iou_thr", type=float, default=0.3, help="IoU threshold for matching")
    p.add_argument("--min_gt_area", type=int, default=20, help="Min area for GT component to form a box")
    p.add_argument("--eval_json", default=None, help="Optional path to dump evaluation summary JSON")
    return p.parse_args()

# ---------------------------
# Main
# ---------------------------

def main():
    args = parse_args()

    video = args.video
    frames_dir = Path(args.frames_root) / video
    masks_dir = Path(args.masks_root) / video
    output_dir = Path(args.output_dir) / video
    output_dir.mkdir(parents=True, exist_ok=True)

    det = load_detection_payload(Path(args.detections))
    roi_frame_dict = load_frame_dict(Path(args.roi_scores))
    roi_seq = roi_frame_dict.get(video, [])
    if not roi_seq:
        print(f"[warn] ROI frame scores missing for {video}")

    lanp_scores = None
    if args.frame_scores:
        lanp_dict = load_frame_dict(Path(args.frame_scores))
        lanp_scores = lanp_dict.get(video)

    gt_labels = load_frame_labels(Path(args.gt_labels), video) if args.gt_labels else None
    gt_pixel = load_pixel_masks(Path(args.gt_pixel_masks), video) if args.gt_pixel_masks else None
    if gt_pixel is not None:
        gt_pixel = (gt_pixel > 0).astype(np.uint8)  # (T,H,W)

    frame_files = det.get("frame_files")
    frame_indices = det.get("frame_indices")
    boxes_seq = det.get("boxes")
    total = len(frame_indices)

    # for dataset-level detection AP
    gt_boxes_by_frame: Dict[int, np.ndarray] = {}
    preds_all: List[Tuple[float, int, np.ndarray]] = []

    # for dataset-level per-frame counts & pixel metrics
    sum_tp = sum_fp = sum_fn = 0
    pix_TP = pix_FP = pix_FN = 0

    video_writer = None
    if args.video_output:
        args.video_output = str(Path(args.video_output))

    preds_for_frame_auc: List[float] = []
    labels_for_frame_auc: List[int] = []

    for idx in range(total):
        if args.max_frames is not None and idx >= args.max_frames:
            break

        frame_number = int(frame_indices[idx])
        frame_name = str(frame_files[idx])
        frame_path = frames_dir / frame_name
        if not frame_path.exists():
            print(f"[skip] frame missing: {frame_path}")
            continue

        frame = cv2.imread(str(frame_path))
        overlay = frame.copy()

        # predicted mask (binary) for pixel-level viz
        mask = None
        mask_path = mask_path_for(frame_name, masks_dir)
        if mask_path and mask_path.exists():
            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                mask = (m > 0).astype(np.uint8)
                overlay = overlay_mask(overlay, mask, (0, 0, 255), args.alpha)

        # predicted boxes / scores for this frame
        boxes = np.asarray(boxes_seq[idx]) if len(boxes_seq) > idx else np.zeros((0, 4))
        roi_scores_frame = roi_seq[idx] if idx < len(roi_seq) else np.zeros((len(boxes),))
        if roi_scores_frame is None or len(roi_scores_frame) == 0:
            roi_scores_frame = np.zeros((len(boxes),), dtype=np.float32)

        # --------- GT pixel mask -> GT boxes (for this frame)
        gt_boxes = np.zeros((0, 4), dtype=np.float32)
        if gt_pixel is not None:
            # handle 0-based vs 1-based frame indexing
            fi = frame_number
            if fi >= len(gt_pixel) and fi - 1 >= 0 and fi - 1 < len(gt_pixel):
                fi = fi - 1
            if 0 <= fi < len(gt_pixel):
                gt_mask = gt_pixel[fi]
                gt_boxes = masks_to_boxes(gt_mask, min_area=args.min_gt_area)
                gt_boxes_by_frame[fi] = gt_boxes
                # pixel metrics (only if we also have predicted mask)
                if mask is not None and gt_mask is not None:
                    m_pred = (mask > 0)
                    m_gt = (gt_mask > 0)
                    tp = int(np.logical_and(m_pred, m_gt).sum())
                    fp = int(np.logical_and(m_pred, np.logical_not(m_gt)).sum())
                    fn = int(np.logical_and(np.logical_not(m_pred), m_gt).sum())
                    pix_TP += tp; pix_FP += fp; pix_FN += fn

        # --------- dataset-level AP accumulator (no threshold; use all preds)
        for j in range(len(boxes)):
            preds_all.append((float(roi_scores_frame[j]), frame_number, boxes[j].astype(np.float32)))

        # --------- per-frame TP/FP/FN at threshold tau (for overlay text)
        keep = roi_scores_frame >= args.roi_tau
        pred_boxes_thr = boxes[keep] if len(boxes) else boxes
        pred_scores_thr = roi_scores_frame[keep] if len(roi_scores_frame) else roi_scores_frame

        tp = fp = fn = 0
        if gt_boxes.shape[0] == 0:
            fp = len(pred_boxes_thr)
        else:
            I = box_iou(pred_boxes_thr, gt_boxes)
            tp, fp, fn, matched = greedy_match(I, thr=args.iou_thr)

        sum_tp += tp; sum_fp += fp; sum_fn += fn

        # --------- draw: GT red, TP green, FP yellow
        if gt_boxes.shape[0] > 0:
            draw_boxes(overlay, gt_boxes, None, color=(0, 0, 255), thickness=2)

        if len(pred_boxes_thr):
            if gt_boxes.shape[0] > 0:
                I = box_iou(pred_boxes_thr, gt_boxes)
                max_iou = I.max(axis=1) if I.size else np.zeros((len(pred_boxes_thr),))
                ok = max_iou >= args.iou_thr
                draw_boxes(overlay, pred_boxes_thr[ ok], pred_scores_thr[ ok], color=(0, 255, 0), thickness=2)
                draw_boxes(overlay, pred_boxes_thr[~ok], pred_scores_thr[~ok], color=(0, 255, 255), thickness=2)
            else:
                draw_boxes(overlay, pred_boxes_thr, pred_scores_thr, color=(0, 255, 255), thickness=2)

        # --------- frame-level AUC bookkeeping (LANP vs frame GT)
        score_line = f"ROI max: {np.max(roi_scores_frame) if len(roi_scores_frame) else 0:.2f}"
        if lanp_scores is not None and idx < len(lanp_scores):
            score_line += f" | LANP: {lanp_scores[idx]:.2f}"
            preds_for_frame_auc.append(float(lanp_scores[idx]))
            if gt_labels is not None and idx < len(gt_labels):
                labels_for_frame_auc.append(int(gt_labels[idx]))

        text = [
            f"Video: {video}",
            f"Frame: {frame_number}",
            score_line,
            f"Detections: {len(boxes)} | GT:{gt_boxes.shape[0]}  TP:{tp} FP:{fp} FN:{fn}",
        ]
        annotate(overlay, text)

        out_path = output_dir / f"{frame_number:06d}_overlay.jpg"
        cv2.imwrite(str(out_path), overlay)

        if args.video_output:
            if video_writer is None:
                h, w = overlay.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(args.video_output, fourcc, 15, (w, h))
            video_writer.write(overlay)

    if video_writer is not None:
        video_writer.release()
        print(f"[info] Saved video to {args.video_output}")

    # --------- Frame AUC (if provided)
    if preds_for_frame_auc and labels_for_frame_auc and len(set(labels_for_frame_auc)) > 1:
        auc = roc_auc_score(labels_for_frame_auc, preds_for_frame_auc)
        print(f"[frame] ROC-AUC: {auc * 100:.2f}% on {len(labels_for_frame_auc)} frames")
    else:
        print("[frame] Not enough labeled frames to compute ROC-AUC")

    # --------- Detection AP at IoU thr
    # Build GT boxes dict with frame index normalization (0/1-based)
    if gt_pixel is not None:
        # normalize frame indices to be consistent with preds_all (frame_number)
        gt_by_frame_norm: Dict[int, np.ndarray] = {}
        T = len(gt_pixel)
        for fi0, boxes in gt_boxes_by_frame.items():
            # map 0-based fi0 -> 1-based fi1 for consistency with frame_number if needed
            fi1 = fi0
            if fi1 < 1:  # if gt was 0-based, convert to 1-based
                fi1 = fi0 + 1
            gt_by_frame_norm[fi1] = boxes
        ap, prec, rec = compute_ap(preds_all, gt_by_frame_norm, iou_thr=args.iou_thr)
        print(f"[det] AP@IoU{args.iou_thr:.2f}: {ap * 100:.2f}%  (preds={len(preds_all)}, GT={sum(len(v) for v in gt_by_frame_norm.values())})")
        # per-threshold PR not printed; AP covers it

        # per-frame thresholded P/R/F1
        det_prec = (sum_tp / (sum_tp + sum_fp)) if (sum_tp + sum_fp) > 0 else 0.0
        det_rec  = (sum_tp / (sum_tp + sum_fn)) if (sum_tp + sum_fn) > 0 else 0.0
        det_f1   = (2 * det_prec * det_rec / (det_prec + det_rec)) if (det_prec + det_rec) > 0 else 0.0
        print(f"[det] @tau={args.roi_tau:.2f}, IoU>={args.iou_thr:.2f} --> P:{det_prec:.3f}  R:{det_rec:.3f}  F1:{det_f1:.3f}")

        # Pixel-level (global)
        if pix_TP + pix_FP + pix_FN > 0:
            pix_prec = pix_TP / max(pix_TP + pix_FP, 1)
            pix_rec  = pix_TP / max(pix_TP + pix_FN, 1)
            pix_iou  = pix_TP / max(pix_TP + pix_FP + pix_FN, 1)
            pix_f1   = (2 * pix_prec * pix_rec / max(pix_prec + pix_rec, 1e-9))
            print(f"[pixel] IoU:{pix_iou:.3f}  P:{pix_prec:.3f}  R:{pix_rec:.3f}  F1:{pix_f1:.3f}")
        else:
            print("[pixel] No pixel-level overlap computed (missing masks or empty)")

        # optional JSON dump
        if args.eval_json:
            summary = {
                "video": video,
                "frame_auc": (roc_auc_score(labels_for_frame_auc, preds_for_frame_auc) if preds_for_frame_auc and labels_for_frame_auc and len(set(labels_for_frame_auc))>1 else None),
                "det_ap_iou": args.iou_thr,
                "det_ap": ap,
                "det_threshold_tau": args.roi_tau,
                "det_precision_at_tau": det_prec,
                "det_recall_at_tau": det_rec,
                "det_f1_at_tau": det_f1,
                "pixel_iou": (pix_TP / max(pix_TP + pix_FP + pix_FN, 1)) if (pix_TP + pix_FP + pix_FN) > 0 else None,
                "pixel_precision": (pix_TP / max(pix_TP + pix_FP, 1)) if (pix_TP + pix_FP) > 0 else None,
                "pixel_recall": (pix_TP / max(pix_TP + pix_FN, 1)) if (pix_TP + pix_FN) > 0 else None,
                "pixel_f1": ((2 * (pix_TP / max(pix_TP + pix_FP, 1)) * (pix_TP / max(pix_TP + pix_FN, 1)) / max((pix_TP / max(pix_TP + fp, 1)) + (pix_TP / max(pix_TP + pix_FN, 1)), 1e-9)) if (pix_TP + pix_FP + pix_FN) > 0 else None),
            }
            # fix small bug in inline formula (use cached vars)
            if (pix_TP + pix_FP + pix_FN) > 0:
                pix_prec = pix_TP / max(pix_TP + pix_FP, 1)
                pix_rec  = pix_TP / max(pix_TP + pix_FN, 1)
                summary["pixel_f1"] = (2 * pix_prec * pix_rec / max(pix_prec + pix_rec, 1e-9))
            Path(args.eval_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.eval_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    else:
        print("[det] Skipped detection/pixel metrics (no gt_pixel_masks provided)")

if __name__ == "__main__":
    main()
