#!/usr/bin/env python3
"""
Visualize Attn U-Net error maps & detections vs GT pixel masks.

- Frame + error heatmap (JET)
- GT pixel masks (green contour)
- Attn U-Net detection boxes (yellow, with score)
- Optional pixel-level IoU / Precision / Recall / F1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Dict, List

import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


def load_pixel_masks(root: Path, video: str) -> Optional[np.ndarray]:
    """
    Load GT pixel masks for a video.
    Assumes root/<video>.npy with shape (T,H,W) and >0 as foreground.
    """
    path = root / f"{video}.npy"
    if not path.exists():
        print(f"[warn] GT pixel mask not found: {path}")
        return None
    arr = np.load(path, allow_pickle=True)
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return (arr > 0).astype(np.uint8)
    # fallback: squeeze if weird extra dims
    return (np.squeeze(arr) > 0).astype(np.uint8)


def load_detections(path: Path) -> Dict[str, np.ndarray]:
    """
    Load detections.npy produced by run_attn_unet_on_lanp_frames.py
    Expected keys: boxes, scores, classes, frame_files, frame_indices, num_frames
    """
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, np.ndarray) and payload.dtype == object:
        return payload.item()
    raise ValueError(f"Unsupported detections format at {path}: {type(payload)}")


def overlay_heatmap(frame: np.ndarray, err: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Resize err(H,W) to frame size and overlay as JET heatmap.
    """
    h, w = frame.shape[:2]
    err_norm = err.astype(np.float32)
    err_norm -= err_norm.min()
    denom = err_norm.max()
    if denom > 0:
        err_norm /= denom
    err_up = cv2.resize(err_norm, (w, h), interpolation=cv2.INTER_LINEAR)
    err_col = cv2.applyColorMap((err_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 1.0 - alpha, err_col, alpha, 0.0)
    return overlay


def draw_gt_mask(overlay: np.ndarray, gt_mask: np.ndarray, color=(0, 255, 0), thickness: int = 2):
    """
    Draw GT mask contour in green.
    gt_mask: (H,W){0,1}
    """
    mask = (gt_mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, thickness)


def draw_boxes(
    overlay: np.ndarray,
    boxes: np.ndarray,
    scores: Optional[np.ndarray] = None,
    classes: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    color=(0, 255, 255),
    thickness: int = 2,
):
    """
    Draw detection boxes (xyxy) with optional scores and class labels.
    """
    if boxes is None or len(boxes) == 0:
        return
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

        label_parts = []
        if classes is not None and i < len(classes) and class_names is not None:
            cls_id = int(classes[i])
            if 0 <= cls_id < len(class_names):
                label_parts.append(class_names[cls_id])
        if scores is not None and i < len(scores):
            label_parts.append(f"{scores[i]:.2f}")

        if label_parts:
            label = " ".join(label_parts)
            cv2.putText(
                overlay,
                label,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize Attn U-Net error maps & detections vs GT pixel masks."
    )
    p.add_argument("--video", required=True, help="Video id, e.g., 01_0015")
    p.add_argument("--frames_root", required=True, help="Root with testing frames (…/testing/frames)")
    p.add_argument("--errmaps_root", required=True, help="Root with Attn U-Net error maps (…/att_unet_errmaps)")
    p.add_argument(
        "--detections_root",
        required=True,
        help="Root with detections.npy (…/att_unet_dets)",
    )
    p.add_argument(
        "--gt_pixel_masks",
        required=True,
        help="Root with test_pixel_mask .npy files (e.g. data/shanghaitech/testing/test_pixel_mask)",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save overlay images.",
    )
    p.add_argument(
        "--err_percentile",
        type=float,
        default=95.0,
        help="Percentile threshold on error map to form predicted pixel mask (for metrics).",
    )
    p.add_argument("--alpha", type=float, default=0.5, help="Heatmap overlay alpha.")
    p.add_argument(
        "--video_output",
        type=str,
        default=None,
        help="Optional mp4 path to save video montage.",
    )
    p.add_argument(
        "--target_class",
        type=str,
        default="all",
        help="Class name to filter detections. Use 'all' to show all detected classes (default: 'all').",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    video = args.video
    frames_dir = Path(args.frames_root) / video
    err_dir = Path(args.errmaps_root) / video
    det_path = Path(args.detections_root) / video / "detections.npy"
    gt_root = Path(args.gt_pixel_masks)
    out_dir = Path(args.output_dir) / video
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load GT pixel masks
    gt_masks = load_pixel_masks(gt_root, video)

    # Load detections
    det = load_detections(det_path)
    boxes_seq = det["boxes"]
    scores_seq = det["scores"]
    frame_files = det["frame_files"]
    frame_indices = det["frame_indices"]
    num_frames = int(det["num_frames"])

    # Optional class information (for YOLO / YOLO-World detections)
    classes_seq = det.get("classes", None)
    class_names = det.get("class_names", None)

    # Determine target class id for detection counting
    target_class_id = None
    use_class_filter = False
    class_names_list = None

    if classes_seq is not None and class_names is not None:
        class_names_list = list(class_names)
        print(f"[info] Loaded class names: {class_names_list}")

        if args.target_class != "all":
            if args.target_class in class_names_list:
                target_class_id = class_names_list.index(args.target_class)
                use_class_filter = True
                print(f"[info] Filtering to class '{args.target_class}' (id={target_class_id})")
            else:
                print(
                    f"[warn] target_class '{args.target_class}' not found in class_names. "
                    f"Available classes: {class_names_list}. Using all classes instead."
                )
        else:
            print(f"[info] Showing all detected classes (target_class='all')")

    # Initialize per-class frame hit counter
    class_frame_hits = {}
    if class_names_list is not None:
        class_frame_hits = {name: 0 for name in class_names_list}

    # Optional video writer
    writer = None
    if args.video_output:
        args.video_output = str(Path(args.video_output))

    # Pixel metrics accumulators
    pix_TP = pix_FP = pix_FN = 0

    # Count frames where target_class is detected
    frames_with_target = 0
    target_frames = []
    
    # For pixel-level ROC-AUC
    roc_labels: List[np.ndarray] = []
    roc_scores: List[np.ndarray] = []

    for i in tqdm(range(num_frames), desc=f"{video} frames"):
        frame_name = str(frame_files[i])
        frame_idx = int(frame_indices[i])

        frame_path = frames_dir / frame_name
        if not frame_path.exists():
            print(f"[skip] missing frame: {frame_path}")
            continue

        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"[skip] failed to read frame: {frame_path}")
            continue
        overlay = frame.copy()

        # Error map (may not exist for 모든 frame)
        err_path = err_dir / f"{Path(frame_name).stem}_err.npy"
        err = None
        if err_path.exists():
            err = np.load(err_path)
            if err.ndim == 3:
                err = err.squeeze()
            err = np.asarray(err, dtype=np.float32)
            overlay = overlay_heatmap(overlay, err, alpha=args.alpha)

        # GT pixel mask
        gt_mask = None
        if gt_masks is not None and frame_idx < len(gt_masks):
            gt_mask = gt_masks[frame_idx]
            draw_gt_mask(overlay, gt_mask, color=(0, 255, 0), thickness=2)

        # Detections for this frame
        boxes = boxes_seq[i] if i < len(boxes_seq) else np.zeros((0, 4), dtype=np.float32)
        scores = scores_seq[i] if i < len(scores_seq) else np.zeros((0,), dtype=np.float32)
        if boxes is None:
            boxes = np.zeros((0, 4), dtype=np.float32)
        if scores is None:
            scores = np.zeros((0,), dtype=np.float32)

        boxes_arr = np.asarray(boxes, dtype=np.float32)
        scores_arr = np.asarray(scores, dtype=np.float32)
        cls_arr = None

        # Handle class filtering and statistics
        if classes_seq is not None and i < len(classes_seq):
            cls_i = classes_seq[i]
            cls_arr = np.asarray(cls_i, dtype=np.int64) if cls_i is not None else np.zeros((0,), dtype=np.int64)

            if cls_arr.shape[0] != boxes_arr.shape[0]:
                print(
                    f"[warn] frame {i}: classes length {cls_arr.shape[0]} != boxes length {boxes_arr.shape[0]}; "
                    "skipping class-based processing for this frame."
                )
                cls_arr = None
            else:
                # Count which classes appear in this frame
                if cls_arr.size > 0 and class_names_list is not None:
                    present_classes = np.unique(cls_arr)
                    for cid in present_classes:
                        if 0 <= cid < len(class_names_list):
                            class_frame_hits[class_names_list[cid]] += 1

                # Apply class filter if specified
                if use_class_filter and target_class_id is not None:
                    mask_target = cls_arr == target_class_id
                    boxes_arr = boxes_arr[mask_target]
                    if scores_arr.shape[0] == cls_arr.shape[0]:
                        scores_arr = scores_arr[mask_target]
                    if cls_arr is not None:
                        cls_arr = cls_arr[mask_target]

                    if mask_target.any():
                        frames_with_target += 1
                        target_frames.append(frame_idx)

        draw_boxes(
            overlay,
            boxes_arr,
            scores_arr,
            classes=cls_arr,
            class_names=class_names_list,
            color=(0, 255, 255),
            thickness=2
        )

        # Pixel-level metrics from err vs GT
        if err is not None and gt_mask is not None:
            # 1) Thresholded mask for IoU / P / R / F1
            thr = np.percentile(err, args.err_percentile)
            pred_mask = (err >= thr).astype(np.uint8)

            # resize to GT size if needed
            H_gt, W_gt = gt_mask.shape
            if pred_mask.shape != gt_mask.shape:
                pred_mask = cv2.resize(
                    pred_mask.astype(np.uint8),
                    (W_gt, H_gt),
                    interpolation=cv2.INTER_NEAREST,
                )

            m_pred = pred_mask > 0
            m_gt = gt_mask > 0

            tp = int(np.logical_and(m_pred, m_gt).sum())
            fp = int(np.logical_and(m_pred, np.logical_not(m_gt)).sum())
            fn = int(np.logical_and(np.logical_not(m_pred), m_gt).sum())
            pix_TP += tp
            pix_FP += fp
            pix_FN += fn

            # 2) Continuous scores for pixel-level ROC-AUC
            #    Resize error map to GT resolution if needed, then flatten.
            if err.shape != gt_mask.shape:
                err_up = cv2.resize(
                    err.astype(np.float32),
                    (W_gt, H_gt),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                err_up = err.astype(np.float32)

            roc_labels.append(m_gt.reshape(-1).astype(np.uint8))
            roc_scores.append(err_up.reshape(-1).astype(np.float32))

        # Annotate some text
        txt1 = f"Video: {video} | Frame: {frame_idx}"
        txt2 = f"Det boxes: {len(boxes)}"
        cv2.putText(overlay, txt1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, txt2, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        out_path = out_dir / f"{frame_idx:06d}_overlay.jpg"
        cv2.imwrite(str(out_path), overlay)

        if args.video_output:
            if writer is None:
                h, w = overlay.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.video_output, fourcc, 15, (w, h))
            writer.write(overlay)

    if writer is not None:
        writer.release()
        print(f"[info] Saved video to {args.video_output}")

    # Print global pixel metrics
    if pix_TP + pix_FP + pix_FN > 0:
        prec = pix_TP / max(pix_TP + pix_FP, 1)
        rec = pix_TP / max(pix_TP + pix_FN, 1)
        iou = pix_TP / max(pix_TP + pix_FP + pix_FN, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        print(f"[pixel] IoU={iou:.3f}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")
    else:
        print("[pixel] No overlap computed (maybe no GT or no err maps).")

    # Pixel-level ROC-AUC over all frames
    if roc_labels and roc_scores:
        y_true = np.concatenate(roc_labels).astype(np.uint8)
        y_score = np.concatenate(roc_scores).astype(np.float32)
        if np.unique(y_true).size > 1:
            auc = roc_auc_score(y_true, y_score)
            print(f"[pixel-ROC] AUC={auc:.4f} (num_pixels={y_true.size})")
        else:
            print("[pixel-ROC] Only one class present in labels; ROC-AUC undefined.")
    else:
        print("[pixel-ROC] No pixel-level scores accumulated (maybe no GT or no err maps).")

    # Summary of per-class detections
    if class_names_list is not None:
        print(f"\n[dets] Per-class frame counts for video '{video}':")
        detected_any = False
        for name, cnt in class_frame_hits.items():
            if cnt > 0:
                print(f"  {name}: {cnt} frames")
                detected_any = True
        if not detected_any:
            print("  (no detections in any class)")

    # Summary of target-class detections (if filtering was applied)
    if use_class_filter and target_class_id is not None:
        print(
            f"\n[dets] Filtered target class '{args.target_class}' detected in "
            f"{frames_with_target} / {num_frames} frames."
        )
        if target_frames:
            print(f"[dets] Frames with '{args.target_class}': {target_frames}")


if __name__ == "__main__":
    main()