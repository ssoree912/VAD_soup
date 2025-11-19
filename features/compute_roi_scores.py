#!/usr/bin/env python3
"""Extract ROI features and anomaly scores from gated detections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from lanp import FrameFeatureBackbone
from lanp.train import reduce_roi_scores
from features import roi_feature_vectors
from features.roi_scorer import ROIScorer


def load_detection_payload(path: Path) -> Dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.dtype == object:
        return payload.item()
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unexpected detection container: {type(payload)}")


def load_memory(path: Path, device: torch.device) -> torch.Tensor:
    payload = np.load(path, allow_pickle=True)
    arr = None
    if isinstance(payload, np.lib.npyio.NpzFile):
        for key in ("normal_memory", "memory", "arr_0"):
            if key in payload.files:
                candidate = payload[key]
                if isinstance(candidate, np.ndarray):
                    arr = candidate
                    break
    elif isinstance(payload, np.ndarray):
        arr = payload
    if arr is None:
        raise ValueError(f"Unable to parse memory tensor from {path}")
    memory = torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(device)
    return memory


def resolve_frame_path(video_dir: Path, frame_key: str, frame_idx: int) -> Path:
    candidate = video_dir / frame_key
    if candidate.exists():
        return candidate
    alt = video_dir / f"{frame_idx:06d}.jpg"
    if alt.exists():
        return alt
    png_alt = alt.with_suffix(".png")
    if png_alt.exists():
        return png_alt
    raise FileNotFoundError(f"Frame not found for key={frame_key} idx={frame_idx} in {video_dir}")


def save_features(path: Path, payload: Dict[str, List[np.ndarray]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, payload, allow_pickle=True)


def save_scores(path: Path, frame_scores: Dict[str, List[np.ndarray]], snippet_scores: Dict[str, np.ndarray]):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frame_scores=np.array([frame_scores], dtype=object),
        snippet_scores=np.array([snippet_scores], dtype=object),
    )


def load_calibration_stats(path: Path) -> Tuple[float, float]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
    else:
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.lib.npyio.NpzFile):
            data = {k: float(arr[k]) for k in arr.files if k in {"mean", "std"}}
        else:
            raise ValueError(f"Unsupported calibration file format: {path}")
    if "mean" not in data or "std" not in data:
        raise ValueError(f"Calibration file must contain 'mean' and 'std' (got keys: {list(data.keys())})")
    mean = float(data["mean"])
    std = float(data["std"])
    std = max(std, 1e-6)
    return mean, std


def find_ae_err_path(ae_root: Optional[Path], video: str, frame_file: str, frame_idx: int) -> Optional[Path]:
    if ae_root is None:
        return None
    video_dir = ae_root / video
    if not video_dir.exists():
        return None
    base = Path(frame_file).stem
    candidates = [
        video_dir / f"{base}_err.npy",
        video_dir / f"{frame_idx:06d}_err.npy",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def compute_ae_roi_scores(err_map: np.ndarray, boxes: np.ndarray, orig_h: int, orig_w: int, agg: str) -> np.ndarray:
    if err_map is None or err_map.size == 0 or boxes.size == 0:
        return np.zeros((boxes.shape[0],), dtype=np.float32)
    if err_map.ndim == 3:
        err_map = err_map.squeeze()
    H_ae, W_ae = err_map.shape
    sx = W_ae / max(orig_w, 1)
    sy = H_ae / max(orig_h, 1)
    scores = np.zeros((boxes.shape[0],), dtype=np.float32)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(float)
        xa1 = int(np.clip(x1 * sx, 0, W_ae - 1))
        ya1 = int(np.clip(y1 * sy, 0, H_ae - 1))
        xa2 = int(np.clip(x2 * sx, xa1 + 1, W_ae))
        ya2 = int(np.clip(y2 * sy, ya1 + 1, H_ae))
        patch = err_map[ya1:ya2, xa1:xa2]
        if patch.size == 0:
            scores[i] = 0.0
            continue
        flat = patch.reshape(-1)
        if agg == "p95":
            scores[i] = float(np.percentile(flat, 95.0))
        else:
            scores[i] = float(flat.mean())
    return scores.astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute ROI features and scores")
    parser.add_argument("--detections_root", required=True, help="Path to detection outputs (*/<video>/detections.npy)")
    parser.add_argument("--frames_root", required=True, help="Root directory containing frame folders per video")
    parser.add_argument("--memory_path", required=True, help="Path to saved LANP normal memory npy/npz")
    parser.add_argument("--output_features", required=True, help="Path to save ROI feature dictionary")
    parser.add_argument("--output_scores", required=True, help="Path to save ROI score npz")
    parser.add_argument("--seg_len", type=int, required=True, help="Segment length for snippet reduction")
    parser.add_argument("--pool_size", type=int, nargs=2, default=[7, 7], help="ROIAlign output size (H W)")
    parser.add_argument("--pooling", choices=["avg", "max"], default="avg")
    parser.add_argument("--score_reducer", choices=["max", "mean"], default="max", help="Reducer for snippet scores")
    parser.add_argument("--device", default=None, help="Device for feature backbone/memory")
    parser.add_argument("--backbone_arch", default="resnet50", help="2D backbone architecture (default: resnet50)")
    parser.add_argument("--videos", nargs="*", default=None, help="Optional subset of videos to process")
    parser.add_argument("--summary", type=str, default=None, help="Optional JSON summary output path")
    parser.add_argument("--save_memory_path", type=str, default=None, help="Optional npy path to save collected ROI features as a memory bank")
    parser.add_argument("--memory_max_samples", type=int, default=None, help="If saving memory, subsample to at most this many ROI features")
    parser.add_argument("--scoring_mode", choices=["max", "knn"], default="max", help="Anomaly scoring mode (max cosine distance or k-NN average)")
    parser.add_argument("--knn_k", type=int, default=5, help="k for k-NN scoring")
    parser.add_argument("--score_calibration", type=str, default=None, help="Optional JSON/NPZ file with mean/std for z-score calibration")
    parser.add_argument(
        "--ae_heatmaps_root",
        type=str,
        default=None,
        help="Root directory with AE dense maps (<video>/<frame>_err.npy)",
    )
    parser.add_argument(
        "--ae_roi_agg",
        choices=["mean", "p95"],
        default="mean",
        help="How to pool AE error within a ROI",
    )
    parser.add_argument(
        "--ae_fusion",
        choices=["none", "ae_only", "weighted", "max"],
        default="none",
        help="Fusion strategy between LANP and AE ROI scores",
    )
    parser.add_argument(
        "--ae_weight",
        type=float,
        default=0.5,
        help="Weight for AE score when ae_fusion=weighted (score=(1-w)*LANP + w*AE)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = FrameFeatureBackbone(arch=args.backbone_arch, pretrained=True, device=device)
    memory = load_memory(Path(args.memory_path), device)
    scorer = ROIScorer(memory, mode=args.scoring_mode, knn_k=args.knn_k)
    ae_root = Path(args.ae_heatmaps_root) if args.ae_heatmaps_root else None
    ae_fusion_mode = args.ae_fusion if ae_root is not None else "none"
    if args.ae_fusion != "none" and ae_root is None:
        print("[warn] --ae_fusion enabled but --ae_heatmaps_root missing; disabling AE fusion")
    if ae_fusion_mode == "weighted" and not (0.0 <= args.ae_weight <= 1.0):
        raise ValueError("--ae_weight must be within [0, 1] when ae_fusion=weighted")
    calibration_stats: Optional[Tuple[float, float]] = None
    if args.score_calibration:
        calibration_stats = load_calibration_stats(Path(args.score_calibration))
    feature_dim = int(memory.shape[1]) if memory.ndim == 2 and memory.shape[1] > 0 else 2048
    detections_root = Path(args.detections_root)
    frames_root = Path(args.frames_root)
    det_files = sorted(detections_root.rglob("detections.npy"))
    if args.videos:
        allowed = set(args.videos)
        det_files = [f for f in det_files if f.parent.name in allowed]
    features_payload: Dict[str, List[np.ndarray]] = {}
    frame_scores_map: Dict[str, List[np.ndarray]] = {}
    summary = {}
    pool_size = tuple(args.pool_size)
    memory_samples: List[np.ndarray] = []
    for det_file in tqdm(det_files, desc="ROI videos"):
        video = det_file.parent.name
        payload = load_detection_payload(det_file)
        boxes_seq = payload["boxes"]
        frame_files = payload["frame_files"]
        frame_indices = payload["frame_indices"]
        video_dir = frames_root / video
        num_frames = int(payload.get("num_frames", len(frame_indices)))
        per_frame_features: List[np.ndarray] = [np.zeros((0, feature_dim), dtype=np.float32) for _ in range(num_frames)]
        per_frame_scores: List[np.ndarray] = [np.zeros((0,), dtype=np.float32) for _ in range(num_frames)]
        frame_loop = tqdm(
            range(min(len(frame_indices), num_frames)),
            desc=f"{video} frames",
            leave=False,
        )
        for idx in frame_loop:
            boxes = np.asarray(boxes_seq[idx])
            if boxes.size == 0:
                continue
            abs_idx = int(frame_indices[idx])
            frame_key = str(frame_files[idx])
            frame_path = resolve_frame_path(video_dir, frame_key, abs_idx)
            image_bgr = cv2.imread(str(frame_path))
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            H0, W0 = image_rgb.shape[:2]
            feature_map = backbone.extract_feature_map(image_rgb)
            spatial_scale = feature_map.shape[-1] / max(image_rgb.shape[1], 1)
            torch_boxes = torch.from_numpy(boxes.astype(np.float32))
            pooled = roi_feature_vectors(
                feature_map.unsqueeze(0),
                [torch_boxes],
                output_size=pool_size,
                spatial_scale=spatial_scale,
                pooling=args.pooling,
            )
            feats_np = pooled.cpu().numpy().astype(np.float32)
            per_frame_features[abs_idx] = feats_np
            if args.save_memory_path and feats_np.size:
                memory_samples.append(feats_np)
            roi_scores_t = scorer.score(pooled)
            if calibration_stats is not None:
                mean, std = calibration_stats
                roi_scores_t = (roi_scores_t - mean) / std
            roi_scores_lanp = roi_scores_t.detach().cpu().numpy().astype(np.float32)

            roi_scores_final = roi_scores_lanp
            if ae_root is not None and ae_fusion_mode != "none":
                err_path = find_ae_err_path(ae_root, video, frame_key, abs_idx)
                if err_path is not None and err_path.exists():
                    err_map = np.load(err_path)
                    roi_scores_ae = compute_ae_roi_scores(err_map, boxes, H0, W0, args.ae_roi_agg)
                    if ae_fusion_mode == "ae_only":
                        roi_scores_final = roi_scores_ae
                    elif ae_fusion_mode == "max":
                        roi_scores_final = np.maximum(roi_scores_lanp, roi_scores_ae)
                    elif ae_fusion_mode == "weighted":
                        w = float(args.ae_weight)
                        roi_scores_final = (1.0 - w) * roi_scores_lanp + w * roi_scores_ae

            per_frame_scores[abs_idx] = roi_scores_final
        features_payload[video] = per_frame_features
        frame_scores_map[video] = per_frame_scores
        summary[video] = {
            "frames_with_detections": int(sum(1 for scores in per_frame_scores if len(scores))),
            "total_frames": num_frames,
        }
    snippet_scores = reduce_roi_scores(frame_scores_map, args.seg_len, reducer=args.score_reducer)
    save_features(Path(args.output_features), features_payload)
    save_scores(Path(args.output_scores), frame_scores_map, snippet_scores)
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.save_memory_path and memory_samples:
        memory_matrix = np.concatenate(memory_samples, axis=0)
        if args.memory_max_samples is not None and memory_matrix.shape[0] > args.memory_max_samples:
            rng = np.random.default_rng()
            idx = rng.choice(memory_matrix.shape[0], args.memory_max_samples, replace=False)
            memory_matrix = memory_matrix[idx]
        mem_path = Path(args.save_memory_path)
        mem_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(mem_path, memory_matrix.astype(np.float32))
        print(f"[memory] Saved {memory_matrix.shape[0]} ROI features to {mem_path}")


if __name__ == "__main__":
    main()
