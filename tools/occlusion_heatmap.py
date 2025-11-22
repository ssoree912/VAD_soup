#!/usr/bin/env python3

"""Generate occlusion-based heatmaps for the anomaly detector.

The script keeps the existing LANP pipeline intact: it reloads a trained
`AD_Model`, perturbs a single video snippet by masking spatial regions inside
the raw frames, re-extracts features for the masked snippet using the same 3D
backbone, and measures how much the anomaly score changes. The accumulated
score deltas become an importance map that is projected back onto the frame as
an RGB heatmap.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lanp.backbone import LANPResNeXtBackbone
from model import AD_Model
from utils import set_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Occlusion/RISE-style localization for LANP anomaly scores."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config used for training.")
    parser.add_argument("--checkpoint", required=True, help="Trained AD_Model checkpoint (.pt).")
    parser.add_argument("--video-name", default=None, help="Video identifier (e.g. 01_0015). Required unless --lanp_scores is provided.")
    parser.add_argument("--video-path", default=None, help="Optional explicit path to frames folder or video file.")
    parser.add_argument("--segment-index", type=int, default=None, help="Temporal snippet index to explain. Required unless --lanp_scores is provided.")
    parser.add_argument("--split", choices=["train", "test"], default="test", help="Dataset split containing the video.")
    parser.add_argument("--backbone-weights", required=True, help="Weights file for the 3D backbone (ResNeXt).")
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=[8],
                        help="One or more grid sizes (number of cells per axis).")
    parser.add_argument("--masking", choices=["grid", "rise"], default="grid", help="Occlusion strategy.")
    parser.add_argument("--num-masks", type=int, default=64, help="Number of random masks for RISE.")
    parser.add_argument("--rise-on-prob", type=float, default=0.5, help="Probability that a cell stays visible in each RISE mask.")
    parser.add_argument("--gaussian-sigma", type=float, default=1.0, help="Sigma for Gaussian smoothing on the upsampled heatmap.")
    parser.add_argument("--output-dir", default="visualizations/heatmaps", help="Directory to store heatmap outputs.")
    parser.add_argument("--detections-root", default=None,
                        help="Optional root containing <video>/detections.npy (YOLO/heatmap-guided format) for box overlays.")
    parser.add_argument("--gt-root", default=None,
                        help="Optional root containing ground-truth boxes (per video).")
    parser.add_argument("--heatmap-thresh", type=float, default=90.0,
                        help="Percentile threshold for converting heatmap to boxes.")
    parser.add_argument("--heatmap-min-area", type=int, default=150,
                        help="Minimum area (pixels) of heatmap-derived boxes.")
    parser.add_argument("--device", default=None, help="Device override (cpu / cuda). Defaults to config/device if available.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for mask sampling.")
    parser.add_argument("--fill-mode", choices=["mean", "zero"], default="mean", help="Pixel fill value inside masked regions.")
    parser.add_argument("--feature-norm", choices=["none", "zscore", "l2"], default=None,
                        help="Feature normalization mode to apply to occluded snippets.")
    parser.add_argument("--feature-stats", default=None,
                        help="Path to .npz file containing feature statistics for normalization.")
    parser.add_argument("--normalize-baseline", action="store_true",
                        help="Apply the chosen feature normalization to baseline features as well.")
    parser.add_argument("--delta-relu", action="store_true",
                        help="Clamp score deltas to max(0, baseline - masked).")
    parser.add_argument("--segment-stride", type=int, default=None,
                        help="Override snippet stride used when mapping index to frames.")
    parser.add_argument("--top-p", type=float, default=0.0,
                        help="Fraction of hottest pixels for Top-p pooling (0 disables).")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="Blend factor for fused score: alpha*TopP + (1-alpha)*baseline.")
    parser.add_argument("--model-name", default="resnext", help="Backbone model family (default: resnext).")
    parser.add_argument("--model-depth", type=int, default=101, help="Backbone depth (default: 101).")
    parser.add_argument("--resnext-cardinality", type=int, default=32, help="ResNeXt cardinality (default: 32).")
    parser.add_argument("--resnet-shortcut", default="B", help="Shortcut type used when training features (default: B).")
    parser.add_argument("--sample-size", type=int, default=112, help="Spatial crop size used for feature extraction.")
    parser.add_argument("--verbose", action="store_true", help="Print intermediate diagnostics.")
    # LANP score-driven batch mode
    parser.add_argument("--lanp_scores", default=None, help="Optional npy/npz dict of LANP frame/snippet scores.")
    parser.add_argument("--top_percent", type=float, default=5.0,
                        help="When --lanp_scores is set, process snippets with scores in the top p%% per video.")
    parser.add_argument("--videos", nargs="*", default=None,
                        help="Optional subset of videos to process when using --lanp_scores.")
    return parser.parse_args()


def load_yaml_config(path: str) -> dict:
    with open(path, "r") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration file {path} did not produce a dictionary.")
    return cfg


def load_detections(root: Path, video: str) -> Optional[dict]:
    """Load per-frame detections saved as npy (heatmap_guided_yoloworld format)."""
    det_path = root / video / "detections.npy"
    if not det_path.exists():
        return None
    arr = np.load(det_path, allow_pickle=True)
    if isinstance(arr, np.lib.npyio.NpzFile):
        data = {k: arr[k] for k in arr.files}
    else:
        data = arr.item() if isinstance(arr, np.ndarray) and arr.dtype == object else None
    if not isinstance(data, dict):
        raise ValueError(f"Unsupported detections format in {det_path}")
    return data


def load_gt_for_video(gt_root: Path, video_name: str) -> Optional[dict]:
    gt_path = gt_root / video_name / "gt_boxes.npy"
    alt_path = gt_root / f"{video_name}.npy"
    if not gt_path.exists():
        gt_path = alt_path
    if not gt_path.exists():
        return None
    arr = np.load(gt_path, allow_pickle=True)
    if isinstance(arr, np.lib.npyio.NpzFile):
        data = {k: arr[k] for k in arr.files}
    elif isinstance(arr, dict):
        data = arr
    elif isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
        maybe = arr.item()
        data = maybe if isinstance(maybe, dict) else None
    else:
        data = None

    if isinstance(data, dict) and "frame_indices" in data and "boxes" in data:
        return data

    masks = np.asarray(arr)
    if masks.ndim >= 3:
        frame_indices = np.arange(masks.shape[0], dtype=int)
        boxes_per_frame = []
        for frame in masks:
            mask = (frame > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            frame_boxes = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w * h <= 0:
                    continue
                frame_boxes.append([x, y, x + w, y + h])
            if frame_boxes:
                boxes_per_frame.append(np.asarray(frame_boxes, dtype=np.float32))
            else:
                boxes_per_frame.append(np.zeros((0, 4), dtype=np.float32))
        return {"frame_indices": frame_indices, "boxes": np.array(boxes_per_frame, dtype=object)}

    return data if data is not None else None


def get_gt_boxes_for_frame(
    gt_cache: dict,
    gt_root: Optional[Path],
    video_name: str,
    frame_index: int,
) -> np.ndarray:
    if gt_root is None:
        return np.zeros((0, 4), dtype=np.float32)
    if video_name not in gt_cache:
        data = load_gt_for_video(gt_root, video_name)
        gt_cache[video_name] = data
    data = gt_cache[video_name]
    if data is None:
        return np.zeros((0, 4), dtype=np.float32)

    frame_indices = np.asarray(data["frame_indices"])
    boxes_all = np.asarray(data["boxes"])
    mask = (frame_indices == frame_index)
    if not mask.any():
        return np.zeros((0, 4), dtype=np.float32)
    selected = boxes_all[mask]
    if boxes_all.dtype == object or selected.dtype == object:
        boxes_list = []
        for entry in selected:
            if entry is None:
                continue
            arr = np.asarray(entry, dtype=np.float32).reshape(-1, 4)
            if arr.size > 0:
                boxes_list.append(arr)
        if not boxes_list:
            return np.zeros((0, 4), dtype=np.float32)
        return np.concatenate(boxes_list, axis=0).astype(np.float32)
    return np.asarray(selected, dtype=np.float32).reshape(-1, 4)


def load_lanp_scores(path: Path) -> dict:
    """Load LANP frame/snippet scores saved as npy/npz (dict-like)."""
    arr = np.load(path, allow_pickle=True)
    data = None
    if isinstance(arr, np.lib.npyio.NpzFile):
        if "data" in arr.files:
            maybe = arr["data"]
            if isinstance(maybe, np.ndarray) and maybe.dtype == object and maybe.size == 1 and isinstance(
                maybe.flat[0], dict
            ):
                data = maybe.flat[0]
        else:
            data = {k: np.asarray(arr[k]) for k in arr.files}
    elif isinstance(arr, np.ndarray) and arr.dtype == object:
        maybe = arr.item()
        if isinstance(maybe, dict):
            data = maybe
    if data is None:
        raise ValueError(f"Unsupported LANP score format in {path}")
    return data


def resolve_device(preferred: str | None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_namespace(cfg: dict) -> SimpleNamespace:
    return SimpleNamespace(**cfg)


def load_video_features(cfg: SimpleNamespace, video_name: str) -> np.ndarray:
    feature_path = Path(cfg.feature_path) / f"{video_name}{cfg.feature_name_end}"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    features = np.load(feature_path)
    if features.ndim == 3:
        features = features.mean(axis=1)
    if features.ndim != 2:
        raise ValueError(f"Unexpected feature shape {features.shape} for {video_name}")

    return features.astype(np.float32, copy=False)


def count_video_segments(cfg: SimpleNamespace, video_name: str) -> int:
    """Lightweight helper to check how many snippets exist for a video."""
    feature_path = Path(cfg.feature_path) / f"{video_name}{cfg.feature_name_end}"
    if not feature_path.exists():
        return 0
    arr = np.load(feature_path, mmap_mode="r")
    if arr.ndim == 3:
        return arr.shape[0]
    if arr.ndim == 2:
        return arr.shape[0]
    return 0


def dataset_roots(cfg: SimpleNamespace, split: str) -> Tuple[Path, Path]:
    dataset_root = Path(cfg.feature_path).resolve().parent
    split_root = dataset_root / ("testing" if split == "test" else "training")
    frames_root = split_root / "frames"
    videos_root = split_root / "videos"
    return frames_root, videos_root


def load_segment_from_frames_dir(frames_dir: Path, start: int, segment_len: int) -> List[np.ndarray]:
    frame_files = sorted(p for p in frames_dir.glob("*.jpg") if p.is_file())
    if not frame_files:
        raise FileNotFoundError(f"No frame images found in {frames_dir}")

    frames: List[np.ndarray] = []
    for offset in range(segment_len):
        idx = min(start + offset, len(frame_files) - 1)
        img = cv2.imread(str(frame_files[idx]), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read frame {frame_files[idx]}")
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames


def load_segment_from_video_file(video_path: Path, start: int, segment_len: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or (start + segment_len)
    frames: List[np.ndarray] = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))
    for offset in range(segment_len):
        ret, frame = cap.read()
        if not ret:
            # Repeat the last available frame if we run out.
            if frames:
                frames.append(frames[-1].copy())
            else:
                raise RuntimeError(f"Video {video_path} ended before reaching snippet start.")
        else:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    if not frames:
        raise RuntimeError(f"Unable to retrieve frames from {video_path}")

    return frames


def load_segment_frames(
    video_name: str,
    segment_index: int,
    segment_len: int,
    frames_root: Path,
    videos_root: Path,
) -> List[np.ndarray]:
    start = segment_index * segment_len
    frames_dir = frames_root / video_name
    if frames_dir.is_dir():
        return load_segment_from_frames_dir(frames_dir, start, segment_len)

    video_path = videos_root / f"{video_name}.avi"
    if video_path.exists():
        return load_segment_from_video_file(video_path, start, segment_len)

    raise FileNotFoundError(
        f"Neither frames ({frames_dir}) nor video file ({video_path}) found."
    )


def resolve_frame_name(frames_root: Path, video_name: str, frame_index: int) -> Optional[str]:
    """Return the filename (not path) for a given frame index in a video folder."""
    frames_dir = frames_root / video_name
    frame_paths = sorted(
        list(frames_dir.glob("*.jpg"))
        + list(frames_dir.glob("*.jpeg"))
        + list(frames_dir.glob("*.png"))
        + list(frames_dir.glob("*.bmp"))
    )
    if not frame_paths:
        return None
    idx = min(max(frame_index, 0), len(frame_paths) - 1)
    return frame_paths[idx].name


def collate_frames(frames: Iterable[np.ndarray]) -> np.ndarray:
    arr = np.stack(frames, axis=0)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def build_mask(
    grid_size: int,
    off_y: int,
    off_x: int,
) -> np.ndarray:
    mask = np.ones((grid_size, grid_size), dtype=np.float32)
    mask[off_y, off_x] = 0.0
    return mask


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask
    resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.float32)


def apply_mask_to_frames(
    frames: np.ndarray,
    mask: np.ndarray,
    fill_rgb: np.ndarray,
) -> List[np.ndarray]:
    h, w = frames.shape[1:3]
    mask_img = resize_mask(mask, w, h)
    masked = []
    for frame in frames:
        frame_f = frame.astype(np.float32)
        masked_frame = frame_f * mask_img[..., None] + fill_rgb[None, None, :] * (1.0 - mask_img[..., None])
        masked.append(masked_frame.astype(np.uint8))
    return masked


BackboneFeatureExtractor = LANPResNeXtBackbone


class FeatureNormalizer:
    def __init__(self, mode: str | None, stats_path: str | None):
        self.mode = (mode or "none").lower()
        self.eps = 1e-6
        self.mean = None
        self.std = None
        if self.mode == "zscore":
            if stats_path is None:
                raise ValueError("Feature normalization 'zscore' requires --feature-stats (.npz with mean/std).")
            stats = np.load(stats_path)
            mean_key = next((k for k in ("mean", "mu", "avg") if k in stats), None)
            std_key = next((k for k in ("std", "sigma", "var") if k in stats), None)
            if mean_key is None or std_key is None:
                raise ValueError(f"Stats file {stats_path} must contain 'mean'/'std' (or 'mu'/'sigma').")
            self.mean = stats[mean_key].astype(np.float32, copy=False)
            self.std = stats[std_key].astype(np.float32, copy=False)
            if self.mean.shape != self.std.shape:
                raise ValueError("Mean and std tensors must share the same shape for z-score normalization.")
        elif self.mode not in ("none", "l2"):
            raise ValueError(f"Unsupported feature-norm mode: {self.mode}")

    def apply(self, feature: np.ndarray) -> np.ndarray:
        if self.mode == "none":
            return feature
        if self.mode == "zscore":
            if self.mean is None or self.std is None:
                raise RuntimeError("Z-score normalizer is not initialized properly.")
            return (feature - self.mean) / (self.std + self.eps)
        if self.mode == "l2":
            norm = np.linalg.norm(feature)
            if norm < self.eps:
                return feature
            return feature / norm
        return feature


def prepare_model(cfg: SimpleNamespace, checkpoint: Path, device: torch.device) -> AD_Model:
    model = AD_Model(cfg.feature_dim, 512, cfg.dropout_rate)
    state = torch.load(str(checkpoint), map_location=device)
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def run_model(model: AD_Model, features: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        scores = model(features.to(device))
    return scores.squeeze(0)


def gaussian_blur(heatmap: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return heatmap
    ksize = max(3, int(math.ceil(sigma * 6)) | 1)
    blurred = cv2.GaussianBlur(heatmap, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    return blurred


def upscale_heatmap(grid_heatmap: np.ndarray, frame_shape: Tuple[int, int], sigma: float) -> np.ndarray:
    height, width = frame_shape
    upsampled = cv2.resize(grid_heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
    upsampled = gaussian_blur(upsampled, sigma)
    upsampled -= upsampled.min()
    max_val = upsampled.max()
    if max_val > 0:
        upsampled /= max_val
    return upsampled


def colorize_heatmap(frame_bgr: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    heat_uint8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame_bgr, 0.5, colored, 0.5, 0)
    return overlay


def heatmap_to_bboxes(
    heatmap: np.ndarray,
    thresh_percentile: float = 90.0,
    min_area: int = 150,
) -> np.ndarray:
    """
    Fused heatmap을 thresholding해서 bounding box 리스트를 만든다.
    return: (N, 4) [x1, y1, x2, y2] float32
    """
    hmap = heatmap.astype(np.float32)
    hmap = hmap - hmap.min()
    max_val = hmap.max()
    if max_val > 0:
        hmap = hmap / max_val

    thr = np.percentile(hmap, thresh_percentile)
    mask = (hmap >= thr).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < min_area:
            continue
        boxes.append([x, y, x + w, y + h])

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    box_a, box_b: [x1, y1, x2, y2]
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = max(0.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(0.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union = area_a + area_b - inter + 1e-6

    return float(inter / union)


def grid_occlusion_heatmap(
    features_tensor: torch.Tensor,
    baseline_score: float,
    segment_index: int,
    frames: np.ndarray,
    extractor: BackboneFeatureExtractor,
    model: AD_Model,
    grid_size: int,
    fill_rgb: np.ndarray,
    device: torch.device,
    feature_normalizer: FeatureNormalizer,
) -> np.ndarray:
    grid_heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)
    for gy in range(grid_size):
        for gx in range(grid_size):
            mask = build_mask(grid_size, gy, gx)
            masked_frames = apply_mask_to_frames(frames, mask, fill_rgb)
            occluded_feature = extractor.extract(masked_frames)
            occluded_feature = feature_normalizer.apply(occluded_feature)
            occluded_tensor = features_tensor.clone()
            occluded_tensor[0, 0, segment_index] = torch.from_numpy(occluded_feature).to(device)
            score = run_model(model, occluded_tensor, device)[segment_index].item()
            grid_heatmap[gy, gx] = baseline_score - score
    return grid_heatmap


def rise_occlusion_heatmap(
    features_tensor: torch.Tensor,
    baseline_score: float,
    segment_index: int,
    frames: np.ndarray,
    extractor: BackboneFeatureExtractor,
    model: AD_Model,
    grid_size: int,
    num_masks: int,
    on_prob: float,
    fill_rgb: np.ndarray,
    device: torch.device,
    rng: np.random.Generator,
    feature_normalizer: FeatureNormalizer,
) -> np.ndarray:
    accum = np.zeros((grid_size, grid_size), dtype=np.float32)

    for _ in range(num_masks):
        mask = (rng.random((grid_size, grid_size)) < on_prob).astype(np.float32)
        masked_frames = apply_mask_to_frames(frames, mask, fill_rgb)
        occluded_feature = extractor.extract(masked_frames)
        occluded_feature = feature_normalizer.apply(occluded_feature)
        occluded_tensor = features_tensor.clone()
        occluded_tensor[0, 0, segment_index] = torch.from_numpy(occluded_feature).to(device)
        score = run_model(model, occluded_tensor, device)[segment_index].item()
        delta = baseline_score - score
        accum += mask * delta

    return accum / max(num_masks * on_prob, 1e-6)


def fuse_heatmaps(heatmaps: List[np.ndarray]) -> np.ndarray:
    if not heatmaps:
        raise ValueError("No heatmaps available for fusion.")
    fused = np.zeros_like(heatmaps[0])
    for heatmap in heatmaps:
        tmp = heatmap - heatmap.min()
        max_val = tmp.max()
        if max_val > 0:
            tmp = tmp / max_val
        fused += tmp
    return fused / len(heatmaps)


def top_p_mean(heatmap: np.ndarray, fraction: float) -> float:
    if fraction <= 0.0:
        return 0.0
    flat = heatmap.flatten()
    k = max(1, int(round(len(flat) * fraction)))
    k = min(k, len(flat))
    top_vals = np.partition(flat, -k)[-k:]
    return float(np.mean(top_vals))


def main() -> None:
    args = parse_args()
    cfg_dict = load_yaml_config(args.config)
    cfg = to_namespace(cfg_dict)

    if args.lanp_scores is None:
        if args.video_name is None or args.segment_index is None:
            raise ValueError("--video-name and --segment-index are required unless --lanp_scores is provided.")
    else:
        if not (0.0 < args.top_percent <= 100.0):
            raise ValueError("--top_percent must be in (0, 100].")

    device = resolve_device(args.device or cfg_dict.get("device"))
    set_seeds(args.seed)

    extractor = BackboneFeatureExtractor(
        weights_path=Path(args.backbone_weights),
        device=device,
        sample_duration=getattr(cfg, "segment_len", 16),
        sample_size=args.sample_size,
        model_name=args.model_name,
        model_depth=args.model_depth,
        resnext_cardinality=args.resnext_cardinality,
        resnet_shortcut=args.resnet_shortcut,
    )

    fill_rgb = extractor.fill_rgb.copy()
    if args.fill_mode == "zero":
        fill_rgb.fill(0.0)

    model = prepare_model(cfg, Path(args.checkpoint), device)
    segment_len = getattr(cfg, "segment_len", 16)
    segment_stride = args.segment_stride or getattr(cfg, "segment_stride", segment_len)

    det_root = Path(args.detections_root) if args.detections_root else None
    det_cache: dict[str, Optional[dict]] = {}
    gt_root = Path(args.gt_root) if args.gt_root else None
    gt_cache: dict[str, Optional[dict]] = {}

    def get_boxes_for_frame(video_name: str, frame_name: Optional[str], frame_index: int) -> Optional[np.ndarray]:
        if det_root is None or frame_name is None:
            return None
        if video_name not in det_cache:
            det_cache[video_name] = load_detections(det_root, video_name)
        det = det_cache.get(video_name)
        if not det:
            return None
        boxes_seq = det.get("boxes", None)
        frame_files = det.get("frame_files", None)
        if boxes_seq is None or frame_files is None:
            return None
        # frame_files is usually an array of filenames (dtype=object)
        try:
            frame_files_list = [str(x) for x in frame_files.tolist()]
        except Exception:  # noqa: BLE001
            frame_files_list = [str(x) for x in frame_files]
        if frame_name in frame_files_list:
            idx = frame_files_list.index(frame_name)
            return boxes_seq[idx]
        # fallback by index if lengths match
        if frame_index < len(frame_files_list) and frame_index < len(boxes_seq):
            return boxes_seq[frame_index]
        return None

    # Unified runner per (video, segment)
    def run_for_segment(video_name: str, segment_index: int) -> None:
        features_np = load_video_features(cfg, video_name)
        norm_mode = args.feature_norm if args.feature_norm is not None else getattr(cfg, "feature_norm", None)
        stats_path = args.feature_stats if args.feature_stats is not None else getattr(cfg, "feature_stats", None)
        feature_normalizer = FeatureNormalizer(norm_mode, stats_path)

        if args.normalize_baseline and feature_normalizer.mode != "none":
            features_np = np.apply_along_axis(feature_normalizer.apply, 1, features_np)

        num_segments = features_np.shape[0]
        if segment_index < 0 or segment_index >= num_segments:
            raise IndexError(
                f"segment-index {segment_index} out of range for video {video_name} "
                f"(available snippets: 0..{num_segments - 1})."
            )

        features_tensor = torch.from_numpy(features_np).unsqueeze(0).unsqueeze(0).to(device)
        start = segment_index * segment_stride
        center_frame_index = start + segment_len // 2

        center_frame_name: Optional[str] = None
        if args.video_path:
            explicit_path = Path(args.video_path)
            if explicit_path.is_dir():
                frames_list = load_segment_from_frames_dir(explicit_path, start, segment_len)
            elif explicit_path.is_file():
                frames_list = load_segment_from_video_file(explicit_path, start, segment_len)
            else:
                raise FileNotFoundError(f"Specified video-path not found: {explicit_path}")
        else:
            frames_root, videos_root = dataset_roots(cfg, args.split)
            frames_list = load_segment_frames(video_name, segment_index, segment_len, frames_root, videos_root)
            center_frame_name = resolve_frame_name(frames_root, video_name, center_frame_index)
        frames_np = collate_frames(frames_list)

        baseline_scores = run_model(model, features_tensor, device)
        baseline_score = baseline_scores[segment_index].item()

        if args.verbose:
            print(f"[info] {video_name} seg {segment_index}: baseline score {baseline_score:.6f}")

        rng = np.random.default_rng(args.seed)
        output_root = Path(args.output_dir)
        dest_dir = output_root / video_name / f"seg{segment_index:03d}"
        dest_dir.mkdir(parents=True, exist_ok=True)

        per_scale_heatmaps: List[Tuple[int, np.ndarray]] = []

        for grid_size in args.grid_sizes:
            if args.masking == "grid":
                grid_heatmap = grid_occlusion_heatmap(
                    features_tensor=features_tensor,
                    baseline_score=baseline_score,
                    segment_index=segment_index,
                    frames=frames_np,
                    extractor=extractor,
                    model=model,
                    grid_size=grid_size,
                    fill_rgb=fill_rgb,
                    device=device,
                    feature_normalizer=feature_normalizer,
                )
            else:
                grid_heatmap = rise_occlusion_heatmap(
                    features_tensor=features_tensor,
                    baseline_score=baseline_score,
                    segment_index=segment_index,
                    frames=frames_np,
                    extractor=extractor,
                    model=model,
                    grid_size=grid_size,
                    num_masks=args.num_masks,
                    on_prob=args.rise_on_prob,
                    fill_rgb=fill_rgb,
                    device=device,
                    rng=rng,
                    feature_normalizer=feature_normalizer,
                )

            if args.delta_relu:
                grid_heatmap = np.maximum(grid_heatmap, 0.0)

            frame_center = frames_np[len(frames_np) // 2]
            heatmap_up = upscale_heatmap(grid_heatmap, frame_center.shape[:2], args.gaussian_sigma)

            per_scale_heatmaps.append((grid_size, heatmap_up))

            scale_base = f"{args.masking}_g{grid_size}"
            np.save(dest_dir / f"{scale_base}_grid.npy", grid_heatmap)
            np.save(dest_dir / f"{scale_base}_heatmap.npy", heatmap_up)
            overlay = colorize_heatmap(cv2.cvtColor(frame_center, cv2.COLOR_RGB2BGR), heatmap_up)
            boxes = get_boxes_for_frame(video_name, center_frame_name, center_frame_index)
            overlay_boxes = overlay.copy()
            if boxes is not None and getattr(boxes, "size", 0) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = [int(round(v)) for v in box]
                    cv2.rectangle(overlay_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(str(dest_dir / f"{scale_base}_overlay.png"), overlay)
            cv2.imwrite(str(dest_dir / f"{scale_base}_overlay_boxes.png"), overlay_boxes)

            if args.verbose:
                max_pos = np.unravel_index(np.argmax(grid_heatmap), grid_heatmap.shape)
                print(f"[info] grid {grid_size}: strongest cell {max_pos} with delta {grid_heatmap[max_pos]:.6f}")

        fused_heatmap = fuse_heatmaps([hm for _, hm in per_scale_heatmaps])
        frame_center = frames_np[len(frames_np) // 2]
        frame_center_bgr = cv2.cvtColor(frame_center, cv2.COLOR_RGB2BGR)
        fused_overlay = colorize_heatmap(frame_center_bgr, fused_heatmap)

        pred_boxes = heatmap_to_bboxes(
            fused_heatmap,
            thresh_percentile=args.heatmap_thresh,
            min_area=args.heatmap_min_area,
        )

        gt_boxes = get_gt_boxes_for_frame(
            gt_cache,
            gt_root,
            video_name,
            center_frame_index,
        )

        fused_overlay_boxes = fused_overlay.copy()

        for box in pred_boxes:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            cv2.rectangle(fused_overlay_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)

        for box in gt_boxes:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            cv2.rectangle(fused_overlay_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

        ious = []
        for pb in pred_boxes:
            best = 0.0
            for gb in gt_boxes:
                best = max(best, box_iou(pb, gb))
            if best > 0:
                ious.append(best)
        mean_iou = np.mean(ious) if ious else 0.0

        if args.verbose:
            print(
                f"[eval] {video_name} seg {segment_index}: "
                f"#pred={len(pred_boxes)}, #gt={len(gt_boxes)}, mean IoU={mean_iou:.3f}"
            )

        fused_base = f"{args.masking}_fused"
        np.save(dest_dir / f"{fused_base}_heatmap.npy", fused_heatmap)
        cv2.imwrite(str(dest_dir / f"{fused_base}_overlay.png"), fused_overlay)
        cv2.imwrite(str(dest_dir / f"{fused_base}_overlay_boxes.png"), fused_overlay_boxes)

        top_p_fraction = max(0.0, min(args.top_p, 1.0))
        top_p_value = top_p_mean(fused_heatmap, top_p_fraction) if top_p_fraction > 0 else 0.0
        fused_score = args.alpha * top_p_value + (1.0 - args.alpha) * baseline_score if top_p_fraction > 0 else baseline_score

        with open(dest_dir / f"{fused_base}_meta.txt", "w") as meta:
            meta.write(f"baseline_score: {baseline_score:.6f}\n")
            meta.write(f"top_p_fraction: {top_p_fraction}\n")
            meta.write(f"top_p_mean: {top_p_value:.6f}\n")
            meta.write(f"fused_score: {fused_score:.6f}\n")
            meta.write(f"grid_sizes: {args.grid_sizes}\n")
            meta.write(f"masking: {args.masking}\n")
            meta.write(f"delta_relu: {args.delta_relu}\n")
            meta.write(f"feature_norm: {feature_normalizer.mode}\n")

    # --------- Single run vs LANP-driven batch ---------
    if args.lanp_scores is None:
        run_for_segment(args.video_name, args.segment_index)
        return

    # LANP-driven batch mode
    score_dict = load_lanp_scores(Path(args.lanp_scores))
    target_videos = args.videos if args.videos else sorted(score_dict.keys())

    for vid in target_videos:
        if vid not in score_dict:
            print(f"[warn] video {vid} not found in LANP scores, skipping")
            continue
        num_segments = count_video_segments(cfg, vid)
        if num_segments <= 0:
            print(f"[warn] no feature file found for {vid}, skipping")
            continue

        scores = np.asarray(score_dict[vid], dtype=np.float32).reshape(-1)
        thr = np.percentile(scores, 100.0 - args.top_percent)
        frame_idxs = np.nonzero(scores >= thr)[0].tolist()
        # Map frame indices to snippet indices using segment_stride, and drop OOR.
        seg_idxs = sorted({int(idx // segment_stride) for idx in frame_idxs if idx >= 0})
        seg_idxs = [i for i in seg_idxs if i < num_segments]
        if not seg_idxs:
            print(f"[warn] no valid segments above threshold for {vid} (num_segments={num_segments})")
            continue
        if args.verbose:
            print(
                f"[info] {vid}: top {args.top_percent:.1f}% threshold={thr:.4f}, "
                f"segments={len(seg_idxs)} (num_segments={num_segments})"
            )
        for seg_idx in seg_idxs:
            try:
                run_for_segment(vid, seg_idx)
            except Exception as e:  # noqa: BLE001
                print(f"[error] failed on {vid} seg {seg_idx}: {e}")
        # optional: separate logs between videos
        if args.verbose:
            print(f"[info] finished {vid}")

if __name__ == "__main__":
    main()
