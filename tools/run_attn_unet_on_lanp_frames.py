#!/usr/bin/env python3
"""Run Attention U-Net predictor on LANP top frames to get error maps + anomaly boxes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.att_unet_predictor import AttUNetPredictor  # noqa: E402
from tools.ae_heatmap_to_detections import mask_to_boxes_and_scores  # noqa: E402


def load_frame_scores(path: Path) -> Dict[str, np.ndarray]:
    """Load LANP frame scores stored as dict-like npy/npz."""
    payload = np.load(path, allow_pickle=True)
    data = None

    if isinstance(payload, np.lib.npyio.NpzFile):
        if "data" in payload.files:
            arr = payload["data"]
            if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
                maybe_dict = arr.flat[0]
                if isinstance(maybe_dict, dict):
                    data = maybe_dict
        else:
            data = {k: np.asarray(payload[k]) for k in payload.files}
    elif isinstance(payload, np.ndarray) and payload.dtype == object:
        obj = payload.item()
        if isinstance(obj, dict):
            data = obj

    if data is None:
        raise ValueError(f"Unsupported LANP frame score format in {path}")
    return data


def load_predictor(ckpt_path: Path, device: torch.device, t: int, base_ch: int) -> AttUNetPredictor:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = AttUNetPredictor(t=t, base_ch=base_ch).to(device)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Attention U-Net predictor on LANP top-p% frames to produce error maps and optional boxes."
    )
    parser.add_argument("--frames_root", required=True, help="Root with per-video frame folders.")
    parser.add_argument("--lanp_scores", required=True, help="LANP frame score npz/npy path.")
    parser.add_argument("--predictor_ckpt", required=True, help="Trained Attention U-Net checkpoint.")
    parser.add_argument("--output_root", required=True, help="Directory to save error maps (<video>/<frame>_err.npy).")
    parser.add_argument("--t", type=int, default=4, help="Number of context frames used by the predictor.")
    parser.add_argument("--image_size", type=int, default=256, help="Input resolution for the predictor.")
    parser.add_argument("--top_percent", type=float, default=20.0, help="Process top-p%% LANP frames per video.")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda).")
    parser.add_argument("--base_channels", type=int, default=64, help="Base channels used when training the predictor.")

    # detection options (optional)
    parser.add_argument("--detections_root", type=str, default=None,
                        help="If set, save detections.npy under this root using the predicted error maps.")
    parser.add_argument("--err_percentile", type=float, default=95.0, help="Per-frame percentile for thresholding.")
    parser.add_argument("--min_area", type=int, default=30, help="Min blob area on error map.")
    parser.add_argument("--morph_kernel", type=int, default=0, help="Morphology kernel (0 disables morphology).")
    parser.add_argument("--score_agg", choices=["mean", "max"], default="mean", help="Box score aggregation.")
    parser.add_argument("--class_id", type=int, default=0, help="Fixed class id for all detections.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 < args.top_percent <= 100.0):
        raise ValueError("--top_percent must be in (0, 100]")

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    frames_root = Path(args.frames_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    det_root = Path(args.detections_root) if args.detections_root else None
    if det_root:
        det_root.mkdir(parents=True, exist_ok=True)

    lanp_scores = load_frame_scores(Path(args.lanp_scores))
    model = load_predictor(Path(args.predictor_ckpt), device, t=args.t, base_ch=args.base_channels)

    transform = T.Compose(
        [
            T.Resize((args.image_size, args.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    def denorm(x: torch.Tensor) -> torch.Tensor:
        return (x * 0.5 + 0.5).clamp(0, 1)

    video_dirs = sorted([p for p in frames_root.iterdir() if p.is_dir()])
    for video_dir in tqdm(video_dirs, desc="Videos"):
        vid = video_dir.name
        if vid not in lanp_scores:
            print(f"[warn] missing LANP scores for video {vid}, skipping")
            continue
        scores = np.asarray(lanp_scores[vid], dtype=np.float32).reshape(-1)
        frame_paths = sorted(list(video_dir.glob("*.jpg")) + list(video_dir.glob("*.png")))
        if not frame_paths:
            print(f"[warn] no frames found for {vid}")
            continue
        frame_count = min(len(frame_paths), len(scores))
        if frame_count <= args.t:
            print(f"[warn] not enough frames for {vid} (need > {args.t})")
            continue

        thr = np.percentile(scores[:frame_count], 100.0 - args.top_percent)
        top_indices = np.nonzero(scores[:frame_count] >= thr)[0]
        print(f"[{vid}] LANP threshold for top {args.top_percent:.1f}% = {thr:.6f} | candidates={len(top_indices)}")
        if top_indices.size == 0:
            continue

        video_out = out_root / vid
        video_out.mkdir(parents=True, exist_ok=True)

        boxes_seq: List[np.ndarray] = []
        scores_seq: List[np.ndarray] = []
        classes_seq: List[np.ndarray] = []
        if det_root:
            boxes_seq = [np.zeros((0, 4), dtype=np.float32) for _ in range(frame_count)]
            scores_seq = [np.zeros((0,), dtype=np.float32) for _ in range(frame_count)]
            classes_seq = [np.zeros((0,), dtype=np.int64) for _ in range(frame_count)]

        processed = 0
        skipped_prefix = 0
        for idx in tqdm(top_indices, desc=f"{vid} frames", leave=False):
            if idx < args.t:
                skipped_prefix += 1
                continue
            start = idx - args.t
            ctx_paths = frame_paths[start:idx]
            if len(ctx_paths) < args.t or idx >= frame_count:
                continue

            target_path = frame_paths[idx]
            target_img = Image.open(target_path).convert("RGB")
            orig_w, orig_h = target_img.size

            ctx_imgs = [transform(Image.open(p).convert("RGB")) for p in ctx_paths]
            target_tensor = transform(target_img)

            inp = torch.cat(ctx_imgs, dim=0).unsqueeze(0).to(device, non_blocking=True)  # (1,3t,H,W)
            tgt = target_tensor.unsqueeze(0).to(device, non_blocking=True)  # (1,3,H,W)

            with torch.no_grad():
                pred = model(inp)

            pred_denorm = denorm(pred)
            tgt_denorm = denorm(tgt)
            err = ((pred_denorm - tgt_denorm) ** 2).mean(dim=1).squeeze(0).cpu().numpy()

            np.save(video_out / f"{target_path.stem}_err.npy", err)
            processed += 1

            if det_root is not None:
                thr_err = float(np.percentile(err, args.err_percentile))
                boxes, det_scores = mask_to_boxes_and_scores(
                    err,
                    thr=thr_err,
                    min_area=args.min_area,
                    score_agg=args.score_agg,
                    morph_kernel=args.morph_kernel,
                )
                if boxes.shape[0] > 0:
                    sx = orig_w / float(err.shape[1])
                    sy = orig_h / float(err.shape[0])
                    boxes_frame = boxes.copy()
                    boxes_frame[:, [0, 2]] *= sx
                    boxes_frame[:, [1, 3]] *= sy

                    boxes_seq[idx] = boxes_frame
                    scores_seq[idx] = det_scores
                    classes_seq[idx] = np.full((boxes.shape[0],), args.class_id, dtype=np.int64)

        print(f"[{vid}] processed={processed}, skipped_prefix(<t)={skipped_prefix}")

        if det_root is not None and processed > 0:
            out_video_dir = det_root / vid
            out_video_dir.mkdir(parents=True, exist_ok=True)
            frame_files = np.array([p.name for p in frame_paths[:frame_count]], dtype=object)
            frame_indices = np.arange(frame_count, dtype=np.int32)
            payload: Dict[str, np.ndarray] = {
                "boxes": np.array(boxes_seq, dtype=object),
                "scores": np.array(scores_seq, dtype=object),
                "classes": np.array(classes_seq, dtype=object),
                "frame_files": frame_files,
                "frame_indices": frame_indices,
                "num_frames": np.array(frame_count, dtype=np.int32),
            }
            np.save(out_video_dir / "detections.npy", payload, allow_pickle=True)
            print(f"[save] detections -> {out_video_dir / 'detections.npy'}")


if __name__ == "__main__":
    main()
