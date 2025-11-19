#!/usr/bin/env python3
"""Run U-Net AE only on LANP high-score frames to produce dense maps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T

from models.unet_ae import UNetAutoencoder


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


def load_ae_model(ckpt_path: Path, device: torch.device, in_ch: int = 3, base_ch: int = 64) -> UNetAutoencoder:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = UNetAutoencoder(in_ch=in_ch, base_ch=base_ch).to(device)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a trained AE over only the top-p% LANP frames and save per-frame error maps."
    )
    parser.add_argument("--frames_root", required=True, help="Test frames root, e.g. data/shanghaitech/testing/frames")
    parser.add_argument("--lanp_scores", required=True, help="LANP frame score npz/npy path")
    parser.add_argument("--ae_ckpt", required=True, help="Trained AE checkpoint")
    parser.add_argument("--output_root", required=True, help="Directory for AE error maps (per video subfolders)")
    parser.add_argument("--top_percent", type=float, default=20.0, help="Top p%% LANP frames per video to process")
    parser.add_argument("--image_size", type=int, default=256, help="AE input resolution")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda)")
    parser.add_argument("--ae_base_channels", type=int, default=64, help="Base channels used when training AE")
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

    lanp_scores = load_frame_scores(Path(args.lanp_scores))
    ae = load_ae_model(Path(args.ae_ckpt), device, base_ch=args.ae_base_channels)

    transform = T.Compose(
        [
            T.Resize((args.image_size, args.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    video_dirs = sorted([p for p in frames_root.iterdir() if p.is_dir()])
    for video_dir in video_dirs:
        vid = video_dir.name
        if vid not in lanp_scores:
            print(f"[warn] missing LANP scores for video {vid}, skipping")
            continue
        scores = np.asarray(lanp_scores[vid], dtype=np.float32).reshape(-1)
        thr = np.percentile(scores, 100.0 - args.top_percent)
        print(f"[{vid}] LANP threshold for top {args.top_percent:.1f}% = {thr:.4f}")

        frame_paths = sorted(list(video_dir.glob("*.jpg")) + list(video_dir.glob("*.png")))
        if not frame_paths:
            print(f"[warn] no frames found for {vid}")
            continue
        if len(frame_paths) != len(scores):
            print(
                f"[warn] frame count ({len(frame_paths)}) != score count ({len(scores)}) for {vid}; "
                "aligning by min length"
            )
        frame_count = min(len(frame_paths), len(scores))

        video_out = out_root / vid
        video_out.mkdir(parents=True, exist_ok=True)

        for idx in range(frame_count):
            if scores[idx] < thr:
                continue
            fp = frame_paths[idx]
            img = Image.open(fp).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                x_hat = ae(x)

            x_denorm = (x * 0.5 + 0.5).clamp(0, 1)
            x_hat_denorm = (x_hat * 0.5 + 0.5).clamp(0, 1)

            err_map = torch.sqrt(((x_denorm - x_hat_denorm) ** 2).sum(dim=1, keepdim=True))
            err = err_map.squeeze().cpu().numpy()

            np.save(video_out / f"{fp.stem}_err.npy", err)

        print(f"[{vid}] saved AE maps to {video_out}")


if __name__ == "__main__":
    main()
