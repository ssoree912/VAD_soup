#!/usr/bin/env python3
"""Generate patch-core heatmaps from precomputed memory bank."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from tools.patchcore_backbone import ResNetFeatureExtractor, feat_to_patches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute patch-core heatmaps.")
    parser.add_argument("--frames_root", required=True, help="Test frames root.")
    parser.add_argument("--memory_path", required=True, help="Memory npy path.")
    parser.add_argument("--heatmaps_root", required=True, help="Where to save *_err.npy heatmaps.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_patches", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extractor = ResNetFeatureExtractor(device=args.device).eval()

    memory = np.load(args.memory_path).astype(np.float32)
    mem_t = torch.from_numpy(memory).to(args.device)
    mem_norm = (mem_t ** 2).sum(dim=1)

    frames_root = Path(args.frames_root)
    out_root = Path(args.heatmaps_root)
    out_root.mkdir(parents=True, exist_ok=True)

    video_dirs = sorted([p for p in frames_root.iterdir() if p.is_dir()])

    for vid_dir in tqdm(video_dirs, desc="Videos"):
        vid = vid_dir.name
        frame_paths = sorted(list(vid_dir.glob("*.jpg")) + list(vid_dir.glob("*.png")))
        if not frame_paths:
            continue
        vid_out = out_root / vid
        vid_out.mkdir(parents=True, exist_ok=True)

        for frame_path in tqdm(frame_paths, desc=vid, leave=False):
            img = cv2.imread(str(frame_path))
            if img is None:
                continue
            with torch.no_grad():
                feat = extractor(img)
                C, h, w = feat.shape
                patches = feat_to_patches(feat)
                patches = patches.to(args.device)

                dists = []
                for i in range(0, patches.shape[0], args.batch_patches):
                    q = patches[i : i + args.batch_patches]
                    q_norm = (q ** 2).sum(dim=1, keepdim=True)
                    dist2 = q_norm + mem_norm.unsqueeze(0) - 2 * (q @ mem_t.t())
                    nn_dist, _ = dist2.min(dim=1)
                    dists.append(nn_dist.cpu())
                dists = torch.cat(dists, dim=0)
                err_map = dists.reshape(h, w).cpu().numpy().astype(np.float32)
            np.save(vid_out / f"{frame_path.stem}_err.npy", err_map)


if __name__ == "__main__":
    main()
