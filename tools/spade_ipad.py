#!/usr/bin/env python3
"""
Lightweight SPADE-style kNN anomaly scoring for IPAD frames using WideResNet50.

- Extracts ImageNet pretrained features from IPAD frame folders.
- Builds a gallery from training split (label==0 assumed normal).
- Computes kNN distance for each test frame and reports frame-level ROC-AUC.
- Saves per-frame scores to npy and a summary JSON.

This is a simplified, LANP-free pipeline for quick baseline comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import wide_resnet50_2
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SPADE-style kNN scoring on IPAD frames.")
    p.add_argument("--train_split", required=True, help="Training split txt (<rel_path>,label,frame_len).")
    p.add_argument("--test_split", required=True, help="Testing split txt (<rel_path>,label,frame_len).")
    p.add_argument("--frames_root", type=str, default="data/IPAD", help="Root containing scenario folders.")
    p.add_argument("--out_dir", type=str, default="results/spade_ipad", help="Directory to save scores/summary.")
    p.add_argument("--top_k", type=int, default=5, help="k for kNN distance averaging.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default=None, help="cuda or cpu (auto if omitted).")
    p.add_argument("--knn_chunk", type=int, default=256, help="Chunk size for cdist to limit peak memory.")
    p.add_argument("--torch_num_threads", type=int, default=None, help="Optional cap for torch.set_num_threads.")
    p.add_argument("--load_train_feat", type=str, default=None, help="Optional precomputed train feature npy.")
    p.add_argument("--load_test_feat", type=str, default=None, help="Optional precomputed test feature npy.")
    p.add_argument("--load_test_labels", type=str, default=None, help="Optional precomputed test labels npy.")
    p.add_argument("--save_train_feat", type=str, default=None, help="Path to save computed train feature npy.")
    p.add_argument("--save_test_feat", type=str, default=None, help="Path to save computed test feature npy.")
    p.add_argument("--save_test_labels", type=str, default=None, help="Path to save computed test labels npy.")
    p.add_argument(
        "--stream_features",
        action="store_true",
        help="Stream batch features to disk (under out_dir/temp_*) to reduce RAM; concatenated after extraction.",
    )
    return p.parse_args()


class FrameDataset(Dataset):
    """
    Reads all frames for a list of video paths; yields individual frames.
    items: List of (video_rel_path, frame_path, label_int)
    """

    def __init__(self, frames_root: Path, split_lines: List[str], transform: T.Compose) -> None:
        self.items: List[Tuple[str, Path, int]] = []
        self.transform = transform
        for line in split_lines:
            if not line.strip():
                continue
            rel_path, lbl, frame_len = line.strip().split(",")
            label = int(lbl)
            vdir = frames_root / rel_path
            if not vdir.is_dir():
                continue
            frame_files = sorted(list(vdir.glob("*.jpg")) + list(vdir.glob("*.png")))
            if not frame_files:
                continue
            for fpath in frame_files:
                self.items.append((rel_path, fpath, label))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        rel_path, fpath, label = self.items[idx]
        img = Image.open(fpath).convert("RGB")
        img_t = self.transform(img)
        return img_t, rel_path, label


def build_dataloader(frames_root: Path, split_lines: List[str], batch_size: int, num_workers: int) -> DataLoader:
    transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = FrameDataset(frames_root, split_lines, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader


def extract_features(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[Dict[str, List[np.ndarray]], List[int]]:
    outputs: List[torch.Tensor] = []
    feats: Dict[str, List[np.ndarray]] = OrderedDict(avgpool=[])
    labels: List[int] = []

    def hook(_, __, output):
        outputs.append(output)

    handles = [model.avgpool.register_forward_hook(hook)]

    model.eval()
    with torch.no_grad():
        for imgs, _, lbl in tqdm(loader, desc="feat", total=len(loader)):
            imgs = imgs.to(device, non_blocking=True)
            _ = model(imgs)
            for k, v in zip(feats.keys(), outputs):
                feats[k].append(v.cpu())
            labels.extend(lbl.tolist())
            outputs.clear()

    for k, v in feats.items():
        feats[k] = torch.cat(v, dim=0).numpy()

    for h in handles:
        h.remove()

    return feats, labels


def extract_features_stream(
    model: nn.Module, loader: DataLoader, device: torch.device, temp_dir: Path
) -> Tuple[List[Path], np.ndarray]:
    """Extract only avgpool features and stream batches to temp_dir as npy files.

    Returns list of npy file paths and label array.
    """

    outputs: List[torch.Tensor] = []
    labels: List[int] = []
    temp_dir.mkdir(parents=True, exist_ok=True)

    def hook(_, __, output):
        outputs.append(output)

    handles = [model.avgpool.register_forward_hook(hook)]
    saved_paths: List[Path] = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (imgs, _, lbl) in tqdm(enumerate(loader), desc="feat", total=len(loader)):
            imgs = imgs.to(device, non_blocking=True)
            _ = model(imgs)
            if not outputs:
                continue
            feat_np = outputs[0].cpu().numpy()
            out_path = temp_dir / f"batch_{batch_idx:06d}.npy"
            np.save(out_path, feat_np)
            saved_paths.append(out_path)
            labels.extend(lbl.tolist())
            outputs.clear()

    for h in handles:
        h.remove()

    return saved_paths, np.array(labels, dtype=np.int32)


def pairwise_knn_scores(
    train_feat: np.ndarray, test_feat: np.ndarray, top_k: int, device: torch.device, chunk: int
) -> np.ndarray:
    """
    train_feat: (N_train, D)
    test_feat: (N_test, D)
    """
    train_t = torch.from_numpy(train_feat).to(device)
    test_t = torch.from_numpy(test_feat).to(device)

    scores: List[np.ndarray] = []
    for start in tqdm(range(0, test_t.shape[0], chunk), desc="knn"):
        end = min(start + chunk, test_t.shape[0])
        t_chunk = test_t[start:end]  # (c, D)
        dist = torch.cdist(t_chunk, train_t)  # (c, N_train)
        topk_vals, _ = torch.topk(dist, k=min(top_k, dist.shape[1]), dim=1, largest=False)
        scores.append(topk_vals.mean(dim=1).cpu().numpy())
    return np.concatenate(scores, axis=0)


def flatten_feat(feat: np.ndarray) -> np.ndarray:
    """Ensure feature array is 2D (N, D)."""
    return feat.reshape(feat.shape[0], -1)


def main() -> None:
    args = parse_args()

    if args.torch_num_threads is not None and args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)

    mp.set_sharing_strategy("file_system")
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_lines = Path(args.train_split).read_text().splitlines()
    test_lines = Path(args.test_split).read_text().splitlines()

    # Only use normal videos for gallery (label == 0)
    train_lines_normal = [ln for ln in train_lines if ln.strip().split(",")[1] == "0"]

    frames_root = Path(args.frames_root)

    # Load model
    model = wide_resnet50_2(pretrained=True, progress=True).to(device)

    train_loader = build_dataloader(frames_root, train_lines_normal, args.batch_size, args.num_workers)
    test_loader = build_dataloader(frames_root, test_lines, args.batch_size, args.num_workers)

    if args.load_train_feat and Path(args.load_train_feat).is_file():
        train_feat = flatten_feat(np.load(args.load_train_feat))
    elif args.stream_features:
        temp_train = out_dir / "temp_train_feats"
        train_paths, _ = extract_features_stream(model, train_loader, device, temp_train)
        train_feat = flatten_feat(np.concatenate([np.load(p) for p in tqdm(train_paths, desc="load train feats")], axis=0))
        if args.save_train_feat:
            np.save(args.save_train_feat, train_feat)
    else:
        train_feats_dict, _ = extract_features(model, train_loader, device)
        train_feat = flatten_feat(train_feats_dict["avgpool"])
        if args.save_train_feat:
            np.save(args.save_train_feat, train_feat)

    test_labels: List[int]
    if args.load_test_feat and Path(args.load_test_feat).is_file():
        test_feat = flatten_feat(np.load(args.load_test_feat))
        if args.load_test_labels and Path(args.load_test_labels).is_file():
            test_labels_arr = np.load(args.load_test_labels)
        else:
            raise ValueError("--load_test_labels must be provided when using --load_test_feat.")
    elif args.stream_features:
        temp_test = out_dir / "temp_test_feats"
        test_paths, test_labels_arr = extract_features_stream(model, test_loader, device, temp_test)
        test_feat = flatten_feat(np.concatenate([np.load(p) for p in tqdm(test_paths, desc="load test feats")], axis=0))
        if args.save_test_feat:
            np.save(args.save_test_feat, test_feat)
        if args.save_test_labels:
            np.save(args.save_test_labels, test_labels_arr)
    else:
        test_feats_dict, test_labels = extract_features(model, test_loader, device)
        test_feat = flatten_feat(test_feats_dict["avgpool"])
        test_labels_arr = np.array(test_labels, dtype=np.int32)
        if args.save_test_feat:
            np.save(args.save_test_feat, test_feat)
        if args.save_test_labels:
            np.save(args.save_test_labels, test_labels_arr)

    scores = pairwise_knn_scores(train_feat, test_feat, args.top_k, device, args.knn_chunk)

    # Build frame-level GT labels (video-level label repeated per frame)
    roc_auc = roc_auc_score(test_labels_arr, scores)
    fpr, tpr, _ = roc_curve(test_labels_arr, scores)

    np.save(out_dir / "frame_scores.npy", scores)
    np.save(out_dir / "frame_labels.npy", test_labels_arr)
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(
            {
                "roc_auc": float(roc_auc),
                "top_k": args.top_k,
                "train_split": args.train_split,
                "test_split": args.test_split,
                "num_train_frames": int(train_feat.shape[0]),
                "num_test_frames": int(test_feat.shape[0]),
            },
            fh,
            indent=2,
        )

    print(f"[result] Frame ROC-AUC={roc_auc:.4f} | scores saved to {out_dir/'frame_scores.npy'}")


if __name__ == "__main__":
    main()
