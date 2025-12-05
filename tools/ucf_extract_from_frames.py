#!/usr/bin/env python3
"""
Extract 3D CNN features for UCF-Crime when data is stored as frames in
training/ and testing/ folders.

This script expects:
  data_root/
    training/<class>/<video_name>/*.jpg
    testing/<class>/<video_name>/*.jpg
    Anomaly_Train.txt
    Anomaly_Test.txt

It will save features to --feature_out (flat directory) with filenames
  <video_name><feature_suffix>  (e.g., Abuse001_x264_res.npy)
matching the names in the TXT splits.

Defaults use torchvision r3d_18 (Kinetics-400) for convenience; pass
--weights to load a custom checkpoint (e.g., ResNeXt-101) and adjust
--feature_suffix to match your config (default: _res.npy).

Note: If you need ResNeXt-101 features identical to the paper, point
--weights to your local ResNeXt-101 Kinetics checkpoint and replace
the model creation block accordingly.
"""

import argparse
import glob
import os
import sys
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.video import r3d_18

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT_DIR)
RESNEXT_ROOT = os.path.join(PROJECT_ROOT, "video-classification-3d-cnn-pytorch")
if RESNEXT_ROOT not in sys.path:
    sys.path.insert(0, RESNEXT_ROOT)

# Kenshohara ResNeXt-3D implementation
try:
    from models.resnext import resnet101 as resnext101_3d  # type: ignore
except ImportError:
    resnext101_3d = None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Root containing training/ testing/ and Anomaly_*.txt")
    ap.add_argument("--feature_out", required=True, help="Output folder for flattened feature files")
    ap.add_argument("--feature_suffix", default="_res.npy", help="Suffix appended to video name (config uses _res.npy)")
    ap.add_argument("--frames_root", default="frames",
                    help="Relative path under data_root where frames are stored (expects training/ and testing/ inside).")
    ap.add_argument("--train_split", default="Anomaly_Train.txt",
                    help="Train split filename (relative to data_root).")
    ap.add_argument("--test_split", default="Anomaly_Test.txt",
                    help="Test split filename (relative to data_root).")
    ap.add_argument("--train_only", action="store_true", help="Process only train split.")
    ap.add_argument("--test_only", action="store_true", help="Process only test split.")
    ap.add_argument("--clip_len", type=int, default=16, help="Frames per clip")
    ap.add_argument("--stride", type=int, default=8, help="Stride between clips")
    ap.add_argument("--resize", type=int, default=128, help="Resize shorter side before center crop")
    ap.add_argument("--crop", type=int, default=112, help="Center crop size")
    ap.add_argument("--weights", default=None, help="Optional model weights checkpoint (state_dict) to load (ResNeXt-101 or torchvision)")
    ap.add_argument("--backbone", choices=["resnext101", "r3d_18"], default="resnext101",
                    help="Backbone to use for feature extraction")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--pad_short", action="store_true",
                    help="If set, pad short videos to clip_len by repeating last frame instead of skipping.")
    ap.add_argument("--align_pseudo", default=None,
                    help="Path to pseudo label scores npy (dict) to align clip count to len(pseudo_label_scores).")
    ap.add_argument("--num_workers", type=int, default=4, help="Num workers for dataloader-ish loop (unused, kept for parity)")
    return ap.parse_args()


def _load_resnext101(weights: str | None, device: torch.device) -> Tuple[nn.Module, int]:
    if resnext101_3d is None:
        raise ImportError("ResNeXt implementation not found. Ensure video-classification-3d-cnn-pytorch is present.")
    model = resnext101_3d(num_classes=400, sample_size=112, sample_duration=16)
    if weights:
        state = torch.load(weights, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        # Remove any 'module.' prefix
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    # Remove classifier
    feature_dim = model.fc.in_features
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    return model, feature_dim


def _load_r3d18(weights: str | None, device: torch.device) -> Tuple[nn.Module, int]:
    model = r3d_18(weights="KINETICS400_V1" if weights is None else None)
    if weights:
        state = torch.load(weights, map_location="cpu")
        model.load_state_dict(state, strict=False)
    feature_dim = model.fc.in_features
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    return model, feature_dim


def load_model(weights: str | None, device: torch.device, backbone: str) -> Tuple[nn.Module, int]:
    if backbone == "resnext101":
        return _load_resnext101(weights, device)
    return _load_r3d18(weights, device)


def build_transform(resize: int, crop: int):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                             std=[0.22803, 0.22145, 0.216989]),
    ])


def iter_clips(frame_paths: List[str], clip_len: int, stride: int):
    for start in range(0, len(frame_paths) - clip_len + 1, stride):
        yield frame_paths[start:start + clip_len]


def load_clip(paths: List[str], transform) -> torch.Tensor:
    imgs = []
    for p in paths:
        img_bgr = cv2.imread(p)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read frame {p}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        imgs.append(transform(img_rgb))
    clip = torch.stack(imgs, dim=1)  # [3, T, H, W]
    return clip


def collect_needed_video_names(txt_path: str) -> List[str]:
    names = []
    with open(txt_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            video_name = line.strip().split(",")[0]
            names.append(video_name)
    return names


def find_frame_paths(video_name: str, data_root: str) -> List[str]:
    """Return sorted frame paths for a given video_name.

    Supports layouts like:
      - <frames_root>/training/<class>/<video_name>/*.{jpg,png}
      - <frames_root>/training/<class>/<video_name>_*.{jpg,png}
      - same under testing/
      - fallback recursive search for directories named video_name
    """
    frames_root = data_root  # already points to frames_root/training|testing in caller
    candidates = []
    # patterns with nested video folder
    candidates += glob.glob(os.path.join(frames_root, "*", "*", video_name, "*.*"))
    # patterns with video_name prefix in class folder
    candidates += glob.glob(os.path.join(frames_root, "*", "*", f"{video_name}*.*"))

    # If none, walk recursively to find a directory named video_name
    if not candidates:
        for root, dirs, files in os.walk(frames_root):
            if os.path.basename(root) == video_name:
                for fname in files:
                    if fname.lower().endswith((".jpg", ".png")):
                        candidates.append(os.path.join(root, fname))
    frames = [p for p in candidates if p.lower().endswith((".jpg", ".png"))]
    frames = sorted(frames, key=lambda x: (os.path.dirname(x), x))
    if frames:
        return frames

    raise FileNotFoundError(f"Frames for {video_name} not found under {frames_root}")


def main():
    args = parse_args()
    if args.train_only and args.test_only:
        raise ValueError("Choose only one of --train_only or --test_only (or neither to run both).")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    os.makedirs(args.feature_out, exist_ok=True)

    model, feat_dim = load_model(args.weights, device, args.backbone)
    transform = build_transform(args.resize, args.crop)

    pseudo_scores = None
    if args.align_pseudo:
        if not os.path.exists(args.align_pseudo):
            print(f"[warn] align_pseudo file not found: {args.align_pseudo}")
        else:
            pseudo_scores = np.load(args.align_pseudo, allow_pickle=True).item()
            print(f"[info] loaded pseudo scores dict with {len(pseudo_scores)} entries")

    needed = set()
    if not args.test_only:
        train_split_path = os.path.join(args.data_root, args.train_split)
        if os.path.exists(train_split_path):
            needed.update(collect_needed_video_names(train_split_path))
        else:
            print(f"[warn] train split not found: {train_split_path}")
    if not args.train_only:
        test_split_path = os.path.join(args.data_root, args.test_split)
        if os.path.exists(test_split_path):
            needed.update(collect_needed_video_names(test_split_path))
        else:
            print(f"[warn] test split not found: {test_split_path}")

    if not needed:
        print("[warn] no videos found from split files; nothing to do.")
        return

    print(f"[info] total videos listed: {len(needed)}")

    processed = 0
    skipped_exists = 0
    skipped_short = 0
    missing_frames = []

    base_train = os.path.join(args.data_root, args.frames_root, "training")
    base_test = os.path.join(args.data_root, args.frames_root, "testing")

    for vid in sorted(needed):
        out_path = os.path.join(args.feature_out, vid + args.feature_suffix)
        if os.path.exists(out_path):
            skipped_exists += 1
            # optional: print minimal log
            continue

        try:
            # prefer training frames if processing train split, otherwise testing
            frames = None
            if not args.test_only:
                try:
                    frames = find_frame_paths(vid, base_train)
                except FileNotFoundError:
                    frames = None
            if frames is None and not args.train_only:
                frames = find_frame_paths(vid, base_test)
            if frames is None:
                raise FileNotFoundError
        except FileNotFoundError:
            missing_frames.append(vid)
            print(f"[missing] frames not found for {vid}")
            continue
        if len(frames) < args.clip_len:
            if not args.pad_short:
                print(f"[skip-short] {vid}: not enough frames ({len(frames)})")
                skipped_short += 1
                continue
            # pad last frame to reach clip_len
            last = frames[-1]
            pad_count = args.clip_len - len(frames)
            frames = frames + [last] * pad_count

        feats = []
        if pseudo_scores is not None and vid in pseudo_scores:
            zlen = len(pseudo_scores[vid].get("pseudo_label_scores", []))
            if zlen == 0:
                print(f"[warn] {vid}: pseudo_label_scores empty, fallback to stride scan")
            else:
                starts = np.linspace(0, max(len(frames) - args.clip_len, 0), num=zlen, dtype=int)
                for s in starts:
                    chunk = frames[s:s + args.clip_len]
                    if len(chunk) < args.clip_len:
                        chunk = chunk + [chunk[-1]] * (args.clip_len - len(chunk))
                    clip = load_clip(chunk, transform).unsqueeze(0).to(device)
                    with torch.no_grad():
                        feat = model(clip)
                    feats.append(feat.squeeze(0).cpu())

        if not feats:
            for chunk in iter_clips(frames, args.clip_len, args.stride):
                clip = load_clip(chunk, transform).unsqueeze(0).to(device)  # [1,3,T,H,W]
                with torch.no_grad():
                    feat = model(clip)  # [1, C]
                feats.append(feat.squeeze(0).cpu())

        if not feats:
            print(f"[warn] {vid}: no clips extracted")
            continue

        feats = torch.stack(feats, dim=0).numpy()  # [num_clips, C]
        np.save(out_path, feats)
        processed += 1
        print(f"[done] {vid} -> {out_path} shape {feats.shape}")

    if missing_frames:
        print(f"[warn] missing frames for {len(missing_frames)} videos (showing up to 10): {missing_frames[:10]}")

    print(f"Finished feature extraction. processed={processed}, existing={skipped_exists}, short_skipped={skipped_short}")


if __name__ == "__main__":
    main()
