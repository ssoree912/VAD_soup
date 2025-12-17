#!/usr/bin/env python3
"""
Generate pseudo-label scores for UCF-Crime features.

Assumes features are saved per video as <feature_dir>/<video_name><suffix> (e.g., _res.npy)
with shape [T, D] where T is the number of clips and D is the feature dimension.

Outputs a dict saved to npy:
{
  video_name: {
    "pseudo_label_scores": [float] * T,
    "pseudo_labels_binary": [int] * T,
    "score_v": float
  },
  ...
}
"""

import argparse
import os
import sys
from typing import Dict

import numpy as np
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.normprop import normality_propagation


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", required=True, help="Directory containing per-video feature .npy files.")
    ap.add_argument("--split", required=True, help="Split file (video_name,label,frame_count). Order preserved.")
    ap.add_argument("--output", required=True, help="Output npy path for generated pseudo labels.")
    ap.add_argument("--suffix", default="_res.npy", help="Feature filename suffix (default: _res.npy).")
    ap.add_argument("--abn_num", type=int, default=7, help="abn_num parameter for normality_propagation.")
    ap.add_argument("--transpose", action="store_true",
                    help="Transpose features before propagation if normprop expects [D, T].")
    ap.add_argument("--no_norm", action="store_true", help="Disable L2 normalization of features.")
    return ap.parse_args()


def load_feature(path: str, normalize: bool, transpose: bool) -> np.ndarray:
    feats = np.load(path)
    if feats.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got shape {feats.shape} at {path}")
    if not normalize:
        return feats.T if transpose else feats
    eps = 1e-8
    feats_norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + eps)
    return feats_norm.T if transpose else feats_norm


def main():
    args = parse_args()

    with open(args.split, "r") as fh:
        vids = [ln.strip().split(",")[0] for ln in fh if ln.strip()]

    pseudo: Dict[str, Dict[str, object]] = {}
    missing = []

    for vid in tqdm(vids, desc="Generating pseudo"):
        feat_path = os.path.join(args.feature_dir, vid + args.suffix)
        if not os.path.exists(feat_path):
            missing.append(vid)
            continue
        try:
            feats = load_feature(feat_path, normalize=not args.no_norm, transpose=args.transpose)
            Z, pseudo_labels, score_v = normality_propagation(
                feats, abn_num=args.abn_num, is_ucf=True
            )
            pseudo[vid] = {
                "pseudo_label_scores": np.asarray(Z, dtype=float).tolist(),
                "pseudo_labels_binary": np.asarray(pseudo_labels, dtype=int).tolist(),
                "score_v": float(score_v),
            }
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {vid}: {exc}")
            continue

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, pseudo, allow_pickle=True)

    print(f"[summary] saved {len(pseudo)} videos to {args.output}")
    if missing:
        print(f"[warn] missing features for {len(missing)} videos (showing up to 10): {missing[:10]}")


if __name__ == "__main__":
    main()
