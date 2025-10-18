#!/usr/bin/env python3
"""
Extract features from UCF-Crime videos using the Kenshohara 3D ResNeXt-101 pipeline.
This script matches the preprocessing and model definitions from
https://github.com/kenshohara/video-classification-3d-cnn-pytorch to produce
compatible feature tensors for downstream tasks.
"""

import argparse
import logging
import os
import sys
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


class VideoFeatureExtractor:
    def __init__(self, model_path, device='cuda', repo_root=None, sample_size=112, sample_duration=16):
        self.logger = logging.getLogger(__name__)
        self.device = self._resolve_device(device)
        self.sample_size = sample_size
        self.sample_duration = sample_duration
        self.repo_root = self._resolve_repo_root(repo_root)
        self._import_kenshohara_modules()
        self.model = self._load_model(model_path)
        self.temporal_transform = self.LoopPadding(self.sample_duration)
        self.spatial_transform = self.Compose([
            self.Scale(self.sample_size),
            self.CenterCrop(self.sample_size),
            self.ToTensor(),
            self.Normalize(self.mean, [1, 1, 1]),
        ])

    def _resolve_device(self, device):
        requested = torch.device(device if torch.cuda.is_available() and device.startswith('cuda') else 'cpu')
        if requested.type == 'cuda' and not torch.cuda.is_available():
            logging.warning("CUDA requested but not available. Falling back to CPU.")
        return requested if requested.type == 'cuda' else torch.device('cpu')

    def _resolve_repo_root(self, repo_root):
        default_root = Path(__file__).resolve().parent.parent / 'video-classification-3d-cnn-pytorch'
        resolved_root = Path(repo_root).expanduser().resolve() if repo_root else default_root
        if not resolved_root.exists():
            raise FileNotFoundError(
                f"Kenshohara repository not found at {resolved_root}. "
                "Clone https://github.com/kenshohara/video-classification-3d-cnn-pytorch "
                "or provide --repo_root."
            )
        if str(resolved_root) not in sys.path:
            sys.path.insert(0, str(resolved_root))
        return resolved_root

    def _import_kenshohara_modules(self):
        try:
            from mean import get_mean
            from model import generate_model
            from spatial_transforms import Compose, Normalize, Scale, CenterCrop, ToTensor
            from temporal_transforms import LoopPadding
        except ImportError as exc:
            raise ImportError(
                "Failed to import Kenshohara modules. "
                "Ensure the repository dependencies are installed."
            ) from exc

        self.get_mean = get_mean
        self.generate_model = generate_model
        self.Compose = Compose
        self.Normalize = Normalize
        self.Scale = Scale
        self.CenterCrop = CenterCrop
        self.ToTensor = ToTensor
        self.LoopPadding = LoopPadding
        self.mean = self.get_mean()

    def _build_model_options(self):
        return SimpleNamespace(
            mode='feature',
            model_name='resnext',
            model_depth=101,
            resnet_shortcut='B',
            resnext_cardinality=32,
            wide_resnet_k=2,
            sample_size=self.sample_size,
            sample_duration=self.sample_duration,
            n_classes=400,
            no_cuda=True  # We handle device placement ourselves.
        )

    def _load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        opt = self._build_model_options()
        model = self.generate_model(opt)

        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        state_dict = self._strip_module_prefix(state_dict)

        model.load_state_dict(state_dict)
        model.eval()
        return model.to(self.device)

    @staticmethod
    def _strip_module_prefix(state_dict):
        if any(key.startswith('module.') for key in state_dict.keys()):
            return {key.replace('module.', '', 1): value for key, value in state_dict.items()}
        return state_dict

    def _read_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Failed to open video: {video_path}")
            return []

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        cap.release()
        return frames

    def _prepare_clip(self, frames, indices):
        if len(indices) < self.sample_duration:
            indices = self.temporal_transform(list(indices))
        clip = [self.spatial_transform(frames[idx]) for idx in indices]
        clip_tensor = torch.stack(clip, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
        return clip_tensor.to(self.device)

    def extract_features_from_video(self, video_path, segment_len=16, overlap=8):
        frames = self._read_video_frames(video_path)
        if not frames:
            return None

        if segment_len <= 0:
            raise ValueError("segment_len must be a positive integer.")
        if overlap >= segment_len:
            raise ValueError("overlap must be smaller than segment_len to make progress.")

        features = []
        step_size = segment_len - overlap

        total_frames = len(frames)
        for start_idx in range(0, total_frames, step_size):
            end_idx = min(start_idx + segment_len, total_frames)
            frame_indices = list(range(start_idx, end_idx))
            if len(frame_indices) == 0:
                break

            clip_tensor = self._prepare_clip(frames, frame_indices)

            with torch.no_grad():
                outputs = self.model(clip_tensor)

            clip_features = outputs.squeeze().cpu().numpy()
            features.append(clip_features)

            if end_idx >= total_frames:
                break

        return np.array(features)


def get_video_files(video_dir, extensions=('.mp4', '.avi', '.mov', '.mkv')):
    """Return all video files under a directory."""
    video_files = []
    for ext in extensions:
        video_files.extend(Path(video_dir).glob(f"**/*{ext}"))
    return [str(path) for path in video_files]


def main():
    parser = argparse.ArgumentParser(description='Extract Kinetics-aligned features from UCF-Crime videos.')
    parser.add_argument('--video_dir', required=True, help='Directory containing UCF-Crime videos.')
    parser.add_argument('--output_dir', required=True, help='Directory where extracted features will be stored.')
    parser.add_argument('--model_path', required=True, help='Path to resnext-101-kinetics.pth from the Kenshohara repo.')
    parser.add_argument('--segment_len', type=int, default=16, help='Number of frames per clip.')
    parser.add_argument('--overlap', type=int, default=8, help='Number of overlapping frames between consecutive clips.')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda / cpu).')
    parser.add_argument('--repo_root', default=None, help='Path to the cloned video-classification-3d-cnn-pytorch repository.')

    args = parser.parse_args()

    logger = setup_logging()
    logger.info("Starting UCF-Crime feature extraction.")

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading feature extraction model...")
    try:
        extractor = VideoFeatureExtractor(
            model_path=args.model_path,
            device=args.device,
            repo_root=args.repo_root,
            sample_duration=args.segment_len
        )
        logger.info(f"Model ready on device: {extractor.device}")
    except Exception as exc:
        logger.error(f"Failed to initialise feature extractor: {exc}")
        return 1

    logger.info(f"Scanning for videos in: {args.video_dir}")
    video_files = get_video_files(args.video_dir)
    logger.info(f"Found {len(video_files)} video files.")

    if not video_files:
        logger.error("No video files found. Aborting.")
        return 1

    successful = 0
    failed = 0

    for video_path in tqdm(video_files, desc="Extracting features"):
        video_name = Path(video_path).stem
        output_file = os.path.join(args.output_dir, f"{video_name}_res.npy")

        if os.path.exists(output_file):
            logger.info(f"Features already exist for {video_name}, skipping.")
            successful += 1
            continue

        try:
            features = extractor.extract_features_from_video(
                video_path,
                segment_len=args.segment_len,
                overlap=args.overlap
            )
        except Exception as exc:
            logger.error(f"Error while processing {video_path}: {exc}")
            failed += 1
            continue

        if features is not None and len(features) > 0:
            np.save(output_file, features)
            logger.info(f"Saved features for {video_name}: {features.shape}")
            successful += 1
        else:
            logger.error(f"Failed to extract features from {video_path}")
            failed += 1

    logger.info("Feature extraction completed.")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Output directory: {args.output_dir}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
