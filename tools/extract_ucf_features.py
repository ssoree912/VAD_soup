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
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


class VideoFeatureExtractor:
    def __init__(self, model_path, device='cuda', repo_root=None, sample_size=112, sample_duration=16,
                 max_segments=None):
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
        self.max_segments = max_segments if (max_segments and max_segments > 0) else None

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

    def _frame_iterator(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Failed to open video: {video_path}")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield Image.fromarray(frame)
        finally:
            cap.release()

    def _assemble_clip(self, frames):
        clip_frames = list(frames)
        if not clip_frames:
            return None

        indices = list(range(len(clip_frames)))
        if len(indices) < self.sample_duration:
            indices = self.temporal_transform(indices)
            clip_frames = [clip_frames[idx] for idx in indices]

        processed = [self.spatial_transform(img) for img in clip_frames[:self.sample_duration]]
        clip_tensor = torch.stack(processed, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
        return clip_tensor.to(self.device)

    def _forward_clip(self, clip_tensor):
        with torch.no_grad():
            outputs = self.model(clip_tensor)
        feature = outputs.squeeze().detach().cpu().float().numpy()
        return feature

    def extract_features_from_video(self, video_path, video_name=None, segment_len=16, overlap=8):
        if segment_len <= 0:
            raise ValueError("segment_len must be a positive integer.")
        if overlap >= segment_len:
            raise ValueError("overlap must be smaller than segment_len to make progress.")

        step_size = segment_len - overlap
        frame_buffer = deque()
        features = []
        segment_count = 0
        video_stem = video_name or Path(video_path).stem

        for frame in self._frame_iterator(video_path):
            frame_buffer.append(frame)

            if len(frame_buffer) < segment_len:
                continue

            clip_tensor = self._assemble_clip(frame_buffer)
            if clip_tensor is None:
                continue

            feature = self._forward_clip(clip_tensor)
            features.append(feature)
            segment_count += 1

            if self.max_segments and segment_count >= self.max_segments:
                self.logger.debug(
                    "Reached max_segments=%s for %s. Stopping further processing.",
                    self.max_segments,
                    video_stem,
                )
                frame_buffer.clear()
                break

            for _ in range(step_size):
                if frame_buffer:
                    frame_buffer.popleft()

        if frame_buffer:
            clip_tensor = self._assemble_clip(frame_buffer)
            if clip_tensor is not None:
                feature = self._forward_clip(clip_tensor)
                features.append(feature)

        if not features:
            return None

        return np.stack(features)


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
    parser.add_argument('--log_level', default='info', choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='Logging verbosity.')
    parser.add_argument('--max_segments', type=int, default=None,
                        help='Maximum number of temporal segments to extract per video.')

    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = setup_logging(log_level)
    logger.info("Starting UCF-Crime feature extraction.")

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading feature extraction model...")
    try:
        extractor = VideoFeatureExtractor(
            model_path=args.model_path,
            device=args.device,
            repo_root=args.repo_root,
            sample_duration=args.segment_len,
            max_segments=args.max_segments,
        )
        logger.info(f"Model ready on device: {extractor.device}")
    except Exception as exc:
        logger.error(f"Failed to initialise feature extractor: {exc}")
        return 1

    logger.debug("Command line arguments: %s", vars(args))
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
            logger.debug("Processing video: %s", video_path)
            features = extractor.extract_features_from_video(
                video_path,
                video_name=video_name,
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
            logger.debug("Saved to %s", output_file)
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
