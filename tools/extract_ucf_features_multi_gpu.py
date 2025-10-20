#!/usr/bin/env python3
"""
Extract features from UCF-Crime videos using the Kenshohara 3D ResNeXt-101 pipeline.
This script matches the preprocessing and model definitions from
https://github.com/kenshohara/video-classification-3d-cnn-pytorch to produce
compatible feature tensors for downstream tasks.
"""

import argparse
import itertools
import logging
import math
import multiprocessing as mp
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


def _get_video_frame_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return total_frames, fps


def _estimate_total_segments(total_frames, segment_len, overlap):
    if total_frames is None or total_frames <= 0:
        return 0
    if segment_len <= 0:
        return 0
    if overlap >= segment_len:
        return 0

    step_size = segment_len - overlap
    if total_frames <= segment_len:
        return 1

    remaining = max(0, total_frames - segment_len)
    return math.ceil(remaining / step_size) + 1


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

    def _frame_iterator(self, video_path, start_frame=0, end_frame=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Failed to open video: {video_path}")
            return
        if start_frame and start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = max(0, start_frame)
        try:
            while True:
                if end_frame is not None and current_frame >= end_frame:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield Image.fromarray(frame)
                current_frame += 1
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

    def extract_features_from_video(
        self,
        video_path,
        video_name=None,
        segment_len=16,
        overlap=8,
        start_segment=0,
        max_segments=None,
    ):
        if segment_len <= 0:
            raise ValueError("segment_len must be a positive integer.")
        if overlap >= segment_len:
            raise ValueError("overlap must be smaller than segment_len to make progress.")

        step_size = segment_len - overlap
        frame_buffer = deque()
        features = []
        produced_segments = 0
        video_stem = video_name or Path(video_path).stem

        effective_limit = None
        if max_segments is not None:
            effective_limit = max_segments
        elif self.max_segments is not None:
            effective_limit = self.max_segments

        start_segment = max(0, start_segment)
        start_frame = start_segment * step_size
        end_frame = None
        if effective_limit is not None:
            frames_needed = segment_len + max(0, effective_limit - 1) * step_size
            end_frame = start_frame + frames_needed

        for frame in self._frame_iterator(video_path, start_frame=start_frame, end_frame=end_frame):
            frame_buffer.append(frame)

            if len(frame_buffer) < segment_len:
                continue

            clip_tensor = self._assemble_clip(frame_buffer)
            if clip_tensor is None:
                continue

            feature = self._forward_clip(clip_tensor)
            features.append(feature)
            produced_segments += 1

            if effective_limit and produced_segments >= effective_limit:
                self.logger.debug(
                    "Reached max_segments=%s for %s. Stopping further processing.",
                    effective_limit,
                    video_stem,
                )
                frame_buffer.clear()
                break

            for _ in range(step_size):
                if frame_buffer:
                    frame_buffer.popleft()

        if frame_buffer and (not effective_limit or produced_segments < effective_limit):
            clip_tensor = self._assemble_clip(frame_buffer)
            if clip_tensor is not None:
                feature = self._forward_clip(clip_tensor)
                features.append(feature)

        if not features:
            return None

        return np.stack(features)


def _gpu_worker(video_path, model_path, repo_root, segment_len, overlap, gpu_id,
                tasks, log_level, temp_dir, result_queue):
    try:
        log_level_value = getattr(logging, log_level.upper(), logging.INFO)
        setup_logging(log_level_value)

        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(gpu_id)
            except Exception:
                pass

        extractor = VideoFeatureExtractor(
            model_path=model_path,
            device=device,
            repo_root=repo_root,
            sample_duration=segment_len,
        )

        worker_results = []
        for start_segment, num_segments, temp_file in tasks:
            try:
                features = extractor.extract_features_from_video(
                    video_path,
                    segment_len=segment_len,
                    overlap=overlap,
                    start_segment=start_segment,
                    max_segments=num_segments,
                )

                if features is not None and len(features) > 0:
                    np.save(temp_file, features)
                    worker_results.append((start_segment, temp_file, int(features.shape[0])))
                else:
                    worker_results.append((start_segment, None, 0))
            except Exception as exc:
                logging.error(
                    "GPU %s failed chunk starting at segment %s: %s",
                    gpu_id,
                    start_segment,
                    exc,
                )
                worker_results.append((start_segment, None, 0))
        result_queue.put(worker_results)
    except Exception as exc:
        logging.error("Worker on GPU %s terminated with error: %s", gpu_id, exc)
        result_queue.put([])


def process_video_multi_gpu(video_path, video_name, args, gpu_ids, logger):
    total_frames, fps = _get_video_frame_info(video_path)
    if total_frames is None:
        logger.error("Unable to read video metadata for %s", video_path)
        return None

    step_size = args.segment_len - args.overlap
    if step_size <= 0:
        logger.error("Invalid segment/overlap configuration.")
        return None

    total_segments = _estimate_total_segments(total_frames, args.segment_len, args.overlap)
    if total_segments == 0:
        logger.error("No segments computed for %s", video_path)
        return None

    if args.max_segments and args.max_segments > 0:
        total_segments = min(total_segments, args.max_segments)

    logger.info(
        "Video %s: %d frames, %.2f fps, distributing %d segments across GPUs %s",
        video_name,
        total_frames,
        fps if fps else 0.0,
        total_segments,
        gpu_ids,
    )

    chunk_size = args.segments_per_chunk or math.ceil(total_segments / len(gpu_ids))
    chunk_size = max(1, chunk_size)

    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    tasks_per_gpu = {gpu_id: [] for gpu_id in gpu_ids}
    start_segment = 0
    gpu_cycle = itertools.cycle(gpu_ids)

    while start_segment < total_segments:
        gpu_id = next(gpu_cycle)
        num_segments = min(chunk_size, total_segments - start_segment)
        temp_file = temp_dir / f"{video_name}_chunk_{start_segment:06d}.npy"
        tasks_per_gpu[gpu_id].append((start_segment, num_segments, str(temp_file)))
        start_segment += num_segments

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []

    for gpu_id, gpu_tasks in tasks_per_gpu.items():
        if not gpu_tasks:
            continue
        p = ctx.Process(
            target=_gpu_worker,
            args=(
                video_path,
                args.model_path,
                args.repo_root,
                args.segment_len,
                args.overlap,
                gpu_id,
                gpu_tasks,
                args.log_level,
                str(temp_dir),
                result_queue,
            ),
        )
        p.start()
        processes.append(p)

    results = []
    for _ in processes:
        worker_results = result_queue.get()
        results.extend(worker_results)

    for p in processes:
        p.join()

    results.sort(key=lambda item: item[0])
    chunks = []
    for start_segment, temp_file, count in results:
        if temp_file and count > 0 and os.path.exists(temp_file):
            chunks.append(np.load(temp_file, allow_pickle=False))
            os.remove(temp_file)

    try:
        temp_dir.rmdir()
    except OSError:
        pass

    if not chunks:
        logger.error("Failed to extract any features for %s", video_name)
        return None

    return np.concatenate(chunks, axis=0)


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
    parser.add_argument('--gpu_ids', default=None,
                        help='Comma-separated list of GPU indices for intra-video parallel processing.')
    parser.add_argument('--segments_per_chunk', type=int, default=None,
                        help='Number of segments per GPU chunk when using --gpu_ids.')
    parser.add_argument('--temp_dir', default='./tmp_ucf_features',
                        help='Temporary directory for intermediate chunk features when using --gpu_ids.')

    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = setup_logging(log_level)
    logger.info("Starting UCF-Crime feature extraction.")

    os.makedirs(args.output_dir, exist_ok=True)

    gpu_ids = []
    if args.gpu_ids:
        try:
            gpu_ids = [int(g.strip()) for g in args.gpu_ids.split(',') if g.strip() != '']
        except ValueError:
            logger.error("Invalid --gpu_ids provided. Use comma-separated integers (e.g., 0,1,2,3).")
            return 1

    use_multi_gpu = len(gpu_ids) > 1

    if use_multi_gpu and not torch.cuda.is_available():
        logger.error("CUDA is not available but --gpu_ids was provided.")
        return 1

    effective_device = args.device
    if len(gpu_ids) == 1:
        effective_device = f"cuda:{gpu_ids[0]}"

    extractor = None
    if not use_multi_gpu:
        logger.info("Loading feature extraction model...")
        try:
            extractor = VideoFeatureExtractor(
                model_path=args.model_path,
                device=effective_device,
                repo_root=args.repo_root,
                sample_duration=args.segment_len,
                max_segments=args.max_segments,
            )
            logger.info(f"Model ready on device: {extractor.device}")
        except Exception as exc:
            logger.error(f"Failed to initialise feature extractor: {exc}")
            return 1
    else:
        logger.info(
            "Multi-GPU mode enabled across GPUs %s with segments_per_chunk=%s",
            gpu_ids,
            args.segments_per_chunk,
        )

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
            if use_multi_gpu:
                features = process_video_multi_gpu(
                    video_path,
                    video_name,
                    args,
                    gpu_ids,
                    logger,
                )
            else:
                features = extractor.extract_features_from_video(
                    video_path,
                    video_name=video_name,
                    segment_len=args.segment_len,
                    overlap=args.overlap,
                    max_segments=args.max_segments,
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
