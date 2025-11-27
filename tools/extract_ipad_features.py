#!/usr/bin/env python3
"""
IPAD 프레임 폴더에서 3D ResNeXt-101(Kinetics) 피처를 추출해 LANP 학습/평가에 사용.

출력: <out_root>/<video_path>_res.npy
  - <video_path> 예: S01/testing/frames/10
  - npy shape: (num_clips, 2048)  # 16프레임 비중첩 클립 단위

기본 설정은 Shanghaitech과 동일하게 16프레임 클립, 입력 112x112, CenterCrop,
ResNeXt-101 Kinetics 사전학습 가중치(resnext-101-kinetics.pth)를 사용한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
VC_ROOT = ROOT / "video-classification-3d-cnn-pytorch"
if str(VC_ROOT) not in sys.path:
    sys.path.append(str(VC_ROOT))

from model import generate_model  # type: ignore  # noqa: E402
from mean import get_mean  # type: ignore  # noqa: E402
from spatial_transforms import Compose, Normalize, Scale, CenterCrop, ToTensor  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract 3D ResNeXt-101 features from IPAD frame folders.")
    p.add_argument("--ipad-root", type=Path, default=Path("data/IPAD"), help="IPAD 루트 (Sxx/Rxx 포함).")
    p.add_argument("--out-root", type=Path, default=Path("features/ipad"), help="피처 저장 루트.")
    p.add_argument("--weights", type=Path, default=Path("resnext-101-kinetics.pth"), help="ResNeXt-101 Kinetics 가중치 경로.")
    p.add_argument("--device", type=str, default=None, help="cuda:0, mps, cpu 등. 지정 없으면 자동.")
    p.add_argument("--batch-size", type=int, default=8, help="클립 배치 크기.")
    p.add_argument("--clip-len", type=int, default=16, help="클립 길이(프레임 수).")
    p.add_argument("--stride", type=int, default=16, help="클립 stride. 기본값=비중첩.")
    p.add_argument("--sample-size", type=int, default=112, help="입력 해상도 (Scale->CenterCrop).")
    p.add_argument("--split-file", type=Path, default=None, help="(optional) 처리할 비디오 리스트 txt (<rel_path>,label,frame_len).")
    return p.parse_args()


def build_transform(sample_size: int) -> Compose:
    mean = get_mean()
    return Compose([Scale(sample_size), CenterCrop(sample_size), ToTensor(), Normalize(mean, [1, 1, 1])])


def numeric_sort_key(path: Path):
    stem = path.stem
    try:
        return int(stem)
    except ValueError:
        return stem


def list_frame_paths(video_dir: Path) -> List[Path]:
    frames = [p for p in video_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    return sorted(frames, key=numeric_sort_key)


def chunk_indices(n_frames: int, clip_len: int, stride: int) -> Iterable[Sequence[int]]:
    for start in range(0, n_frames, stride):
        end = min(start + clip_len, n_frames)
        idxs = list(range(start, end))
        if len(idxs) < clip_len:
            idxs += [idxs[-1]] * (clip_len - len(idxs))  # 마지막 프레임으로 패딩
        yield idxs


def load_model(weights: Path, device: torch.device, sample_size: int, clip_len: int) -> torch.nn.Module:
    # video-classification-3d-cnn-pytorch 옵션과 동일하게 구성
    opt = SimpleNamespace(
        mode="feature",
        model_name="resnext",
        model_depth=101,
        resnet_shortcut="B",
        resnext_cardinality=32,
        sample_size=sample_size,
        sample_duration=clip_len,
        n_classes=400,
        wide_resnet_k=2,
        no_cuda=(device.type != "cuda"),
    )
    model = generate_model(opt)

    state = torch.load(weights, map_location=device)
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def extract_video_features(
    video_dir: Path,
    model: torch.nn.Module,
    transform: Compose,
    device: torch.device,
    clip_len: int,
    stride: int,
    batch_size: int,
) -> np.ndarray:
    frame_paths = list_frame_paths(video_dir)
    if not frame_paths:
        raise ValueError(f"No frames found in {video_dir}")

    clips: List[torch.Tensor] = []
    features: List[np.ndarray] = []

    for idxs in chunk_indices(len(frame_paths), clip_len, stride):
        frames = [Image.open(frame_paths[i]).convert("RGB") for i in idxs]
        clip = torch.stack([transform(img) for img in frames], dim=0).permute(1, 0, 2, 3)  # (C,T,H,W)
        clips.append(clip)

        if len(clips) == batch_size:
            batch = torch.stack(clips, dim=0).to(device, non_blocking=True)
            with torch.no_grad():
                feat = model(batch)
            features.append(feat.cpu().numpy())
            clips.clear()

    if clips:
        batch = torch.stack(clips, dim=0).to(device, non_blocking=True)
        with torch.no_grad():
            feat = model(batch)
        features.append(feat.cpu().numpy())

    return np.concatenate(features, axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.weights, device, sample_size=args.sample_size, clip_len=args.clip_len)
    transform = build_transform(args.sample_size)

    out_root = args.out_root
    all_video_dirs = []
    if args.split_file:
        lines = Path(args.split_file).read_text().splitlines()
        for line in lines:
            if not line.strip():
                continue
            rel = line.strip().split(",")[0]
            vdir = args.ipad_root / rel
            if vdir.is_dir():
                all_video_dirs.append(vdir)
            else:
                print(f"[warn] skip missing {vdir}")
    else:
        for scenario_dir in sorted(p for p in args.ipad_root.iterdir() if p.is_dir()):
            for split in ("training", "testing"):
                frames_root = scenario_dir / split / "frames"
                if not frames_root.is_dir():
                    continue
                all_video_dirs.extend(sorted([p for p in frames_root.iterdir() if p.is_dir()]))

    if not all_video_dirs:
        raise RuntimeError(f"No video frame folders found under {args.ipad_root} (split_file={args.split_file})")

    for video_dir in tqdm(all_video_dirs, desc="Videos"):
        rel = video_dir.relative_to(args.ipad_root)  # e.g., S01/testing/frames/10
        out_path = out_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_path = out_path.with_name(out_path.name + "_res.npy")

        try:
            feats = extract_video_features(
                video_dir,
                model=model,
                transform=transform,
                device=device,
                clip_len=args.clip_len,
                stride=args.stride,
                batch_size=args.batch_size,
            )
        except Exception as e:
            print(f"[error] {rel}: {e}")
            continue

        np.save(save_path, feats)
        print(f"[save] {save_path} | clips={feats.shape[0]} dim={feats.shape[1]}")


if __name__ == "__main__":
    main()
