#!/usr/bin/env python3
"""
Frame-folder 기반으로 ResNeXt-101(Kinetics) 피처를 추출해 LANP가 사용하는
`*_res.npy` 형태로 저장합니다.

예시:
python tools/extract_resnext_features_from_frames.py \
  --frames_root data/STTAD/rgb-images \
  --output_root data/STTAD/features \
  --model_path /path/to/resnext-101-kinetics.pth \
  --device cuda
"""

from __future__ import annotations

import argparse
import sys
import subprocess
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image


def resolve_video_repo(script_dir: Path) -> Path:
    """
    video-classification-3d-cnn-pytorch 경로를 찾아 sys.path에 추가.
    """
    candidates: List[Path] = [
        script_dir / "video-classification-3d-cnn-pytorch",
        script_dir.parent / "video-classification-3d-cnn-pytorch",
    ]
    for cand in candidates:
        if cand.exists():
            sys.path.insert(0, str(cand))
            return cand
    raise FileNotFoundError("video-classification-3d-cnn-pytorch 폴더를 찾을 수 없습니다.")


SCRIPT_DIR = Path(__file__).resolve().parent
VIDEO_REPO = resolve_video_repo(SCRIPT_DIR)

# 이제 video-classification-3d-cnn-pytorch 모듈 import
from model import generate_model  # type: ignore  # noqa: E402
from mean import get_mean  # type: ignore  # noqa: E402
from spatial_transforms import Compose, Normalize, Scale, CenterCrop, ToTensor  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ResNeXt-101(Kinetics)로 프레임 폴더 피처 추출")
    p.add_argument("--frames_root", default=None, help="클래스/비디오별 프레임 폴더 루트")
    p.add_argument("--videos_root", default=None, help="비디오 파일 루트 (frames_root보다 우선)")
    p.add_argument("--output_root", default="data/STTAD/features", help="출력 루트 (class/video_res.npy)")
    p.add_argument("--model_path", required=True, help="resnext-101-kinetics.pth 가중치 경로")
    p.add_argument("--device", default=None, help="cpu|cuda|cuda:0|mps (미지정 시 자동)")
    p.add_argument("--batch_size", type=int, default=8, help="클립 배치 크기")
    p.add_argument("--exts", nargs="+", default=[".jpg", ".png"], help="프레임 확장자 목록")
    p.add_argument("--video_exts", nargs="+", default=[".mp4", ".mov", ".avi"], help="비디오 확장자 목록")
    p.add_argument("--temp_dir", default=None, help="비디오를 프레임으로 임시 추출할 디렉터리(기본: tmp).")
    return p.parse_args()


class OptStub:
    """
    generate_model이 기대하는 옵션을 최소한으로 흉내냄.
    """

    def __init__(self, device: torch.device):
        self.mode = "feature"
        self.model_name = "resnext"
        self.model_depth = 101
        self.resnext_cardinality = 32
        self.resnet_shortcut = "A"
        self.wide_resnet_k = 2
        self.sample_size = 112
        self.sample_duration = 16
        self.n_classes = 400
        self.no_cuda = device.type != "cuda"


def select_device(requested: str | None) -> torch.device:
    if requested:
        req = requested.lower()
        if req.startswith("cuda") and torch.cuda.is_available():
            return torch.device(req if ":" in req else "cuda:0")
        if req == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_path: Path, device: torch.device):
    opt = OptStub(device)
    model = generate_model(opt)
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # DataParallel 저장본이면 module. prefix 제거
    if len(state) > 0 and next(iter(state)).startswith("module."):
        state = OrderedDict((k.replace("module.", "", 1), v) for k, v in state.items())

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    mean = get_mean()
    spatial = Compose(
        [
            Scale(opt.sample_size),
            CenterCrop(opt.sample_size),
        ToTensor(),
        Normalize(mean, [1, 1, 1]),
    ]
    )
    return model, opt, spatial


def iter_frame_dirs(frames_root: Path, exts: Iterable[str]) -> Iterable[Path]:
    exts_norm = {e.lower() for e in exts}
    for class_dir in sorted(frames_root.iterdir()):
        if not class_dir.is_dir():
            continue
        for vid_dir in sorted(class_dir.iterdir()):
            if not vid_dir.is_dir():
                continue
            # 프레임이 실제로 있는 폴더만 처리
            has_frames = any(
                p.suffix.lower() in exts_norm for p in vid_dir.iterdir() if p.is_file()
            )
            if has_frames:
                yield vid_dir


def iter_video_files(videos_root: Path, exts: Iterable[str]) -> Iterable[Path]:
    exts_norm = {e.lower() for e in exts}
    for path in sorted(videos_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in exts_norm:
            yield path


def load_clip(paths: List[Path], spatial, target_len: int) -> torch.Tensor | None:
    # 개별 프레임 로드 실패 시 건너뛰고, 유효 프레임만 사용
    imgs = []
    for p in paths[:target_len]:
        try:
            imgs.append(spatial(Image.open(p).convert("RGB")))
        except Exception:
            # 깨진 프레임은 스킵
            continue

    if not imgs:
        return None

    # 길이가 모자라면 마지막 유효 프레임을 반복해서 pad
    while len(imgs) < target_len:
        imgs.append(imgs[-1])

    clip = torch.stack(imgs[:target_len], dim=1)  # C, T, H, W
    return clip


def process_video(
    frame_dir: Path,
    rel_dir: Path,
    model,
    device: torch.device,
    spatial,
    opt: OptStub,
    output_root: Path,
    batch_size: int,
    exts: List[str],
):
    frames = sorted(
        [p for p in frame_dir.iterdir() if p.suffix.lower() in exts and p.is_file()]
    )
    if not frames:
        return

    clips: List[torch.Tensor] = []
    step = opt.sample_duration
    for idx in range(0, len(frames), step):
        chunk = frames[idx : idx + step]
        if not chunk:
            continue
        clip = load_clip(chunk, spatial, step)
        if clip is None:
            continue
        clips.append(clip)

    outputs: List[torch.Tensor] = []
    for start in range(0, len(clips), batch_size):
        batch = torch.stack(clips[start : start + batch_size]).to(device)
        with torch.no_grad():
            feat = model(batch)
            # generate_model(mode=feature)는 global pool feature를 반환
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            outputs.append(feat.cpu())

    feats = torch.cat(outputs, dim=0).numpy().astype("float32")
    out_path = output_root / f"{rel_dir.as_posix()}_res.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, feats)
    print(f"[saved] {out_path} {feats.shape}")


def main():
    args = parse_args()
    frames_root = Path(args.frames_root).resolve() if args.frames_root else None
    videos_root = Path(args.videos_root).resolve() if args.videos_root else None
    output_root = Path(args.output_root).resolve()
    model_path = Path(args.model_path).resolve()
    device = select_device(args.device)

    print(f"[info] using device: {device}")
    print(f"[info] frames_root={frames_root}")
    print(f"[info] videos_root={videos_root}")
    print(f"[info] output_root={output_root}")
    print(f"[info] model_path={model_path}")
    print(f"[info] video repo at {VIDEO_REPO}")

    model, opt, spatial = load_model(model_path, device)
    exts = [e.lower() for e in args.exts]

    if videos_root:
        tmp_root = Path(args.temp_dir) if args.temp_dir else Path(tempfile.mkdtemp(prefix="tudat_frames_"))
        try:
            for vid_file in iter_video_files(videos_root, args.video_exts):
                rel = vid_file.relative_to(videos_root)
                rel_dir = rel.with_suffix("")  # class/video
                frame_dir = tmp_root / rel_dir
                frame_dir.mkdir(parents=True, exist_ok=True)
                # ffmpeg로 프레임 추출
                cmd = [
                    "ffmpeg",
                    "-i",
                    str(vid_file),
                    "-qscale:v",
                    "2",
                    "-vsync",
                    "0",
                    str(frame_dir / "%05d.jpg"),
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                process_video(
                    frame_dir=frame_dir,
                    rel_dir=rel_dir,
                    model=model,
                    device=device,
                    spatial=spatial,
                    opt=opt,
                    output_root=output_root,
                    batch_size=args.batch_size,
                    exts=exts,
                )
        finally:
            if not args.temp_dir and tmp_root.exists():
                shutil.rmtree(tmp_root, ignore_errors=True)
    elif frames_root:
        for vid_dir in iter_frame_dirs(frames_root, exts):
            rel_dir = vid_dir.relative_to(frames_root)
            process_video(
                frame_dir=vid_dir,
                rel_dir=rel_dir,
                model=model,
                device=device,
                spatial=spatial,
                opt=opt,
                output_root=output_root,
                batch_size=args.batch_size,
                exts=exts,
            )
    else:
        raise ValueError("frames_root 또는 videos_root 중 하나는 지정해야 합니다.")


if __name__ == "__main__":
    main()
