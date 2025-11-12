#!/usr/bin/env python3
"""
Extract per-box deep features (ResNet-50 penultimate activations) from cached YOLO detections.

Outputs are stored as an object-array `deep_features.npy` so that each entry aligns
with a frame in the ShanghaiTech split order, matching `convert_yolo_detections.py`
and `make_anomaly_prompts.py`.
"""

import argparse
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T


class SplitEntry(NamedTuple):
    name: str
    label: int
    frames: int


def _load_split_entries(split_file: Path) -> List[SplitEntry]:
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    entries: List[SplitEntry] = []
    with open(split_file, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) < 3:
                raise ValueError(f"Malformed split line: {line}")
            entries.append(SplitEntry(name=parts[0], label=int(parts[1]), frames=int(parts[2])))
    if not entries:
        raise ValueError(f"No entries parsed from {split_file}")
    return entries


def _sorted_items(det_map: Dict[int, Dict[str, np.ndarray]]):
    def keyfun(k):
        try:
            return (0, int(k))
        except (TypeError, ValueError):
            return (1, str(k))

    return sorted(det_map.items(), key=lambda kv: keyfun(kv[0]))


def _index_frame_files(frames_dir: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not frames_dir.exists():
        return mapping
    for p in frames_dir.iterdir():
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            mapping[p.stem] = p.name
    return mapping


def _resolve_frame_name(mapping: Dict[str, str], key: str) -> Optional[str]:
    if key in mapping:
        return mapping[key]
    no_zero = key.lstrip("0")
    if no_zero and no_zero in mapping:
        return mapping[no_zero]
    pad6 = key.zfill(6)
    if pad6 in mapping:
        return pad6
    return None


def _crop_with_pad(image: np.ndarray, box: Sequence[float], pad: float = 0.1) -> Optional[np.ndarray]:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = (x2 - x1), (y2 - y1)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    pad_w, pad_h = bw * (1.0 + pad), bh * (1.0 + pad)
    nx1 = max(0, int(cx - pad_w * 0.5))
    ny1 = max(0, int(cy - pad_h * 0.5))
    nx2 = min(w, int(cx + pad_w * 0.5))
    ny2 = min(h, int(cy + pad_h * 0.5))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return image[ny1:ny2, nx1:nx2]


def _build_resnet50(device: torch.device):
    model = tv_models.resnet50(pretrained=True)
    model.fc = nn.Identity()
    model.eval().to(device)
    return model


def parse_args():
    ap = argparse.ArgumentParser("Extract deep features from YOLO detections")
    ap.add_argument("--dataset_name", type=str, default="shanghaitech")
    ap.add_argument("--split", type=str, required=True, choices=["training", "testing"])
    ap.add_argument("--data_root", type=str, default="./data/shanghaitech")
    ap.add_argument("--detections_root", type=str, default="./artifacts/detections")
    ap.add_argument("--frames_root", type=str, default=None, help="Override frames dir (defaults to data_root/split/frames)")
    ap.add_argument("--output_root", type=str, default="./artifacts/features")
    ap.add_argument("--train_split_file", type=str, default="./data/shanghaitech/train_split.txt")
    ap.add_argument("--test_split_file", type=str, default="./data/shanghaitech/test_split.txt")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--crop_pad", type=float, default=0.10)
    return ap.parse_args()


def main():
    args = parse_args()
    split_tag = "train" if args.split == "training" else "test"
    split_file = Path(args.train_split_file if split_tag == "train" else args.test_split_file)
    split_entries = _load_split_entries(split_file)

    detections_dir = Path(args.detections_root) / args.split
    if not detections_dir.exists():
        raise FileNotFoundError(f"Detections directory missing: {detections_dir}")

    frames_root = Path(args.frames_root) if args.frames_root else Path(args.data_root) / args.split / "frames"
    out_dir = Path(args.output_root) / args.dataset_name / split_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_resnet50(device)
    preprocess = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((args.img_size, args.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    deep_frames: List[np.ndarray] = []
    expected_frames_total = sum(entry.frames for entry in split_entries)

    with torch.no_grad():
        for entry in split_entries:
            det_path = detections_dir / entry.name / "detections.npy"
            if not det_path.exists():
                raise FileNotFoundError(f"Missing detections for video {entry.name}: {det_path}")
            det_map = np.load(det_path, allow_pickle=True).item()
            frame_items = _sorted_items(det_map)
            if len(frame_items) != entry.frames:
                raise ValueError(
                    f"Frame count mismatch for {entry.name}: split={entry.frames}, detections={len(frame_items)}"
                )

            frame_dir = frames_root / entry.name
            frame_lookup = _index_frame_files(frame_dir)

            for frame_key, payload in frame_items:
                boxes = np.asarray(payload.get("boxes", np.zeros((0, 4), dtype=np.float32)), dtype=np.float32).reshape(
                    -1, 4
                )
                feats = np.zeros((len(boxes), 2048), dtype=np.float32)
                if len(boxes) == 0:
                    deep_frames.append(feats)
                    continue

                frame_name = _resolve_frame_name(frame_lookup, str(frame_key))
                if not frame_name:
                    deep_frames.append(feats)
                    continue
                img_bgr = cv2.imread(str(frame_dir / frame_name))
                if img_bgr is None:
                    deep_frames.append(feats)
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                crops: List[torch.Tensor] = []
                idxs: List[int] = []
                for idx, box in enumerate(boxes):
                    crop = _crop_with_pad(img_rgb, box, pad=args.crop_pad)
                    if crop is None:
                        continue
                    crops.append(preprocess(crop))
                    idxs.append(idx)

                if crops:
                    for start in range(0, len(crops), args.batch_size):
                        end = start + args.batch_size
                        batch = torch.stack(crops[start:end], dim=0).to(device, non_blocking=True)
                        emb = model(batch)
                        emb = torch.flatten(emb, 1).cpu().numpy().astype(np.float32, copy=False)
                        for offset, det_idx in enumerate(idxs[start:end]):
                            feats[det_idx] = emb[offset]

                deep_frames.append(feats)

    if len(deep_frames) != expected_frames_total:
        raise RuntimeError(
            f"Collected {len(deep_frames)} frames but expected {expected_frames_total}. Check for missing detections."
        )

    out_path = out_dir / "deep_features.npy"
    np.save(out_path, np.array(deep_frames, dtype=object), allow_pickle=True)
    print(f"[save] {out_path}  (frames={len(deep_frames)})")


if __name__ == "__main__":
    main()
