#!/usr/bin/env python 3
"""
Street Obstacle Sequences (SOS) split builder.

정책:
- train/          : 정상 주행 (단일 비디오로 취급, label=0)
- test/sequence_* : 장애물 포함 (label=1)
- LANP 학습: train(정상) + test 시퀀스 중 앞의 N개를 이상으로 포함 (기본 N=3)
- UNet 학습: train(정상)만 사용
- Test: 나머지 test 시퀀스 모두 이상으로 평가

라인 포맷: <relative_video_path>,<label>,<frame_len>
예) train,0,12345 / test/sequence_003,1,510
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate SOS splits (train normal, test anomaly).")
    p.add_argument("--sos-root", type=Path, default=Path("data/street_obstacle_sequences"), help="Dataset root.")
    p.add_argument("--out-dir", type=Path, default=None, help="Output dir for split txt (default: sos-root).")
    p.add_argument("--lanp_anom_from_test", type=int, default=3, help="How many test sequences to include as anomalies in LANP train.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    return p.parse_args()


def count_frames(seq_dir: Path) -> int:
    return len([f for f in seq_dir.iterdir() if f.suffix.lower() in {".jpg", ".png"}])


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = args.out_dir or args.sos_root

    # Train: single folder with normal frames
    train_dir = args.sos_root / "train"
    if not train_dir.is_dir():
        raise RuntimeError(f"Train dir not found: {train_dir}")

    # Test: anomaly sequences
    test_root = args.sos_root / "test"
    seq_dirs = sorted([d for d in test_root.iterdir() if d.is_dir()])
    rng.shuffle(seq_dirs)
    lanp_train_anom = seq_dirs[: min(args.lanp_anom_from_test, len(seq_dirs))]
    test_seqs = seq_dirs[min(args.lanp_anom_from_test, len(seq_dirs)) :]

    lanp_lines: List[str] = []
    unet_lines: List[str] = []
    test_lines: List[str] = []
    summary: Dict[str, int] = {}

    rel_train = "train"
    train_frames = count_frames(train_dir)
    lanp_lines.append(f"{rel_train},0,{train_frames}\n")
    unet_lines.append(f"{rel_train},0,{train_frames}\n")

    for d in lanp_train_anom:
        rel = f"test/{d.name}"
        flen = count_frames(d)
        lanp_lines.append(f"{rel},1,{flen}\n")
    for d in test_seqs:
        rel = f"test/{d.name}"
        flen = count_frames(d)
        test_lines.append(f"{rel},1,{flen}\n")

    summary = {
        "train_normal_frames": train_frames,
        "lanp_train_anomaly_seqs": len(lanp_train_anom),
        "test_anomaly_seqs": len(test_seqs),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lanp_train_split.txt").write_text("".join(lanp_lines))
    (out_dir / "unet_train_split.txt").write_text("".join(unet_lines))
    (out_dir / "test_split.txt").write_text("".join(test_lines))

    print("Wrote splits to", out_dir.resolve())
    print(summary)


if __name__ == "__main__":
    main()
