#!/usr/bin/env python3
"""
TU-DAT split/라벨 디버그용 스크립트.

- config를 읽어 normal/abnormal 클래스 목록을 확인
- train/test split 겹침 여부, 항목 개수
- feature 파일 존재 여부
- 라벨 매핑(클래스→정상/이상) 통계
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Debug TU-DAT splits and label mapping.")
    p.add_argument("--config", default="config/config_tudat.yaml", help="YAML config path")
    p.add_argument("--show_samples", type=int, default=5, help="샘플 출력 개수")
    p.add_argument("--video_exts", nargs="+", default=[".mp4", ".mov", ".avi"], help="비디오 확장자")
    p.add_argument("--train_split", default=None, help="train split 경로 (config 값 덮어쓰기)")
    p.add_argument("--test_split", default=None, help="test split 경로 (config 값 덮어쓰기)")
    return p.parse_args()


def load_split(path: str | None) -> List[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


def is_normal_class(cls: str, normal_classes: Iterable[str], abnormal_classes: Iterable[str]) -> bool:
    nset = set(normal_classes)
    aset = set(abnormal_classes)
    if nset:
        return cls in nset
    return cls not in aset


def feature_exists(entry: str, feature_root: Path, suffix: str) -> bool:
    path = feature_root / entry
    if suffix:
        if suffix.startswith("_") and not suffix.endswith(".npy"):
            suffix = suffix + ".npy"
        path = path.with_name(path.name + suffix if not suffix.endswith(".npy") else path.name + suffix)
    if not str(path).endswith(".npy"):
        path = path.with_suffix(".npy")
    return path.exists()


def main() -> None:
    args = parse_args()
    cfg = yaml.load(Path(args.config).read_text(), Loader=yaml.FullLoader)

    normal_classes = cfg.get("normal_classes", []) or []
    abnormal_classes = cfg.get("abnormal_classes", []) or []
    feature_path = Path(cfg.get("feature_path", "data/TU-DAT/features"))
    train_split = args.train_split or cfg.get("training_split", "")
    test_split = args.test_split or cfg.get("testing_split", "")
    suffix = cfg.get("feature_name_end", "_res.npy") or ""

    train_entries = load_split(train_split)
    test_entries = load_split(test_split)

    train_set = set(train_entries)
    test_set = set(test_entries)
    overlap = train_set & test_set

    print(f"[config] normal_classes={normal_classes}")
    print(f"[config] abnormal_classes={abnormal_classes}")
    print(f"[paths] feature_path={feature_path}")
    print(f"[split] train={len(train_set)}, test={len(test_set)}, overlap={len(overlap)}")
    if overlap:
        print("  overlap samples:", sorted(list(overlap))[: args.show_samples])

    def summarize(entries: Iterable[str], name: str) -> None:
        c = Counter()
        missing_feat = 0
        for e in entries:
            cls = e.split("/")[0] if "/" in e else e
            lbl = "normal" if is_normal_class(cls, normal_classes, abnormal_classes) else "abnormal"
            c[lbl] += 1
            if not feature_exists(e, feature_path, suffix):
                missing_feat += 1
        print(f"[{name}] normal={c['normal']}, abnormal={c['abnormal']}, missing_features={missing_feat}")

    summarize(train_set, "train")
    summarize(test_set, "test")

    # 샘플 출력
    if args.show_samples > 0:
        print("[samples] train:")
        for e in list(train_set)[: args.show_samples]:
            cls = e.split("/")[0] if "/" in e else e
            lbl = "N" if is_normal_class(cls, normal_classes, abnormal_classes) else "A"
            print(f"  {e} -> {lbl}")
        print("[samples] test:")
        for e in list(test_set)[: args.show_samples]:
            cls = e.split("/")[0] if "/" in e else e
            lbl = "N" if is_normal_class(cls, normal_classes, abnormal_classes) else "A"
            print(f"  {e} -> {lbl}")


if __name__ == "__main__":
    sys.exit(main())
