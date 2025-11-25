import os
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

from data.base_dataset import BaseDataset


def _strip_suffix(stem: str, suffix: str) -> str:
    """Remove trailing suffix (without extension) from stem if present."""
    if suffix.endswith(".npy"):
        suffix = suffix[:-4]
    if suffix and stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


class Dataset_TUDAT(BaseDataset):
    """
    TU-DAT loader.

    Assumptions:
    - Features stored under feature_path/<class>/<video>_res.npy
    - Classes are split into normal_classes and abnormal_classes.
    - Training uses only normal_classes; evaluation uses both.
    - Frame-level labels are unavailable, so evaluation labels are per snippet
      expanded to frames (all 0 for normal, all 1 for abnormal).
    """

    def initialize(self, args, sample_type="uniform", is_train=True, is_normal=True, eval_train=False):
        self.dataset_name = args.dataset
        self.seg_len = args.segment_len
        self.process_len = args.process_len
        self.sample_type = sample_type
        self.is_train = is_train
        self.is_normal = is_normal
        self.eval_train = eval_train
        self.feature_path = args.feature_path
        self.feature_name_end = args.feature_name_end
        self.normal_classes: Set[str] = set(getattr(args, "normal_classes", []))
        self.abnormal_classes: Set[str] = set(getattr(args, "abnormal_classes", []))
        self.train_list = self._load_split_list(getattr(args, "training_split", None))
        self.test_list = self._load_split_list(getattr(args, "testing_split", None))
        self.logger_info = None
        self.video_info_dict: Dict[str, dict] = {}

        self._build_entries()

    def _load_split_list(self, path: str | None) -> Set[str] | None:
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            return None
        entries = [line.strip() for line in p.read_text().splitlines() if line.strip()]
        return set(entries)

    def _scan_feature_files(self, allowed: Set[str] | None = None) -> List[Path]:
        root = Path(self.feature_path)
        suffix = self.feature_name_end
        glob_pat = f"*{suffix}" if suffix.endswith(".npy") else f"*{suffix}.npy"
        files = []
        for path in root.glob(f"**/{glob_pat}"):
            key = self._video_name_from_path(path)
            if allowed is not None and key not in allowed:
                continue
            files.append(path)
        files.sort()
        return files

    def _class_from_path(self, path: Path) -> str:
        # First directory under feature_path is treated as class
        rel = path.relative_to(self.feature_path)
        return rel.parts[0] if rel.parts else ""

    def _video_name_from_path(self, path: Path) -> str:
        rel = path.relative_to(self.feature_path)
        cls = rel.parts[0] if rel.parts else ""
        subparts = list(rel.parts[1:-1])  # optional subfolders
        base = _strip_suffix(path.stem, self.feature_name_end)
        parts = [cls] + subparts + [base]
        return "/".join(p for p in parts if p)

    def _is_normal_class(self, cls: str) -> bool:
        if self.normal_classes:
            return cls in self.normal_classes
        return cls not in self.abnormal_classes

    def _build_train_entries(self):
        feature_files = self._scan_feature_files(self.train_list)
        for fpath in feature_files:
            cls = self._class_from_path(fpath)
            feature = np.load(fpath)
            if feature.ndim == 3:
                feature = np.mean(feature, axis=1)
            T = feature.shape[0]
            sample_idxs = self.uniform_sampling(T)
            feature = feature[sample_idxs]
            if self._is_normal_class(cls):
                pseudo_label = np.zeros(len(sample_idxs))
                high_conf = 1
            else:
                pseudo_label = np.ones(len(sample_idxs))
                high_conf = 0
            reweight = np.ones_like(pseudo_label)
            video_key = self._video_name_from_path(fpath)
            self.video_info_dict[video_key] = {
                "feature": feature,
                "pseudo_label": pseudo_label,
                "reweight": reweight,
                "high_confidence_norvideo": high_conf,
            }

        num_nor = sum(1 for k in self.video_info_dict if self._is_normal_class(k.split('/')[0]))
        num_abn = len(self.video_info_dict) - num_nor
        self.logger_info = (
            f"Loaded {len(self.video_info_dict)} TU-DAT training videos "
            f"(normal {num_nor}, abnormal {num_abn})."
        )

    def _build_eval_entries(self):
        feature_files = self._scan_feature_files(self.test_list)
        for fpath in feature_files:
            cls = self._class_from_path(fpath)
            label_video = 0 if self._is_normal_class(cls) else 1
            feature = np.load(fpath)
            if feature.ndim == 3:
                feature = np.mean(feature, axis=1)
            T = feature.shape[0]
            snipts_len = T
            # Expand snippet labels to frame-level blocks
            frame_label = 0.0 if label_video == 0 else 1.0
            label_test = np.full((snipts_len, self.seg_len), frame_label, dtype=np.float32)

            video_key = self._video_name_from_path(fpath)
            info = {
                "feature": feature,
                "label_video": label_video,
                "label_test": label_test,
            }
            self.video_info_dict[video_key] = info

        self.logger_info = f"Loaded {len(self.video_info_dict)} TU-DAT eval videos."

    def _build_entries(self):
        if self.is_train:
            self._build_train_entries()
        else:
            self._build_eval_entries()
