import os
from pathlib import Path
from typing import Dict, Set

import numpy as np

from data.base_dataset import BaseDataset
from data.normprop import normality_propagation


class Dataset_STTAD(BaseDataset):
    """
    STTAD loader that consumes pre-extracted snippet features.

    Expectations:
    - Features are stored under ``feature_path`` mirroring the label/frames layout,
      e.g., ``<feature_path>/B-B_crash/B-B_crash_0_res.npy``.
    - ``training_split`` / ``testing_split`` point to the provided
      ``trainlist.txt`` / ``testlist.txt`` files that enumerate label files.
    - ``anno_path`` should point to the ``labels`` root.
    - ``frames_root`` (optional) points to RGB frames for frame counting; if not
      provided it defaults to ``<labels_root>/../rgb-images``.
    """

    def initialize(self, args, sample_type="uniform", is_train=True, is_normal=True, eval_train=False):
        self.dataset_name = args.dataset
        self.seg_len = args.segment_len
        self.process_len = args.process_len
        self.sample_type = sample_type
        self.is_train = is_train
        self.is_normal = is_normal
        self.eval_train = eval_train
        self.r = args.r
        self.h = args.h
        self.feature_path = args.feature_path
        self.feature_name_end = args.feature_name_end
        self.labels_root = Path(args.anno_path)
        self.frames_root = Path(getattr(args, "frames_root", self.labels_root.parent / "rgb-images"))
        self.logger_info = None
        self.video_info_dict: Dict[str, dict] = {}

        np_scores_path = getattr(args, "np_scores_path", None)
        if np_scores_path:
            self.normprop_scores_dict = np.load(np_scores_path, allow_pickle=True).item()
        else:
            self.normprop_scores_dict = None

        split_file = args.training_split if (self.is_train or (not self.is_train and self.eval_train)) else args.testing_split
        self.video_meta = self._parse_split_file(split_file)
        self._build_entries()

    def _parse_split_file(self, split_path: str) -> Dict[str, dict]:
        path = Path(split_path)
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")

        meta: Dict[str, dict] = {}
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rel = Path(line)
            if len(rel.parts) < 4:
                continue
            cls_name = rel.parts[-3]
            vid_name = rel.parts[-2]
            frame_stem = rel.stem
            key = f"{cls_name}/{vid_name}"
            entry = meta.setdefault(key, {"class": cls_name, "video": vid_name, "label_frames": set()})
            try:
                entry["label_frames"].add(int(frame_stem))
            except ValueError:
                continue

        for key, entry in meta.items():
            entry["frame_total"] = self._count_frames(entry["class"], entry["video"], entry["label_frames"])
        return meta

    def _count_frames(self, class_name: str, video_name: str, labeled_frames: Set[int]) -> int:
        frame_dir = self.frames_root / class_name / video_name
        jpgs = sorted(frame_dir.glob("*.jpg"))
        pngs = sorted(frame_dir.glob("*.png"))
        total = len(jpgs) + len(pngs)
        if total > 0:
            return total
        if labeled_frames:
            return max(labeled_frames)
        return 0

    def _label_vector(self, labeled_frames: Set[int], frame_total: int, target_len: int) -> np.ndarray:
        """
        Build binary frame-level labels of length ``target_len`` (frames, not snippets).
        If annotation frames exceed ``target_len``, they are clamped to the last index.
        """
        if target_len <= 0:
            return np.zeros(0, dtype=np.float32)

        label_vec = np.zeros(target_len, dtype=np.float32)
        last_idx = target_len - 1
        for idx in labeled_frames:
            pos = max(0, min(idx - 1, last_idx))
            label_vec[pos] = 1.0

        if frame_total > target_len and target_len > 0:
            # Preserve tail value to roughly reflect longer videos.
            label_vec = np.pad(label_vec, (0, frame_total - target_len), mode="edge")
            label_vec = label_vec[:target_len]
        return label_vec

    def _sample_indices(self, length: int) -> np.ndarray:
        if length <= 0:
            return np.array([], dtype=int)
        if length <= self.process_len:
            return np.arange(length)
        return self.uniform_sampling(length)

    def _build_entries(self):
        if self.is_train:
            self._build_train_entries()
        else:
            self._build_eval_entries()

    def _build_train_entries(self):
        video_names = []
        score_v_list = []
        missing_feats = []
        for video_name, meta in self.video_meta.items():
            feat_path = os.path.join(self.feature_path, video_name + self.feature_name_end)
            if not os.path.isfile(feat_path):
                missing_feats.append(video_name)
                continue

            feature_ori = np.load(feat_path)
            if feature_ori.ndim == 3:
                feature_ori = np.mean(feature_ori, axis=1)
            T = feature_ori.shape[0]
            abn_num = max(1, int(self.r * T))

            if self.normprop_scores_dict is not None and video_name in self.normprop_scores_dict:
                Z = self.normprop_scores_dict[video_name]["pseudo_label_scores"]
                score_v = self.normprop_scores_dict[video_name]["score_v"]
                sorted_idxs_F = np.argsort(Z)
                abn_idxs = sorted_idxs_F[:abn_num]
                pseudo_label = np.zeros(T)
                pseudo_label[abn_idxs] = 1
            else:
                Z, pseudo_label, score_v = normality_propagation(feature_ori, abn_num=abn_num, is_ucf=False)

            sample_idxs = self._sample_indices(T)
            pseudo_label = pseudo_label[sample_idxs]
            feature_ori = feature_ori[sample_idxs]
            reweight = np.ones_like(pseudo_label)

            info = {
                "feature": feature_ori,
                "pseudo_label": pseudo_label,
                "reweight": reweight,
                "high_confidence_norvideo": 0,
            }
            self.video_info_dict[video_name] = info
            video_names.append(video_name)
            score_v_list.append(score_v)

        video_names = np.array(video_names)
        score_v_list = np.array(score_v_list)
        if len(video_names) == 0:
            skipped_msg = ""
            if missing_feats:
                skipped_msg = f" | skipped missing features: {len(missing_feats)}"
            self.logger_info = f"No STTAD training videos found.{skipped_msg}"
            return

        abnormal_num_v = max(1, int(len(video_names) * 0.5))
        normal_num_v = max(0, len(video_names) - abnormal_num_v)

        sorted_idxs = np.argsort(score_v_list)
        abn_idx_v = sorted_idxs[-abnormal_num_v:] if abnormal_num_v > 0 else []
        nor_idx_v = sorted_idxs[:normal_num_v] if normal_num_v > 0 else []
        nor_idx_v_high_cofidence = sorted_idxs[: int(normal_num_v * self.h)] if normal_num_v > 0 else []

        for video_name in video_names[nor_idx_v]:
            self.video_info_dict[video_name]["pseudo_label"] = np.zeros_like(
                self.video_info_dict[video_name]["pseudo_label"]
            )

        for video_name in video_names[nor_idx_v_high_cofidence]:
            self.video_info_dict[video_name]["high_confidence_norvideo"] = 1

        skipped_msg = f" | skipped missing features: {len(missing_feats)}" if missing_feats else ""
        self.logger_info = (
            f"Loaded {len(video_names)} STTAD training videos | "
            f"pseudo normal: {len(nor_idx_v)}, pseudo abnormal: {len(abn_idx_v)}"
            f"{skipped_msg}"
        )

    def _build_eval_entries(self):
        missing_feats = []
        for video_name, meta in self.video_meta.items():
            feat_path = os.path.join(self.feature_path, video_name + self.feature_name_end)
            if not os.path.isfile(feat_path):
                missing_feats.append(video_name)
                continue

            feature = np.load(feat_path)
            if feature.ndim == 3:
                feature = np.mean(feature, axis=1)

            snipts_len = feature.shape[0]
            frame_total = max(meta.get("frame_total", 0), snipts_len * self.seg_len)
            target_len = snipts_len * self.seg_len

            if self.eval_train:
                label_test = np.zeros(snipts_len, dtype=np.float32)
            else:
                label_vec = self._label_vector(meta["label_frames"], frame_total, target_len)
                label_test = label_vec.reshape(snipts_len, self.seg_len)

            info = {
                "feature": feature,
                "label_video": 1,
                "label_test": label_test,
            }
            self.video_info_dict[video_name] = info

        skipped_msg = f" | skipped missing features: {len(missing_feats)}" if missing_feats else ""
        self.logger_info = f"Loaded {len(self.video_info_dict)} STTAD eval videos.{skipped_msg}"
