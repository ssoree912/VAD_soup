import glob
import os
from typing import Dict, List, Tuple

import numpy as np

from data.base_dataset import BaseDataset
from data.normprop import normality_propagation


def _video_to_label_path(anno_root: str, video_name: str) -> str:
    """
    video_name 예: S01/testing/frames/10
    라벨 파일: data/IPAD/S01/test_label/010.npy (3자리 zero-pad)
    """
    parts = video_name.split("/")
    scenario = parts[0]
    vid_leaf = parts[-1]
    try:
        vid_int = int(vid_leaf)
    except ValueError:
        vid_int = int("".join(filter(str.isdigit, vid_leaf)))
    fname = f"{vid_int:03d}.npy"
    return os.path.join(anno_root, scenario, "test_label", fname)


class Dataset_IPAD(BaseDataset):
    """
    IPAD 특징:
      - 스플릿 파일의 video_name은 Sxx/... 형태의 상대경로.
      - 테스트 라벨은 시나리오별 test_label/*.npy (프레임 단위 0/1).
    """

    def initialize(self, args, sample_type: str = "uniform", is_train: bool = True, is_normal: bool = True, eval_train: bool = False):
        self.dataset_name = args.dataset
        self.seg_len = args.segment_len
        self.normprop_scores_path = args.np_scores_path
        self.process_len = args.process_len
        self.sample_type = sample_type
        self.is_train = is_train
        self.is_normal = is_normal
        self.eval_train = eval_train
        self.r = args.r
        self.h = args.h
        self.feature_path = args.feature_path
        self.feature_name_end = args.feature_name_end
        self.anno_root = args.anno_root
        self.logger_info = None
        self.video_info_dict: Dict[str, Dict] = {}
        self.test_anno_cache: Dict[str, np.ndarray] = {}

        if self.is_train or (self.is_train is False and self.eval_train is True):
            self.video_list = open(args.training_split, "r").readlines()
        else:
            self.video_list = open(args.testing_split, "r").readlines()

        if self.normprop_scores_path is not None:
            self.normprop_scores_dict = np.load(self.normprop_scores_path, allow_pickle=True).item()
        else:
            self.normprop_scores_dict = None

        self.parser_info()

    def _load_test_annotation(self, video_name: str) -> np.ndarray:
        if video_name in self.test_anno_cache:
            return self.test_anno_cache[video_name]
        label_path = _video_to_label_path(self.anno_root, video_name)
        if not os.path.isfile(label_path):
            # 없으면 전부 정상으로 가정
            return np.zeros((1,), dtype=np.float32)
        anno_ = np.load(label_path)
        if len(anno_) % self.seg_len == 0:
            annotation = anno_
        else:
            anno_last = anno_[-1]
            annotation = list(anno_)
            for _ in range(len(anno_), (len(anno_) // self.seg_len + 1) * self.seg_len):
                annotation.append(anno_last)
            annotation = np.array(annotation)
        self.test_anno_cache[video_name] = annotation.astype(np.float32)
        return self.test_anno_cache[video_name]

    def parser_info(self):
        if self.is_train:
            video_name_list: List[str] = []
            score_v_list: List[float] = []
            gts_list: List[int] = []

            abnormal_num_v = int(len(self.video_list) * 0.27)
            normal_num_v = len(self.video_list) - abnormal_num_v

            for item in self.video_list:
                video_name, _, frame_len = item.strip().split(",")
                feat_path = os.path.join(self.feature_path, video_name + self.feature_name_end)
                feature_ori = np.load(feat_path)
                T = feature_ori.shape[0]
                abn_num = int(self.r * T)
                if len(feature_ori.shape) == 3:
                    feature_ori = np.mean(feature_ori, axis=1)
                if self.normprop_scores_dict is not None:
                    Z = self.normprop_scores_dict[video_name]["pseudo_label_scores"]
                    score_v = self.normprop_scores_dict[video_name]["score_v"]
                    sorted_idxs_F = np.argsort(Z)
                    abn_idxs = sorted_idxs_F[:abn_num]
                    pseudo_label = np.zeros(T)
                    pseudo_label[abn_idxs] = 1
                else:
                    Z, pseudo_label, score_v = normality_propagation(feature_ori, abn_num=abn_num, is_ucf=False)

                video_name_list.append(video_name)
                score_v_list.append(score_v)
                gt_v = int(item.strip().split(",")[1])
                gts_list.append(gt_v)

                sample_idxs = self.uniform_sampling(feature_ori.shape[0])
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

            video_name_list = np.array(video_name_list)
            score_v_list = np.array(score_v_list)
            gts_list = np.array(gts_list)
            sorted_idxs = np.argsort(score_v_list)  # from small to large
            abn_idx_v = sorted_idxs[-abnormal_num_v:]
            nor_idx_v = sorted_idxs[:normal_num_v]

            nor_idx_v_high_confidence = sorted_idxs[: int(normal_num_v * self.h)]
            nor_video_names = video_name_list[nor_idx_v]
            nor_video_names_high_confidence = video_name_list[nor_idx_v_high_confidence]

            for video_name in nor_video_names:
                self.video_info_dict[video_name]["pseudo_label"] = np.zeros(
                    len(self.video_info_dict[video_name]["pseudo_label"])
                )

            for video_name in nor_video_names_high_confidence:
                self.video_info_dict[video_name]["high_confidence_norvideo"] = 1

            pred_video = np.array([1] * len(abn_idx_v) + [0] * len(nor_idx_v))
            gt_video = np.concatenate((gts_list[abn_idx_v], gts_list[nor_idx_v]))

            tp = np.sum((pred_video) * (gt_video))
            tn = np.sum((1 - pred_video) * (1 - gt_video))
            fp = np.sum(pred_video * (1 - gt_video))
            fn = np.sum((1 - pred_video) * (gt_video))

            self.logger_info = "tp: {}/{:.2f}, tn: {}/{:.2f}, fp: {}, fn: {}".format(
                tp, tp / len(abn_idx_v), tn, tn / len(nor_idx_v), fp, fn
            )
        else:
            for item in self.video_list:
                video_name, video_label, frame_len = item.strip().split(",")
                feat_path = os.path.join(self.feature_path, video_name + self.feature_name_end)
                frame_len = int(frame_len)
                if frame_len % self.seg_len == 0:
                    snipts_len = frame_len // self.seg_len
                else:
                    snipts_len = frame_len // self.seg_len + 1

                if self.eval_train:
                    label_test = np.zeros(snipts_len)
                else:
                    if int(video_label) == 0:
                        label_test = np.zeros((snipts_len, self.seg_len))
                    else:
                        test_anno = self._load_test_annotation(video_name)
                        label_test = np.array(
                            [test_anno[i : i + self.seg_len] for i in np.arange(0, len(test_anno), self.seg_len)]
                        )

                info = {"feature": np.load(feat_path), "label_video": int(video_label), "label_test": label_test}
                self.video_info_dict[video_name] = info
