import os
import numpy as np
from data.base_dataset import BaseDataset


class Dataset_SOS(BaseDataset):
    """
    Street Obstacle Sequences (SOS) dataset.
    - Frames live under: data/street_obstacle_sequences/raw_data/sequence_xxx/*.jpg
    - Split files list relative paths like: raw_data/sequence_001,label,frame_len
    - There is no frame-level annotation; we repeat the video label per segment.
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
        self.logger_info = None
        self.video_info_dict = {}

        if self.is_train or (self.is_train is False and self.eval_train is True):
            self.video_list = open(args.training_split, "r").readlines()
        else:
            self.video_list = open(args.testing_split, "r").readlines()

        self.parser_info()

    def parser_info(self):
        if self.is_train:
            video_name_list = []
            score_v_list = []
            gts_list = []
            labels_int = [int(item.strip().split(",")[1]) for item in self.video_list]
            abnormal_num_v = int(np.sum(labels_int))
            normal_num_v = len(self.video_list) - abnormal_num_v

            for i, item in enumerate(self.video_list):
                video_name, video_label, frame_len = item.strip().split(",")
                feat_path = os.path.join(self.feature_path, video_name + self.feature_name_end)
                feature_ori = np.load(feat_path)
                T = feature_ori.shape[0]
                abn_num = int(self.r * T)
                if len(feature_ori.shape) == 3:
                    feature_ori = np.mean(feature_ori, axis=1)

                # No pseudo label scores provided; use normality_propagation fallback
                from data.normprop import normality_propagation

                Z, pseudo_label, score_v = normality_propagation(feature_ori, abn_num=abn_num, is_ucf=False)

                video_name_list.append(video_name)
                score_v_list.append(score_v)
                gt_v = int(video_label)
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

            if len(abn_idx_v) > 0 and len(nor_idx_v) > 0:
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
                self.logger_info = "SOS train stats: normal_v={}, abnormal_v={}".format(normal_num_v, abnormal_num_v)
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
                    label_test = np.zeros((snipts_len, self.seg_len))
                    if int(video_label) == 1:
                        label_test = np.ones((snipts_len, self.seg_len))

                info = {"feature": np.load(feat_path), "label_video": int(video_label), "label_test": label_test}
                self.video_info_dict[video_name] = info
