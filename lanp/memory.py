from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn


class MemoryModule(nn.Module):
    """
    Reusable implementation of LANP's normality memory buffer.
    """

    def __init__(self, dataset, device):
        super().__init__()
        self.dataset = dataset
        self.device = device

        self.video_name_all = np.array(list(self.dataset.video_info_dict.keys()))
        self.video_num = len(self.video_name_all)
        example_feature = self.dataset.video_info_dict[self.video_name_all[0]]["feature"]

        self.process_len = example_feature.shape[0]
        self.dim = example_feature.shape[-1]

        self.pseudo_labels: Dict[str, torch.Tensor] = {
            video_name: torch.from_numpy(self.dataset.video_info_dict[video_name]["pseudo_label"]).to(torch.float32)
            for video_name in self.video_name_all
        }

        self.video_names_nor_hc = np.array(
            [
                video_name
                for video_name in self.video_name_all
                if self.dataset.video_info_dict[video_name]["high_confidence_norvideo"] == 1
            ]
        )

        self.normal_memory = self.build_memory()
        self.logger_info = f"Memory module built with {len(self.video_names_nor_hc)} high-confidence normal videos."

    def _load_feature_tensor(self, video_name: str) -> torch.Tensor:
        feature = self.dataset.video_info_dict[video_name]["feature"]
        if feature.dtype != np.float32:
            feature = feature.astype(np.float32, copy=False)
        feature_tensor = torch.from_numpy(feature)
        if feature_tensor.dim() == 3:
            feature_tensor = feature_tensor.mean(dim=1)
        return feature_tensor.to(self.device)

    def build_memory(self) -> torch.Tensor:
        memory = []
        for video_name in self.video_names_nor_hc:
            feature_tensor = self._load_feature_tensor(video_name)
            memory.append(torch.mean(feature_tensor, dim=0))
        if len(memory) == 0:
            return torch.zeros(0, self.dim, device=self.device)
        return torch.stack(memory, dim=0)

    def update_dataloader(self) -> bool:
        if self.normal_memory.numel() == 0:
            return False

        dataset_name = self.dataset.dataset_name
        for video_name in self.video_name_all:
            feature_tensor = self._load_feature_tensor(video_name)
            feature_tensor = feature_tensor.unsqueeze(0)

            if dataset_name == "ucf-crime":
                memory_expanded = self.normal_memory.unsqueeze(1).repeat(1, feature_tensor.shape[1], 1)
                feature_expanded = feature_tensor.repeat(self.normal_memory.shape[0], 1, 1)
                dist = nn.CosineSimilarity(dim=-1, eps=1e-6)(feature_expanded, memory_expanded)
                dist = 1 - dist
            else:
                memory_expanded = self.normal_memory.unsqueeze(1)
                feature_expanded = feature_tensor
                diff = feature_expanded - memory_expanded
                dist = torch.pow((torch.sum(diff * diff, dim=-1) / self.dim), 0.5)

            dist, _ = torch.min(dist, dim=0)
            pseudo_label_tensor = self.pseudo_labels[video_name].to(self.device)
            if pseudo_label_tensor.shape[0] != dist.shape[0]:
                pseudo_label_tensor = pseudo_label_tensor[: dist.shape[0]]

            reweight = torch.exp(-torch.abs(dist - pseudo_label_tensor))
            self.dataset.video_info_dict[video_name]["reweight"] = reweight.cpu().numpy()

        return True


# Backwards compatibility alias
Memory_module = MemoryModule

