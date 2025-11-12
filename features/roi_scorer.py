from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F


class ROIScorer:
    """
    Compute object-level anomaly scores by comparing ROI features against the LANP
    normality memory using cosine distance.
    """

    def __init__(self, memory: torch.Tensor, normalize: bool = True, eps: float = 1e-6):
        if memory.dim() != 2:
            raise ValueError("Memory tensor must be shaped (M, D).")
        self.eps = eps
        self.device = memory.device
        self.normalize = normalize
        self.memory = (
            F.normalize(memory, p=2, dim=-1, eps=self.eps) if normalize else memory.clone()
        )

    def score(self, roi_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            roi_features: Tensor of shape (N, D)
        Returns:
            Tensor of shape (N,) containing cosine distances (1 - max cosine sim).
        """
        if roi_features.numel() == 0:
            return torch.zeros((roi_features.shape[0],), device=self.device)

        feats = roi_features.to(self.device)
        if self.normalize:
            feats = F.normalize(feats, p=2, dim=-1, eps=self.eps)

        sims = torch.matmul(feats, self.memory.t())
        max_sim, _ = torch.max(sims, dim=1)
        scores = 1.0 - max_sim
        return scores


def score_frames(
    scorer: ROIScorer,
    roi_features_per_frame: Sequence[torch.Tensor],
) -> List[torch.Tensor]:
    """
    Convenience function to score multiple frames worth of detections.
    """
    outputs: List[torch.Tensor] = []
    for feats in roi_features_per_frame:
        if feats is None or feats.numel() == 0:
            outputs.append(torch.zeros((0,), device=scorer.device))
            continue
        outputs.append(scorer.score(feats))
    return outputs


def to_numpy_dict(frame_scores: Dict[str, List[torch.Tensor]]) -> Dict[str, List[np.ndarray]]:
    """
    Convert a nested dict of torch tensors into numpy arrays for serialization.
    """
    result: Dict[str, List[np.ndarray]] = {}
    for video, per_frame in frame_scores.items():
        result[video] = [score.detach().cpu().numpy().astype(np.float32, copy=False) for score in per_frame]
    return result

