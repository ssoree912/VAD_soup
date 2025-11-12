from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

try:
    from torchvision.ops import roi_align
except ImportError as exc:  # pragma: no cover
    raise ImportError("torchvision.ops.roi_align is required for ROI pooling") from exc


def roi_align_frames(
    feature_maps: torch.Tensor,
    boxes_per_frame: Sequence[Optional[torch.Tensor]],
    output_size: Tuple[int, int] = (7, 7),
    spatial_scale: float = 1.0,
    sampling_ratio: int = -1,
    aligned: bool = True,
) -> torch.Tensor:
    """
    Applies ROIAlign over a stack of per-frame feature maps.

    Args:
        feature_maps: Tensor with shape (F, C, H, W) where F is the number of frames.
        boxes_per_frame: Iterable of tensors, one per frame, each shaped (N_i, 4) in xyxy.
        output_size: Spatial resolution of pooled features.
        spatial_scale: Scale between input boxes and feature map coordinates.
    """
    if feature_maps.dim() != 4:
        raise ValueError("feature_maps must be shaped (frames, channels, height, width).")
    device = feature_maps.device
    rois: List[torch.Tensor] = []
    for frame_idx, boxes in enumerate(boxes_per_frame):
        if boxes is None or len(boxes) == 0:
            continue
        box_tensor = torch.as_tensor(boxes, dtype=torch.float32, device=device)
        idx = torch.full((box_tensor.shape[0], 1), frame_idx, dtype=torch.float32, device=device)
        rois.append(torch.cat([idx, box_tensor], dim=1))
    if not rois:
        return torch.zeros((0, feature_maps.shape[1], *output_size), device=device)
    rois_tensor = torch.cat(rois, dim=0)
    pooled = roi_align(
        feature_maps,
        rois_tensor,
        output_size=output_size,
        spatial_scale=spatial_scale,
        sampling_ratio=sampling_ratio,
        aligned=aligned,
    )
    return pooled


def roi_feature_vectors(
    feature_maps: torch.Tensor,
    boxes_per_frame: Sequence[Optional[torch.Tensor]],
    output_size: Tuple[int, int] = (7, 7),
    spatial_scale: float = 1.0,
    pooling: str = "avg",
) -> torch.Tensor:
    pooled = roi_align_frames(feature_maps, boxes_per_frame, output_size, spatial_scale)
    pooling = pooling.lower()
    if pooling == "avg":
        vectors = torch.mean(pooled.view(pooled.shape[0], pooled.shape[1], -1), dim=-1)
    elif pooling == "max":
        vectors, _ = torch.max(pooled.view(pooled.shape[0], pooled.shape[1], -1), dim=-1)
    else:
        raise ValueError(f"Unsupported pooling '{pooling}'.")
    return vectors

