#!/usr/bin/env python3
"""Shared feature extractor utilities for PatchCore-style heatmaps."""

from __future__ import annotations

from typing import Tuple

import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image


class ResNetFeatureExtractor(nn.Module):
    """Pretrained ResNet18 up to layer3."""

    def __init__(self, device: str = "cuda") -> None:
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        backbone.eval()
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        ).to(device)
        self.device = device
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.no_grad()
    def forward(self, img_bgr: "cv2.Mat") -> torch.Tensor:
        """
        Args:
            img_bgr: np.ndarray (H,W,3), BGR order.
        Returns:
            torch.Tensor of shape (C,h,w)
        """
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        pil = T.Resize((256, 256))(pil)
        x = self.transform(pil).unsqueeze(0).to(self.device)
        feat = self.features(x)
        return feat.squeeze(0)


def feat_to_patches(feat: torch.Tensor) -> torch.Tensor:
    """Flatten spatial feature map (C,h,w) into (h*w, C)."""
    C, h, w = feat.shape
    return feat.permute(1, 2, 0).reshape(h * w, C)
