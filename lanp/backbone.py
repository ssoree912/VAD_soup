from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.models.video as tv_video
import torchvision.transforms as T
from PIL import Image

KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD = [0.22803, 0.22145, 0.216989]

BACKBONE_REPO_ROOT = Path(__file__).resolve().parents[1] / "video-classification-3d-cnn-pytorch"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _load_video_model(
    arch: str,
    pretrained: bool,
    weights_path: Optional[Union[str, Path]],
) -> nn.Module:
    if not hasattr(tv_video, arch):
        raise ValueError(f"Unsupported video backbone '{arch}'.")

    builder = getattr(tv_video, arch)
    try:
        # torchvision>=0.15 exposes weights arg
        weight_arg = "DEFAULT" if pretrained else None
        model = builder(weights=weight_arg)
    except TypeError:
        model = builder(pretrained=pretrained)

    if weights_path:
        state = torch.load(str(weights_path), map_location="cpu")
        # allow checkpoints saved with {"state_dict": ...}
        if "state_dict" in state and not isinstance(state["state_dict"], torch.Tensor):
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)

    # Strip classification head to expose penultimate features.
    if hasattr(model, "fc"):
        model.fc = nn.Identity()
    if hasattr(model, "classifier"):
        model.classifier = nn.Identity()

    model.eval()
    return model


class TorchvisionBackboneExtractor:
    """
    Lightweight wrapper around torchvision video backbones (e.g., r3d_18).
    """

    def __init__(
        self,
        arch: str = "r3d_18",
        pretrained: bool = False,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        resize_hw: Tuple[int, int] = (112, 112),
    ):
        self.device = _resolve_device(device)
        self.model = _load_video_model(arch, pretrained, weights_path).to(self.device)
        self.resize_hw = resize_hw
        self.transform = T.Compose(
            [
                T.Resize(self.resize_hw, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=KINETICS_MEAN, std=KINETICS_STD),
            ]
        )

    @staticmethod
    def _to_pil(frame: Union[np.ndarray, Image.Image]) -> Image.Image:
        if isinstance(frame, Image.Image):
            return frame
        if isinstance(frame, np.ndarray):
            arr = frame
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=2)
            if arr.shape[2] == 3:
                arr = arr[..., ::-1]  # BGR -> RGB
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)
        raise TypeError(f"Unsupported frame type: {type(frame)}")

    def preprocess_clip(self, frames: Sequence[Union[np.ndarray, Image.Image]]) -> torch.Tensor:
        if not frames:
            raise ValueError("At least one frame is required to build a clip.")
        tensors: List[torch.Tensor] = []
        for frame in frames:
            pil = self._to_pil(frame)
            tensors.append(self.transform(pil))
        clip = torch.stack(tensors, dim=1)  # (C, T, H, W)
        return clip

    def extract_clip(self, frames: Sequence[Union[np.ndarray, Image.Image]]) -> torch.Tensor:
        clip = self.preprocess_clip(frames).unsqueeze(0).to(self.device, non_blocking=True)
        with torch.no_grad():
            feat = self.model(clip)
        return feat.squeeze(0).detach().cpu()

    def extract_sequence(
        self,
        frames: Sequence[Union[np.ndarray, Image.Image]],
        clip_len: int,
        stride: int,
    ) -> torch.Tensor:
        """
        Slide a temporal window across a sequence and stack embeddings.
        """
        if clip_len <= 0:
            raise ValueError("clip_len must be > 0")
        if stride <= 0:
            raise ValueError("stride must be > 0")

        features: List[torch.Tensor] = []
        num_frames = len(frames)
        for start in range(0, max(1, num_frames - clip_len + 1), stride):
            end = start + clip_len
            window = frames[start:end]
            if len(window) < clip_len:
                # pad by repeating last frame
                last = window[-1]
                window = list(window) + [last] * (clip_len - len(window))
            feat = self.extract_clip(window)
            features.append(feat)
        if not features:
            return torch.zeros((0,), dtype=torch.float32)
        return torch.stack(features, dim=0)

    def freeze_bn(self):
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()

    def forward_raw(self, clip_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward a pre-built clip tensor of shape (1, C, T, H, W).
        """
        if clip_tensor.dim() != 5:
            raise ValueError("Expected clip tensor with shape (N, C, T, H, W).")
        with torch.no_grad():
            feat = self.model(clip_tensor.to(self.device))
        return feat.detach().cpu()


def _load_backbone_module(module_label: str, relative_path: str, repo_root: Path) -> object:
    module_name = f"_lanp_backbone_{module_label}"
    module_path = repo_root / relative_path
    if not module_path.exists():
        raise FileNotFoundError(f"Expected backbone file not found: {module_path}")
    # Ensure the backbone repo is at the front of sys.path so its local `models`
    # package is resolved instead of any top-level `models` in this project.
    repo_root_str = str(repo_root)
    added_sys_path = False
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
        added_sys_path = True

    # Drop any previously-imported top-level `models` that point outside the backbone repo
    # to avoid import cache collisions.
    removed_modules = {}
    for name, mod in list(sys.modules.items()):
        if name == "models" or name.startswith("models."):
            mod_path = getattr(mod, "__file__", "") or ""
            if repo_root_str not in mod_path:
                removed_modules[name] = sys.modules.pop(name)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load backbone module {module_label} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[assignment]
    finally:
        if added_sys_path and repo_root_str in sys.path:
            sys.path.remove(repo_root_str)
        sys.modules.update(removed_modules)
    return module


class LANPResNeXtBackbone:
    """
    Wrapper around the ResNeXt-101 backbone used by the original LANP paper.
    """

    def __init__(
        self,
        weights_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        sample_duration: int = 16,
        sample_size: int = 112,
        model_name: str = "resnext",
        model_depth: int = 101,
        resnext_cardinality: int = 32,
        resnet_shortcut: str = "B",
        repo_root: Optional[Union[str, Path]] = None,
    ) -> None:
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Backbone weights not found: {weights_path}")
        self.device = _resolve_device(device)
        repo_root = Path(repo_root) if repo_root else BACKBONE_REPO_ROOT

        backbone_mean_module = _load_backbone_module("mean", "mean.py", repo_root)
        backbone_model_module = _load_backbone_module("model", "model.py", repo_root)

        spatial_transforms_module = _load_backbone_module("spatial_transforms", "spatial_transforms.py", repo_root)
        self.CenterCrop = spatial_transforms_module.CenterCrop  # type: ignore[attr-defined]
        self.Compose = spatial_transforms_module.Compose  # type: ignore[attr-defined]
        self.Normalize = spatial_transforms_module.Normalize  # type: ignore[attr-defined]
        self.Scale = spatial_transforms_module.Scale  # type: ignore[attr-defined]
        self.ToTensor = spatial_transforms_module.ToTensor  # type: ignore[attr-defined]

        generate_backbone = backbone_model_module.generate_model  # type: ignore[attr-defined]
        backbone_mean = backbone_mean_module.get_mean  # type: ignore[attr-defined]

        opt = SimpleNamespace()
        opt.model_name = model_name
        opt.model_depth = model_depth
        opt.arch = f"{opt.model_name}-{opt.model_depth}"
        opt.resnext_cardinality = resnext_cardinality
        opt.resnet_shortcut = resnet_shortcut
        opt.n_classes = 400
        opt.sample_size = sample_size
        opt.sample_duration = sample_duration
        opt.mode = "feature"
        opt.mean = backbone_mean()
        opt.batch_size = 1
        opt.n_threads = 1
        opt.no_cuda = self.device.type == "cpu"

        self.opt = opt
        self.model = generate_backbone(opt)
        state = torch.load(str(weights_path), map_location=self.device)
        state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state

        model_is_parallel = isinstance(self.model, torch.nn.DataParallel)
        state_is_parallel = any(k.startswith("module.") for k in state_dict.keys())

        if model_is_parallel and not state_is_parallel:
            state_dict = {f"module.{k}": v for k, v in state_dict.items()}
        if not model_is_parallel and state_is_parallel:
            state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        if not opt.no_cuda:
            self.model.to(self.device)

        self.sample_duration = sample_duration
        self.spatial_transform = self.Compose(
            [
                self.Scale(opt.sample_size),
                self.CenterCrop(opt.sample_size),
                self.ToTensor(),
                self.Normalize(opt.mean, [1.0, 1.0, 1.0]),
            ]
        )
        # Default fill color for occlusion: dataset mean in RGB (0-255 scale).
        self.fill_rgb = np.array(opt.mean, dtype=np.float32)

    def _pad_frames(self, frames: Sequence[np.ndarray]) -> List[np.ndarray]:
        if len(frames) >= self.sample_duration:
            return list(frames[: self.sample_duration])
        padded = list(frames)
        while len(padded) < self.sample_duration:
            padded.append(padded[-1])
        return padded

    def extract(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        frames = self._pad_frames(frames)
        clip_tensors = []
        for frame in frames:
            pil_image = Image.fromarray(frame, mode="RGB")
            clip_tensors.append(self.spatial_transform(pil_image))
        clip_tensor = torch.stack(clip_tensors, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
        clip_tensor = clip_tensor.to(self.device).type(torch.float32)

        with torch.no_grad():
            outputs = self.model(clip_tensor)

        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        if outputs.dim() > 2:
            outputs = outputs.view(outputs.size(0), -1)

        feature = outputs.squeeze(0).detach().cpu().numpy()
        return feature.astype(np.float32, copy=False)


class FrameFeatureBackbone:
    """2D CNN backbone that exposes convolutional feature maps per frame."""

    def __init__(
        self,
        arch: str = "resnet50",
        pretrained: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.device = _resolve_device(device)
        builder = getattr(tv_models, arch, None)
        if builder is None:
            raise ValueError(f"Unsupported 2D backbone '{arch}'.")
        try:
            model = builder(weights="IMAGENET1K_V2" if pretrained else None)
        except TypeError:
            model = builder(pretrained=pretrained)
        modules = list(model.children())[:-2]
        self.feature_extractor = nn.Sequential(*modules).to(self.device)
        self.feature_extractor.eval()
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def _prepare_tensor(self, frame_rgb: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        if isinstance(frame_rgb, np.ndarray):
            arr = frame_rgb
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=2)
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            pil = Image.fromarray(arr.astype(np.uint8))
        elif isinstance(frame_rgb, Image.Image):
            pil = frame_rgb
        else:
            raise TypeError(f"Unsupported frame type: {type(frame_rgb)}")
        tensor = self.transform(pil).unsqueeze(0).to(self.device)
        return tensor

    def extract_feature_map(self, frame_rgb: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        tensor = self._prepare_tensor(frame_rgb)
        with torch.no_grad():
            fmap = self.feature_extractor(tensor)
        return fmap.squeeze(0).detach().cpu()

    def extract_batch(self, frames: Sequence[Union[np.ndarray, Image.Image]]) -> torch.Tensor:
        if not frames:
            return torch.zeros((0,), dtype=torch.float32)
        tensors = torch.cat([self._prepare_tensor(frame) for frame in frames], dim=0)
        with torch.no_grad():
            fmap = self.feature_extractor(tensors)
        return fmap.detach().cpu()


# Backwards compatibility alias for modules that previously imported BackboneFeatureExtractor
BackboneFeatureExtractor = TorchvisionBackboneExtractor
