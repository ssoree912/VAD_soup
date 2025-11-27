from pathlib import Path
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class FramePredictionDataset(Dataset):
    """
    Sliding-window dataset: uses t frames to predict the (t+1)-th frame.
    Assumes directory structure root/<video_id>/*.jpg|png sorted by name.
    """

    def __init__(
        self,
        root_dir: str,
        t: int = 4,
        image_size: int = 256,
        stride: int = 1,
        return_metadata: bool = False,
        recursive: bool = False,
        filter_training_only: bool = False,
        video_list: List[str] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.t = t
        self.return_metadata = return_metadata
        self.video_filter = set(video_list) if video_list else None

        self.videos: List[Tuple[str, List[Path]]] = []
        self.samples: List[Tuple[int, int]] = []  # (video_idx, start_frame_idx)

        if not recursive:
            video_dirs = sorted([p for p in self.root_dir.iterdir() if p.is_dir()])
        else:
            # Gather all leaf dirs; optionally keep only .../training/frames/<vid>
            video_dirs = []
            for d in self.root_dir.rglob("*"):
                if not d.is_dir():
                    continue
                if filter_training_only:
                    parts = d.parts
                    if len(parts) < 3 or parts[-2] != "frames" or parts[-3] != "training":
                        continue
                video_dirs.append(d)

        for vid_idx, vdir in enumerate(sorted(video_dirs)):
            frames = sorted(list(vdir.glob("*.jpg")) + list(vdir.glob("*.png")))
            if len(frames) <= t:
                continue
            # Use relative path as ID in recursive mode to avoid collisions.
            vid_id = str(vdir.relative_to(self.root_dir)) if recursive else vdir.name
            if self.video_filter is not None and vid_id not in self.video_filter:
                continue
            self.videos.append((vid_id, frames))
            for start in range(0, len(frames) - t, stride):
                self.samples.append((vid_idx, start))

        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        vid_idx, start = self.samples[idx]
        vid_name, frames = self.videos[vid_idx]
        ctx_paths = frames[start : start + self.t]
        target_path = frames[start + self.t]

        ctx_imgs = [self.transform(Image.open(p).convert("RGB")) for p in ctx_paths]
        target_img = self.transform(Image.open(target_path).convert("RGB"))

        inp = torch.cat(ctx_imgs, dim=0)  # (3*t, H, W)

        if self.return_metadata:
            return inp, target_img, {
                "video": vid_name,
                "target_idx": start + self.t,
                "target_name": target_path.name,
            }
        return inp, target_img
