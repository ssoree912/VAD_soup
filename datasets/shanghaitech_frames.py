from pathlib import Path
from typing import List

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class ShanghaiTechFrames(Dataset):
    """Loads all normal frames under the given root directory."""

    def __init__(self, root_dir: str, image_size: int = 256) -> None:
        self.root_dir = Path(root_dir)
        jpg_paths: List[Path] = list(self.root_dir.rglob("*.jpg"))
        png_paths: List[Path] = list(self.root_dir.rglob("*.png"))
        self.image_paths = sorted(jpg_paths + png_paths)

        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img)
