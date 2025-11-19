import argparse
from pathlib import Path
from typing import Optional, Set

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from models.unet_ae import UNetAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AE heatmaps for ShanghaiTech frames.")
    parser.add_argument("--frames_root", type=str, default="data/shanghaitech/testing/frames")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained AE checkpoint.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="artifacts/ae_unet_shanghaitech/heatmaps",
        help="Directory to store .npy/.png outputs.",
    )
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument(
        "--frame_list",
        type=str,
        default=None,
        help="Optional text file listing relative frame paths (video/frame.jpg) to process.",
    )
    return parser.parse_args()


def load_model(ckpt_path: str, device: torch.device) -> UNetAutoencoder:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = UNetAutoencoder(in_ch=3, base_ch=64).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def maybe_load_frame_list(path: Optional[str]) -> Optional[Set[str]]:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        entries = {line.strip() for line in f if line.strip()}
    return entries


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.ckpt, device)
    frames_root = Path(args.frames_root)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    allow_list = maybe_load_frame_list(args.frame_list)

    transform = T.Compose(
        [
            T.Resize((args.image_size, args.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    video_dirs = sorted([p for p in frames_root.iterdir() if p.is_dir()])
    for video_dir in video_dirs:
        video_out = out_root / video_dir.name
        video_out.mkdir(parents=True, exist_ok=True)

        frame_paths = sorted(list(video_dir.glob("*.jpg")) + list(video_dir.glob("*.png")))
        for fp in frame_paths:
            rel_key = f"{video_dir.name}/{fp.name}"
            if allow_list is not None and rel_key not in allow_list:
                continue

            img = Image.open(fp).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                x_hat = model(x)

            x_denorm = (x * 0.5 + 0.5).clamp(0, 1)
            x_hat_denorm = (x_hat * 0.5 + 0.5).clamp(0, 1)

            err_map = torch.sqrt(((x_denorm - x_hat_denorm) ** 2).sum(dim=1, keepdim=True))
            err_map_np = err_map.squeeze().cpu().numpy()

            np.save(video_out / f"{fp.stem}_err.npy", err_map_np)

            em_norm = (err_map_np - err_map_np.min()) / (err_map_np.max() - err_map_np.min() + 1e-8)
            em_uint8 = (em_norm * 255).astype(np.uint8)
            Image.fromarray(em_uint8).save(video_out / f"{fp.stem}_err.png")


if __name__ == "__main__":
    main()
