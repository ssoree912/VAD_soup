import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.shanghaitech_frames import ShanghaiTechFrames
from models.unet_ae import UNetAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net autoencoder on ShanghaiTech frames.")
    parser.add_argument(
        "--frames_root",
        type=str,
        default="data/shanghaitech/training/frames",
        help="Directory containing class-normal training frames.",
    )
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="artifacts/ae_unet_shanghaitech",
        help="Directory to save checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ShanghaiTechFrames(args.frames_root, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = UNetAutoencoder(in_ch=3, base_ch=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
        for x in progress:
            x = x.to(device, non_blocking=True)
            x_hat = model(x)

            loss = F.l1_loss(x_hat, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader.dataset)
        print(f"[Epoch {epoch + 1}/{args.epochs}] loss={avg_loss:.4f}")

        ckpt_path = out_dir / f"unet_ae_epoch{epoch + 1}.pth"
        torch.save({"epoch": epoch + 1, "state_dict": model.state_dict()}, ckpt_path)


if __name__ == "__main__":
    main()
