import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import FramePredictionDataset
from models.att_unet_predictor import AttUNetPredictor


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 difference between gradient magnitudes (x,y)."""
    dx_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]

    dx_tgt = target[:, :, 1:, :] - target[:, :, :-1, :]
    dy_tgt = target[:, :, :, 1:] - target[:, :, :, :-1]

    loss_x = (dx_pred.abs() - dx_tgt.abs()).abs().mean()
    loss_y = (dy_pred.abs() - dy_tgt.abs()).abs().mean()
    return loss_x + loss_y


def loss_fn(pred: torch.Tensor, target: torch.Tensor, alpha_rec: float, alpha_gra: float) -> torch.Tensor:
    l_rec = F.mse_loss(pred, target)
    l_gra = gradient_loss(pred, target)
    return alpha_rec * l_rec + alpha_gra * l_gra


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Attention U-Net frame predictor (t frames -> next frame).")
    parser.add_argument("--frames_root", type=str, default="data/shanghaitech/training/frames",
                        help="Root with training videos (normal only).")
    parser.add_argument("--t", type=int, default=4, help="Number of context frames.")
    parser.add_argument("--stride", type=int, default=1, help="Sliding-window stride between samples.")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_frames_root", type=str, default=None,
                        help="Optional validation frames root for best-epoch selection.")
    parser.add_argument("--val_batch_size", type=int, default=None,
                        help="Validation batch size (default: same as train).")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha_rec", type=float, default=1.0)
    parser.add_argument("--alpha_gra", type=float, default=1.0)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--out_dir", type=str, default="artifacts/att_unet_predictor",
                        help="Directory to save checkpoints.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FramePredictionDataset(
        args.frames_root,
        t=args.t,
        image_size=args.image_size,
        stride=args.stride,
    )
    if len(dataset) == 0:
        raise ValueError(f"No training samples found under {args.frames_root} with t={args.t}")

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader: Optional[DataLoader] = None
    if args.val_frames_root:
        val_dataset = FramePredictionDataset(
            args.val_frames_root,
            t=args.t,
            image_size=args.image_size,
            stride=args.stride,
        )
        if len(val_dataset) == 0:
            raise ValueError(f"No validation samples found under {args.val_frames_root} with t={args.t}")
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size or args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    model = AttUNetPredictor(t=args.t, base_ch=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_epoch = -1

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
        for x, target in progress:
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            pred = model(x)
            loss = loss_fn(pred, target, args.alpha_rec, args.alpha_gra)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / max(1, total_samples)
        log_msg = f"[Epoch {epoch + 1}/{args.epochs}] train_loss={avg_loss:.5f}"

        val_loss = None
        if val_loader is not None:
            model.eval()
            val_total = 0.0
            val_samples = 0
            with torch.no_grad():
                for x, target in val_loader:
                    x = x.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    pred = model(x)
                    loss = loss_fn(pred, target, args.alpha_rec, args.alpha_gra)
                    bs = x.size(0)
                    val_total += loss.item() * bs
                    val_samples += bs
            val_loss = val_total / max(1, val_samples)
            log_msg += f" | val_loss={val_loss:.5f}"
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch + 1
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "args": vars(args),
                        "val_loss": val_loss,
                    },
                    out_dir / "att_unet_best.pth",
                )

        print(log_msg)

        ckpt_path = out_dir / f"att_unet_epoch{epoch + 1}.pth"
        torch.save(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "args": vars(args),
                "val_loss": val_loss,
            },
            ckpt_path,
        )

    if val_loader is not None and best_epoch > 0:
        print(f"[best] epoch={best_epoch} val_loss={best_val:.5f} -> {out_dir/'att_unet_best.pth'}")


if __name__ == "__main__":
    main()
