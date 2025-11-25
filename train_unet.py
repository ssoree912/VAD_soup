import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Iterable, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from datasets import FramePredictionDataset
from models.att_unet import AttUNetPredictor

#예측 이미지와 타겟 이미지의 엣지 정보가 얼마나 다른지 측정 
def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 difference between gradient magnitudes (x,y)."""
    dx_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :] #x방향 그래디언트 계산
    dy_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1] #y방향 그래디언트 계산

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
    parser.add_argument("--frames_root", type=str, default=None,
                        help="Root with training frames (video folders with images).")
    parser.add_argument("--videos_root", type=str, default=None,
                        help="If frames are not pre-extracted, point to video root to auto-extract.")
    parser.add_argument("--t", type=int, default=4, help="Number of context frames.")
    parser.add_argument("--stride", type=int, default=1, help="Sliding-window stride between samples.")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_frames_root", type=str, default=None,
                        help="Optional validation frames root for best-epoch selection.")
    parser.add_argument("--val_videos_root", type=str, default=None,
                        help="If val frames are absent, point to validation video root.")
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
    parser.add_argument("--resume", type=str, default=None,
                        help="Optional path to a checkpoint (.pth) to resume training from.")
    parser.add_argument("--video_exts", nargs="+", default=[".mp4", ".mov", ".avi"],
                        help="Video extensions to scan when extracting frames.")
    parser.add_argument("--temp_dir", type=str, default=None,
                        help="Optional temp dir for frame extraction (default: create under system temp).")
    return parser.parse_args()


def iter_video_files(root: Path, exts: Iterable[str]) -> Iterable[Path]:
    exts_norm = {e.lower() for e in exts}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in exts_norm:
            yield path


def ensure_frames_root(frames_root: Optional[Path], videos_root: Optional[Path], exts: List[str],
                       temp_dir: Optional[Path]) -> Path:
    """
    Ensure frames_root exists and contains frames. If not, extract from videos_root using ffmpeg.
    """
    if frames_root is not None:
        frames_root = frames_root.resolve()
    if videos_root is not None:
        videos_root = videos_root.resolve()

    if frames_root and frames_root.exists():
        # Check if frames already present
        has_frames = any(frames_root.rglob("*.jpg")) or any(frames_root.rglob("*.png"))
        if has_frames:
            return frames_root

    if videos_root is None:
        raise ValueError("frames_root is empty/missing and videos_root was not provided for extraction.")

    target_root = frames_root if frames_root is not None else (temp_dir or Path(tempfile.mkdtemp(prefix="unet_frames_")))
    target_root = target_root.resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    for vid_file in iter_video_files(videos_root, exts):
        rel = vid_file.relative_to(videos_root).with_suffix("")
        out_dir = target_root / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-i", str(vid_file),
            "-qscale:v", "2",
            "-vsync", "0",
            str(out_dir / "%05d.jpg"),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return target_root


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Enable cuDNN autotuner for fixed-size inputs on GPU
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    train_frames_root = ensure_frames_root(
        Path(args.frames_root) if args.frames_root else None,
        Path(args.videos_root) if args.videos_root else None,
        args.video_exts,
        Path(args.temp_dir) if args.temp_dir else None,
    )

    dataset = FramePredictionDataset(
        str(train_frames_root),
        t=args.t,
        image_size=args.image_size,
        stride=args.stride,
    )
    if len(dataset) == 0:
        raise ValueError(f"No training samples found under {train_frames_root} with t={args.t}")

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0 and device.type == "cuda"),
    )

    val_loader: Optional[DataLoader] = None
    if args.val_frames_root or args.val_videos_root:
        val_frames_root = ensure_frames_root(
            Path(args.val_frames_root) if args.val_frames_root else None,
            Path(args.val_videos_root) if args.val_videos_root else None,
            args.video_exts,
            Path(args.temp_dir) if args.temp_dir else None,
        )
        val_dataset = FramePredictionDataset(
            str(val_frames_root),
            t=args.t,
            image_size=args.image_size,
            stride=args.stride,
        )
        if len(val_dataset) == 0:
            raise ValueError(f"No validation samples found under {val_frames_root} with t={args.t}")
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size or args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
            persistent_workers=(args.num_workers > 0 and device.type == "cuda"),
        )

    model = AttUNetPredictor(t=args.t, base_ch=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_epoch = -1
    start_epoch = 0

    resume_path = Path(args.resume) if args.resume is not None else None
    if resume_path is not None and resume_path.is_file():
        ckpt = torch.load(resume_path, map_location=device)
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and ckpt["scaler"] is not None and isinstance(scaler, GradScaler):
            scaler.load_state_dict(ckpt["scaler"])
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"])
        if "best_val" in ckpt:
            best_val = float(ckpt["best_val"])
        if "best_epoch" in ckpt:
            best_epoch = int(ckpt["best_epoch"])
        print(f"[resume] Loaded checkpoint from {resume_path} (epoch={start_epoch})")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
        for x, target in progress:
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                pred = model(x)
                loss = loss_fn(pred, target, args.alpha_rec, args.alpha_gra)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
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
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict() if use_amp else None,
                        "args": vars(args),
                        "val_loss": val_loss,
                        "best_val": best_val,
                        "best_epoch": best_epoch,
                    },
                    out_dir / "att_unet_best.pth",
                )

        print(log_msg)

        ckpt_path = out_dir / f"att_unet_epoch{epoch + 1}.pth"
        torch.save(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if use_amp else None,
                "args": vars(args),
                "val_loss": val_loss,
                "best_val": best_val,
                "best_epoch": best_epoch,
            },
            ckpt_path,
        )

    if val_loader is not None and best_epoch > 0:
        print(f"[best] epoch={best_epoch} val_loss={best_val:.5f} -> {out_dir/'att_unet_best.pth'}")


if __name__ == "__main__":
    main()
