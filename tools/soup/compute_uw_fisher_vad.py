#!/usr/bin/env python3
"""
Uncertainty-weighted Fisher extractor for VAD checkpoints (UGM-ready).

Example:
python tools/soup/compute_uw_fisher_vad.py \
  --config config/config_ucf.yaml \
  --checkpoints ckpts/model1.pth ckpts/model2.pth \
  --output_dir results/fisher_uw \
  --device cuda --gpu_id 0 --max_batches 80 --uw_gamma 1.0
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import yaml
from torch.utils.data import DataLoader, SequentialSampler

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset_loader import CreateDataset
from model import AD_Model
from tools.soup.fisher_vad import FisherVAD
from tools.soup.uncertainty_vad import EpistemicUncertaintyVAD
from utils import set_seeds


def setup_logger(log_level: str) -> logging.Logger:
    logger = logging.getLogger("compute_uw_fisher_vad")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.propagate = False
    return logger


def load_config(config_path: str) -> argparse.Namespace:
    with open(config_path, "r") as handle:
        cfg = yaml.load(handle, Loader=yaml.FullLoader)
    return argparse.Namespace(**cfg)


def build_sequential_loader(orig_loader: DataLoader) -> DataLoader:
    """Rebuild a DataLoader with deterministic ordering."""
    dataset = getattr(orig_loader, "dataset", None)
    if dataset is None:
        raise RuntimeError("Original loader has no dataset; cannot build sequential loader.")

    sampler = SequentialSampler(dataset)
    kwargs = dict(
        batch_size=orig_loader.batch_size,
        sampler=sampler,
        num_workers=orig_loader.num_workers,
        pin_memory=getattr(orig_loader, "pin_memory", False),
        drop_last=getattr(orig_loader, "drop_last", False),
    )
    collate_fn = getattr(orig_loader, "collate_fn", None)
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn
    prefetch_factor = getattr(orig_loader, "prefetch_factor", None)
    if prefetch_factor is not None and kwargs["num_workers"] > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    if getattr(orig_loader, "persistent_workers", False) and kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = True

    return DataLoader(dataset, **kwargs)


def load_models(ckpt_paths: List[str], feature_dim: int, dropout_rate: float,
                device: torch.device, logger: logging.Logger) -> List[AD_Model]:
    models: List[AD_Model] = []
    for idx, ckpt_path in enumerate(ckpt_paths):
        logger.info("Loading checkpoint %d/%d: %s", idx + 1, len(ckpt_paths), ckpt_path)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        state_dict = torch.load(ckpt_path, map_location=device)
        model = AD_Model(feature_dim, 512, dropout_rate)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        models.append(model)
    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute uncertainty-weighted Fisher matrices for VAD experiments."
    )
    parser.add_argument("--config", required=True, help="Training config YAML used for the checkpoints.")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint paths to process.")
    parser.add_argument("--output_dir", required=True, help="Directory to save UW-Fisher files.")
    parser.add_argument("--device", default=None, help="Device override (cpu / cuda / mps).")
    parser.add_argument("--gpu_id", type=int, default=None, help="GPU index when using CUDA.")
    parser.add_argument("--max_batches", type=int, default=100, help="Max batches for uncertainty/Fisher computation.")
    parser.add_argument("--fisher_floor", type=float, default=1e-8, help="Clamp Fisher entries to at least this value.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--uw_var_unbiased", action="store_true", help="Use unbiased variance for epistemic uncertainty.")
    parser.add_argument("--uw_shrink_alpha", type=float, default=0.0, help="Shrink variance toward mean [0,1].")
    parser.add_argument("--uw_gamma", type=float, default=1.0, help="Exponent on 1/var (w = var^-gamma).")
    parser.add_argument("--uw_qclip", type=float, default=100.0, help="Quantile clip for weights (0-100, 100=off).")
    parser.add_argument("--uw_wmin", type=float, default=0.0, help="Minimum weight after normalization.")
    parser.add_argument("--uw_wmax", type=float, default=0.0, help="Maximum weight (<=0 disables).")
    parser.add_argument("--log_level", default="INFO", help="Logging level.")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(args.log_level)
    set_seeds(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = load_config(args.config)

    if args.device:
        cfg.device = args.device
    if args.gpu_id is not None:
        cfg.gpu_id = args.gpu_id

    # Setup device
    if cfg.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{cfg.gpu_id}")
    elif cfg.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon GPU (MPS)")
    elif cfg.device in ["cuda", "mps"]:
        device = torch.device("cpu")
        logger.warning("%s requested but not available. Falling back to CPU.", cfg.device.upper())
    else:
        device = torch.device("cpu")

    logger.info("Using device: %s", device)

    # Datasets and loaders
    logger.info("Building datasets and dataloaders...")
    _, train_loader, _, _ = CreateDataset(cfg, logger)
    seq_loader = build_sequential_loader(train_loader)

    # Load models
    models = load_models(args.checkpoints, cfg.feature_dim, cfg.dropout_rate, device, logger)

    # Compute uncertainty weights
    logger.info("Computing epistemic uncertainty across %d models...", len(models))
    unc_helper = EpistemicUncertaintyVAD(models=models, device=device, logger=logger)
    weights_by_video, var_stats, weight_stats = unc_helper.compute_weights(
        loader=seq_loader,
        max_batches=args.max_batches,
        unbiased_var=bool(args.uw_var_unbiased),
        eps=1e-8,
        shrink_alpha=float(args.uw_shrink_alpha),
        gamma=float(args.uw_gamma),
        qclip=float(args.uw_qclip),
        wmin=float(args.uw_wmin),
        wmax=float(args.uw_wmax),
    )

    # Compute Fisher per checkpoint
    logger.info("Computing UW-Fisher for each checkpoint...")
    for idx, (ckpt_path, model) in enumerate(zip(args.checkpoints, models)):
        fisher_calc = FisherVAD(model, device, logger)
        fishers = fisher_calc.compute_fisher_for_model(
            seq_loader,
            max_batches=args.max_batches,
            sample_weights=weights_by_video,
        )
        fishers = [torch.clamp(f, min=float(args.fisher_floor)) for f in fishers]
        fishers_cpu = [f.detach().cpu() for f in fishers]

        ckpt_path = Path(ckpt_path)
        folder_tag = ckpt_path.parent.name or "ckpt"
        out_name = f"fisher_uw_{folder_tag}_{ckpt_path.stem}.pt"
        out_path = Path(args.output_dir) / out_name

        payload = {
            "fisher_list": fishers_cpu,
            "metadata": {
                "type": "uncertainty_weighted",
                "checkpoint": str(ckpt_path),
                "max_batches": int(args.max_batches),
                "fisher_floor": float(args.fisher_floor),
                "uw_settings": {
                    "uw_var_unbiased": bool(args.uw_var_unbiased),
                    "uw_shrink_alpha": float(args.uw_shrink_alpha),
                    "uw_gamma": float(args.uw_gamma),
                    "uw_qclip": float(args.uw_qclip),
                    "uw_wmin": float(args.uw_wmin),
                    "uw_wmax": float(args.uw_wmax),
                },
                "var_stats": var_stats,
                "weight_stats": weight_stats,
                "num_models": len(models),
            },
        }
        torch.save(payload, out_path)
        logger.info("Saved UW-Fisher %d/%d to %s", idx + 1, len(models), out_path)

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()
