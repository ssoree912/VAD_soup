#!/usr/bin/env python3
"""
Fisher Soup helper that takes model directories (containing best_auc checkpoints)
and automatically computes Fisher information before performing Fisher-weighted soup.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import torch
import yaml

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset_loader import CreateDataset
from fisher_soup_vad import FisherSoupVAD, load_models_and_fishers
from fisher_vad import FisherVAD
from model import AD_Model
from utils import set_seeds


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("folder_fisher_soup")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def load_config(config_path: str) -> argparse.Namespace:
    with open(config_path, "r") as handle:
        cfg = yaml.load(handle, Loader=yaml.FullLoader)
    return argparse.Namespace(**cfg)


def resolve_device(requested: Optional[str], gpu_id: int, logger: logging.Logger) -> torch.device:
    if requested:
        req = requested.lower()
        if req.startswith("cuda"):
            if torch.cuda.is_available():
                return torch.device(requested if ":" in requested else f"cuda:{gpu_id}")
            logger.warning("CUDA requested but not available; falling back to CPU.")
            return torch.device("cpu")
        if req == "mps":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            logger.warning("MPS requested but not available; falling back to CPU.")
            return torch.device("cpu")
        if req == "cpu":
            return torch.device("cpu")
        logger.warning("Unknown device %s; falling back to CPU.", requested)
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def generate_coefficients(strategy: str,
                          n_models: int,
                          n_combinations: int,
                          random_seed: int,
                          fisher_soup: FisherSoupVAD) -> List[Sequence[float]]:
    if strategy == "uniform":
        return [[1.0 / n_models] * n_models]
    if strategy == "grid":
        if n_models == 2:
            return fisher_soup.create_pairwise_grid_coeffs(n_combinations)
        return fisher_soup.create_random_coeffs(n_models, n_combinations, random_seed)
    if strategy == "random":
        return fisher_soup.create_random_coeffs(n_models, n_combinations, random_seed)
    raise ValueError(f"Unknown strategy: {strategy}")


def compute_fisher_for_checkpoint(checkpoint_path: Path,
                                  cfg: argparse.Namespace,
                                  fisher_output: Path,
                                  device: torch.device,
                                  logger: logging.Logger) -> Path:
    if fisher_output.exists():
        logger.info("Using cached Fisher info at %s", fisher_output)
        return fisher_output

    logger.info("Computing Fisher information for %s", checkpoint_path)
    model = AD_Model(cfg.feature_dim, 512, cfg.dropout_rate)
    state_dict = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    fisher_vad = FisherVAD(model=model, device=device, logger=logger)

    # Prepare training loader for Fisher computation
    _, train_loader, _, _ = CreateDataset(cfg, logger)

    fisher_info = fisher_vad.compute_fisher_for_model(train_loader)
    fisher_tensors = [tensor.detach().cpu() for tensor in fisher_info]

    fisher_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fisher_tensors, fisher_output)
    logger.info("Saved Fisher info to %s", fisher_output)
    return fisher_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Fisher info and perform Fisher soup for checkpoints stored in folders."
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        required=True,
        help="One or more folders containing best_auc checkpoints.",
    )
    parser.add_argument("--config", required=True, help="Configuration YAML file.")
    parser.add_argument("--output", required=True, help="Output path for merged model.")
    parser.add_argument("--device", default=None, help="Device override (cpu, cuda, cuda:0, mps).")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU index when using CUDA.")
    parser.add_argument("--fisher_dir", default=None,
                        help="Directory to cache Fisher information (defaults to output directory).")
    parser.add_argument("--strategy", choices=["grid", "random", "uniform"], default="grid")
    parser.add_argument("--n_combinations", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--fisher_floor", type=float, default=1e-6)
    parser.add_argument("--no_favor_target", action="store_true")
    parser.add_argument("--no_normalize_fishers", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger()

    cfg = load_config(args.config)
    set_seeds(args.random_seed)

    device = resolve_device(args.device or getattr(cfg, "device", None), args.gpu_id, logger)
    logger.info("Using device: %s", device)

    # Override cfg device for downstream loaders
    cfg.device = device.type
    cfg.gpu_id = args.gpu_id

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fisher_cache_dir = Path(args.fisher_dir) if args.fisher_dir else output_path.parent / "fisher_cache"
    fisher_cache_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_paths: List[Path] = []
    fisher_paths: List[Path] = []

    for folder in args.folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        ckpt_path = folder_path / "best_auc.pkl"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint best_auc.pkl not found in {folder}")

        checkpoint_paths.append(ckpt_path)
        fisher_path = fisher_cache_dir / f"fisher_{folder_path.name}.pt"
        fisher_paths.append(
            compute_fisher_for_checkpoint(ckpt_path, cfg, fisher_path, device, logger)
        )

    # Load models and fishers (with mask support)
    models, fishers, masks = load_models_and_fishers(
        checkpoint_paths=[str(p) for p in checkpoint_paths],
        fisher_paths=[str(p) for p in fisher_paths],
        feature_dim=cfg.feature_dim,
        dropout_rate=cfg.dropout_rate,
        device=device,
        logger=logger
    )

    fisher_soup = FisherSoupVAD(device, logger)
    combined_mask = fisher_soup.combine_masks(masks)
    for model, mask in zip(models, masks):
        fisher_soup.apply_mask_to_model(model, mask)

    coefficients_set = generate_coefficients(
        strategy=args.strategy,
        n_models=len(models),
        n_combinations=args.n_combinations,
        random_seed=args.random_seed,
        fisher_soup=fisher_soup
    )
    logger.info("Generated %d coefficient combinations using %s strategy",
                len(coefficients_set), args.strategy)

    test_loader = None
    if args.evaluate:
        logger.info("Preparing dataset for evaluation...")
        test_loader, _, _, _ = CreateDataset(cfg, logger)

    if test_loader is not None:
        results = fisher_soup.search_merging_coefficients(
            models=models,
            coefficients_set=coefficients_set,
            test_loader=test_loader,
            feature_dim=cfg.feature_dim,
            dropout_rate=cfg.dropout_rate,
            segment_len=cfg.segment_len,
            fishers=fishers,
            fisher_floor=args.fisher_floor,
            favor_target_model=not args.no_favor_target,
            normalize_fishers=not args.no_normalize_fishers,
            combined_mask=combined_mask,
            print_results=True
        )
        best_result = max(results, key=lambda x: x.score["roc_auc"])
        best_coefficients = best_result.coefficients
        logger.info("Using best coefficients: %s", best_coefficients)
    else:
        best_coefficients = coefficients_set[0]
        best_result = None
        logger.info("Using coefficients without evaluation: %s", best_coefficients)

    merged_models = fisher_soup.generate_merged_for_coeffs_set(
        models=models,
        coefficients_set=[best_coefficients],
        feature_dim=cfg.feature_dim,
        dropout_rate=cfg.dropout_rate,
        fishers=fishers,
        fisher_floor=args.fisher_floor,
        favor_target_model=not args.no_favor_target,
        normalize_fishers=not args.no_normalize_fishers,
        combined_mask=combined_mask
    )
    _, final_model = next(merged_models)
    fisher_soup.apply_mask_to_model(final_model, combined_mask)

    torch.save(final_model.state_dict(), str(output_path))
    logger.info("Saved merged checkpoint to %s", output_path)

    if combined_mask is not None:
        mask_path = output_path.with_suffix(output_path.suffix + ".mask")
        mask_cpu = {key: tensor.to("cpu") for key, tensor in combined_mask.items()}
        torch.save(mask_cpu, mask_path)
        logger.info("Saved combined mask to %s", mask_path)

    coefficients_serialized = [float(c) for c in best_coefficients]

    metadata = {
        "folders": [str(p) for p in args.folders],
        "checkpoints": [str(p) for p in checkpoint_paths],
        "fisher_paths": [str(p) for p in fisher_paths],
        "coefficients": coefficients_serialized,
        "strategy": args.strategy,
        "n_combinations": args.n_combinations,
        "fisher_floor": args.fisher_floor,
        "favor_target_model": not args.no_favor_target,
        "normalize_fishers": not args.no_normalize_fishers,
        "random_seed": args.random_seed,
        "device": str(device),
        "mask_applied": combined_mask is not None,
    }
    if combined_mask is not None:
        total_mask_elems = sum(tensor.numel() for tensor in combined_mask.values())
        nonzero_mask = sum(int(tensor.sum().item()) for tensor in combined_mask.values())
        metadata["mask_stats"] = {
            "retained_weights": nonzero_mask,
            "total_weights": total_mask_elems,
            "density": float(nonzero_mask / total_mask_elems) if total_mask_elems else 0.0,
        }
    if best_result is not None:
        metadata["evaluation_score"] = {
            key: float(value) for key, value in best_result.score.items()
        }

    metadata_path = output_path.with_suffix(output_path.suffix + "_metadata.yaml")
    with open(metadata_path, "w") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)
    logger.info("Saved metadata to %s", metadata_path)

    if args.evaluate and test_loader is not None:
        final_score = fisher_soup.evaluate_model(final_model, test_loader, cfg.segment_len)
        logger.info("Final merged model performance:")
        logger.info("  PR AUC: %.4f", final_score["pr_auc"])
        logger.info("  ROC AUC: %.4f", final_score["roc_auc"])

    logger.info("Folder Fisher Soup completed successfully!")


if __name__ == "__main__":
    main()
