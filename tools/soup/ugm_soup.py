import argparse
import copy
import itertools
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Sequence

import torch
import yaml

# Ensure project root on sys.path (tools/soup -> tools -> project root)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(ROOT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset_loader import CreateDataset
from model import AD_Model
from utils import calc_metrics, set_seeds


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("ugm_soup")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def _load_fisher_raw(path: str):
    return torch.load(path, map_location="cpu")


def _extract_fisher_list(fisher_obj) -> Optional[List[torch.Tensor]]:
    """Accept list, dict with 'fisher_list', or dict of param_i keys."""
    if isinstance(fisher_obj, list):
        return fisher_obj
    if isinstance(fisher_obj, dict):
        if "fisher_list" in fisher_obj and isinstance(fisher_obj["fisher_list"], list):
            return fisher_obj["fisher_list"]

        def _key_sort_key(key: str):
            suffix = key.split("_")[-1]
            return (0, int(suffix)) if suffix.isdigit() else (1, key)

        return [fisher_obj[k] for k in sorted(fisher_obj.keys(), key=_key_sort_key)]
    return None


def _normalize_fisher_to_state_dict(fisher_obj,
                                    model_state: Dict[str, torch.Tensor],
                                    eps: float) -> Dict[str, torch.Tensor]:
    """
    Convert various Fisher formats to a state_dict-shaped dict.

    Accepts:
    - dict matching model_state keys
    - list of fishers (ordered as mergeable params with dim>1)
    - dict with key 'fisher_list' -> list
    - dict with param_0/param_1... keys -> list order
    """
    if isinstance(fisher_obj, dict) and set(fisher_obj.keys()) == set(model_state.keys()):
        return fisher_obj

    fisher_list = _extract_fisher_list(fisher_obj)
    if fisher_list is None:
        raise ValueError("Unsupported Fisher format; expected dict keyed like state_dict or list/fisher_list.")

    out: Dict[str, torch.Tensor] = {}
    list_idx = 0
    mergeable_needed = sum(
        1 for t in model_state.values()
        if torch.is_tensor(t) and t.is_floating_point() and t.dim() > 1
    )
    if len(fisher_list) < mergeable_needed:
        raise ValueError(
            f"Fisher list length {len(fisher_list)} is smaller than expected mergeable params {mergeable_needed}."
        )

    for key, tensor in model_state.items():
        if torch.is_tensor(tensor) and tensor.is_floating_point() and tensor.dim() > 1:
            fisher_tensor = fisher_list[list_idx]
            list_idx += 1
            out[key] = fisher_tensor
        elif torch.is_tensor(tensor):
            out[key] = torch.full_like(tensor, eps)
        else:
            out[key] = tensor

    return out


def _load_config(config_path: str) -> argparse.Namespace:
    with open(config_path, "r") as handle:
        cfg = yaml.load(handle, Loader=yaml.FullLoader)
    return argparse.Namespace(**cfg)


def _build_model(args: argparse.Namespace, state_dict: Dict[str, torch.Tensor], device: torch.device) -> AD_Model:
    model = AD_Model(args.feature_dim, 512, args.dropout_rate)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def _evaluate_model(model: AD_Model, args: argparse.Namespace, device: torch.device, logger: logging.Logger):
    logger.info("Preparing datasets for evaluation...")
    test_loader, _, _, _ = CreateDataset(args, logger)

    total_scores = []
    total_labels = []

    model.eval()
    with torch.no_grad():
        for features, label_frames, _ in test_loader:
            features = features.type(torch.float).to(device)
            label_frames = label_frames.type(torch.float)
            outputs = model(features)
            scores = outputs.squeeze().cpu().numpy()

            for score, label in zip(scores, label_frames[0]):
                total_scores.extend([score] * args.segment_len)
                total_labels.extend(label.detach().cpu().numpy().astype(int).tolist())

    prauc_frames, rocauc_frames = calc_metrics(total_scores, total_labels)
    logger.info("UGM evaluation — PR AUC: %.2f%%, ROC AUC: %.2f%%", prauc_frames, rocauc_frames)
    return {
        "pr_auc": float(prauc_frames),
        "roc_auc": float(rocauc_frames),
    }


def _validate_keys(models: List[Dict[str, torch.Tensor]],
                   fishers: List[Dict[str, torch.Tensor]]) -> None:
    model_keys = set(models[0].keys())
    fisher_keys = set(fishers[0].keys())

    for idx, state in enumerate(models[1:], start=1):
        if set(state.keys()) != model_keys:
            raise ValueError(f"Model state_dict keys differ at index {idx}.")

    for idx, fisher in enumerate(fishers[1:], start=1):
        if set(fisher.keys()) != fisher_keys:
            raise ValueError(f"Fisher keys differ at index {idx}.")

    if model_keys != fisher_keys:
        raise ValueError("Model keys and Fisher keys must match one-to-one for UGB merging.")


def ugm_merge_state_dicts(models: List[Dict[str, torch.Tensor]],
                          fishers: List[Dict[str, torch.Tensor]],
                          alphas: torch.Tensor,
                          ref_idx: int = 0,
                          eps: float = 1e-8,
                          use_ref_fisher: bool = True) -> Dict[str, torch.Tensor]:
    """
    Merge multiple model state_dicts using uncertainty-based gradient matching (diagonal form).
    """
    if len(models) != len(fishers):
        raise ValueError("Number of models and fishers must match.")
    if not models:
        raise ValueError("At least one model is required for merging.")
    if ref_idx < 0 or ref_idx >= len(models):
        raise IndexError(f"ref_idx {ref_idx} is out of range for {len(models)} models.")

    _validate_keys(models, fishers)

    K = len(models)
    alphas = torch.as_tensor(alphas, dtype=torch.float32)
    if alphas.numel() != K:
        raise ValueError(f"alphas must have length {K}, got {alphas.numel()}.")

    merged = copy.deepcopy(models[ref_idx])
    ref_sd = models[ref_idx]

    if use_ref_fisher:
        H0_sd = fishers[ref_idx]
    else:
        H0_sd = {k: torch.full_like(v, eps) for k, v in ref_sd.items()}

    barH: Dict[str, torch.Tensor] = {}
    for key in ref_sd.keys():
        ref_tensor = ref_sd[key]
        if not torch.is_tensor(ref_tensor) or not ref_tensor.is_floating_point():
            continue

        fish = torch.stack([f[key] for f in fishers], dim=0)
        a = alphas.view(K, *([1] * (fish.dim() - 1)))
        bar = H0_sd[key].clone()
        bar = bar + torch.sum(a * fish, dim=0)
        barH[key] = bar.clamp_min(eps)

    for key in merged.keys():
        ref_tensor = ref_sd[key]
        if not torch.is_tensor(ref_tensor) or not ref_tensor.is_floating_point():
            merged[key] = ref_tensor
            continue

        fish = torch.stack([f[key] for f in fishers], dim=0)
        params = torch.stack([m[key] for m in models], dim=0)

        a = alphas.view(K, *([1] * (params.dim() - 1)))

        H0 = H0_sd[key]
        H0_plus_Ht = H0.unsqueeze(0) + fish

        denom = barH[key]
        W = H0_plus_Ht / denom.unsqueeze(0)
        weights = a * W

        inc = params - ref_tensor.unsqueeze(0)
        delta = torch.sum(weights * inc, dim=0)
        merged[key] = ref_tensor + delta

    return merged


def merge_from_paths(checkpoints: Sequence[str],
                     fisher_paths: Sequence[str],
                     alphas: Sequence[float],
                     output_path: str,
                     ref_idx: int = 0,
                     eps: float = 1e-8,
                     use_ref_fisher: bool = True,
                     logger: Optional[logging.Logger] = None) -> Dict[str, torch.Tensor]:
    logger = logger or _setup_logger()

    if len(checkpoints) != len(fisher_paths):
        raise ValueError("Checkpoint and Fisher path counts must match.")

    logger.info("Loading %d checkpoints and Fisher files...", len(checkpoints))
    models = [_load_state_dict(p) for p in checkpoints]
    raw_fishers = [_load_fisher_raw(p) for p in fisher_paths]
    fishers = [
        _normalize_fisher_to_state_dict(raw, model_state=sd, eps=eps)
        for raw, sd in zip(raw_fishers, models)
    ]

    merged = ugm_merge_state_dicts(
        models=models,
        fishers=fishers,
        alphas=torch.tensor(alphas, dtype=torch.float32),
        ref_idx=ref_idx,
        eps=eps,
        use_ref_fisher=use_ref_fisher,
    )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(merged, output_path)
    logger.info("Saved UGB-merged checkpoint to %s", output_path)
    return merged


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Uncertainty-based gradient matching soup for state_dict checkpoints."
    )
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Checkpoint paths to merge.")
    parser.add_argument("--fishers", nargs="+", required=True,
                        help="Diagonal Fisher/Hessian paths aligned with checkpoints.")
    parser.add_argument("--alphas", nargs="+", type=float, required=True,
                        help="Scalar coefficients (one per checkpoint).")
    parser.add_argument("--output", required=True,
                        help="Path to save the merged checkpoint.")
    parser.add_argument("--ref_idx", type=int, default=0,
                        help="Reference checkpoint index.")
    parser.add_argument("--eps", type=float, default=1e-8,
                        help="Numerical stability constant.")
    parser.add_argument("--no_ref_fisher", action="store_true",
                        help="Use constant prior instead of reference Fisher for H0.")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation on merged checkpoint.")
    parser.add_argument("--config", help="YAML config path required if --evaluate is set.")
    parser.add_argument("--device", default=None, help="Override device for evaluation (cpu, cuda, or mps).")
    parser.add_argument("--gpu_id", type=int, default=None, help="GPU index to use when device is cuda.")
    parser.add_argument("--metrics_json", default=None,
                        help="Path to save evaluation metrics JSON (defaults to <output>_metrics.json).")
    parser.add_argument("--grid_search", action="store_true",
                        help="Run grid search over alpha combinations instead of a single fixed alpha vector.")
    parser.add_argument("--grid_values", nargs="+", type=float, default=None,
                        help="Base values for each alpha when running grid search (e.g., 0.0 0.5 1.0). If not set, a default [0.0, 0.5, 1.0] will be used.")
    parser.add_argument("--grid_normalize", action="store_true",
                        help="If set, normalize each alpha combination so that the sum over models is 1. Combinations with all zeros are skipped.")
    parser.add_argument("--select_by", choices=["roc_auc", "pr_auc"], default="roc_auc",
                        help="Metric to use when selecting the best combination during grid search.")
    return parser.parse_args()


def main():
    args = _parse_args()
    logger = _setup_logger()

    use_ref_fisher = not args.no_ref_fisher

    # Single-merge mode
    if not args.grid_search:
        merged_state = merge_from_paths(
            checkpoints=args.checkpoints,
            fisher_paths=args.fishers,
            alphas=args.alphas,
            output_path=args.output,
            ref_idx=args.ref_idx,
            eps=args.eps,
            use_ref_fisher=use_ref_fisher,
            logger=logger,
        )

        if not args.evaluate:
            return

        if not args.config:
            raise ValueError("--config is required when --evaluate is set.")

        cfg = _load_config(args.config)
        set_seeds(getattr(cfg, "seed", 42))

        eval_device_arg = args.device or getattr(cfg, "device", None)
        eval_gpu_id = args.gpu_id if args.gpu_id is not None else getattr(cfg, "gpu_id", 0)

        if eval_device_arg == "cuda" and torch.cuda.is_available():
            device = torch.device(f"cuda:{eval_gpu_id}")
        elif eval_device_arg == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple Silicon GPU (MPS)")
        elif eval_device_arg in ["cuda", "mps"]:
            device = torch.device("cpu")
            logger.warning("%s requested but not available. Falling back to CPU.", eval_device_arg.upper())
        else:
            device = torch.device("cpu")

        model = _build_model(cfg, merged_state, device)
        metrics = _evaluate_model(model, cfg, device, logger)

        metrics_path = args.metrics_json or f"{os.path.splitext(args.output)[0]}_metrics.json"
        with open(metrics_path, "w") as handle:
            json.dump({"soup_path": args.output, **metrics}, handle, indent=2)
        logger.info("Saved evaluation metrics to %s", metrics_path)
        return

    # Grid-search mode
    if not args.config:
        raise ValueError("--config is required when --grid_search is set.")

    cfg = _load_config(args.config)
    set_seeds(getattr(cfg, "seed", 42))

    eval_device_arg = args.device or getattr(cfg, "device", None)
    eval_gpu_id = args.gpu_id if args.gpu_id is not None else getattr(cfg, "gpu_id", 0)

    if eval_device_arg == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{eval_gpu_id}")
    elif eval_device_arg == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon GPU (MPS)")
    elif eval_device_arg in ["cuda", "mps"]:
        device = torch.device("cpu")
        logger.warning("%s requested but not available. Falling back to CPU.", eval_device_arg.upper())
    else:
        device = torch.device("cpu")

    K = len(args.checkpoints)
    grid_values = args.grid_values or [0.0, 0.5, 1.0]
    logger.info("Running grid search over alphas with base values: %s", grid_values)

    best_metrics: Optional[Dict[str, float]] = None
    best_alphas: Optional[List[float]] = None
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for combo in itertools.product(grid_values, repeat=K):
        combo = list(combo)
        if args.grid_normalize:
            s = sum(combo)
            if s == 0:
                continue
            alphas = [c / s for c in combo]
        else:
            alphas = combo

        logger.info("Trying alphas: %s", alphas)

        merged_state = merge_from_paths(
            checkpoints=args.checkpoints,
            fisher_paths=args.fishers,
            alphas=alphas,
            output_path=args.output,
            ref_idx=args.ref_idx,
            eps=args.eps,
            use_ref_fisher=use_ref_fisher,
            logger=logger,
        )

        model = _build_model(cfg, merged_state, device)
        metrics = _evaluate_model(model, cfg, device, logger)

        metric_value = metrics.get(args.select_by)
        if metric_value is None:
            raise ValueError(f"Metric {args.select_by} not found in evaluation metrics.")

        if best_metrics is None or metric_value > best_metrics.get(args.select_by, float("-inf")):
            best_metrics = metrics
            best_alphas = alphas
            best_state = merged_state
            torch.save(best_state, args.output)
            logger.info("New best %s = %.4f with alphas %s", args.select_by, metric_value, alphas)

    if best_metrics is None or best_alphas is None:
        logger.warning("No valid alpha combination found during grid search.")
        return

    metrics_path = args.metrics_json or f"{os.path.splitext(args.output)[0]}_metrics.json"
    with open(metrics_path, "w") as handle:
        json.dump({"soup_path": args.output, "alphas": best_alphas, **best_metrics}, handle, indent=2)
    logger.info("Grid search completed. Best alphas: %s", best_alphas)
    logger.info("Saved best merged checkpoint to %s and metrics to %s", args.output, metrics_path)


if __name__ == "__main__":
    main()
