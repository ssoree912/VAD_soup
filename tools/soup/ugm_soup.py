import argparse
import copy
import logging
import os
from typing import Dict, List, Optional, Sequence

import torch


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("ugb_soup")
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


def _load_fisher_dict(path: str) -> Dict[str, torch.Tensor]:
    fisher = torch.load(path, map_location="cpu")
    if not isinstance(fisher, dict):
        raise ValueError(f"Expected Fisher data at {path} to be a dict of tensors.")
    return fisher


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

    Args:
        models: List of model state_dicts. All dicts must share identical keys.
        fishers: List of state_dict-shaped Fisher/Hessian diagonal approximations (same keys/shapes as models).
        alphas: Tensor of shape [K] containing scalar coefficients for each model.
        ref_idx: Index of the reference model (theta_ref).
        eps: Small constant for numerical stability.
        use_ref_fisher: If True, use the reference model Fisher as H0, otherwise use a constant eps prior.

    Returns:
        Merged state_dict following Eq.(12) diagonal approximation.
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
            # Non-floating buffers (e.g., counters) stay from the reference model.
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
                     logger: Optional[logging.Logger] = None) -> str:
    logger = logger or _setup_logger()

    if len(checkpoints) != len(fisher_paths):
        raise ValueError("Checkpoint and Fisher path counts must match.")

    logger.info("Loading %d checkpoints and Fisher files...", len(checkpoints))
    models = [_load_state_dict(p) for p in checkpoints]
    fishers = [_load_fisher_dict(p) for p in fisher_paths]

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
    return output_path


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
    return parser.parse_args()


def main():
    args = _parse_args()
    logger = _setup_logger()

    use_ref_fisher = not args.no_ref_fisher
    merge_from_paths(
        checkpoints=args.checkpoints,
        fisher_paths=args.fishers,
        alphas=args.alphas,
        output_path=args.output,
        ref_idx=args.ref_idx,
        eps=args.eps,
        use_ref_fisher=use_ref_fisher,
        logger=logger,
    )


if __name__ == "__main__":
    main()
