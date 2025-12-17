import logging
from typing import Dict, List, Optional, Sequence, Tuple

import torch


class EpistemicUncertaintyVAD:
    """Compute epistemic uncertainty and stabilized weights for VAD models."""

    def __init__(self, models: Sequence[torch.nn.Module], device: torch.device, logger: Optional[logging.Logger] = None):
        self.models = list(models)
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

    @torch.no_grad()
    def compute_model_outputs(self, loader, max_batches: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Run models over the loader and collect per-video outputs.

        Returns:
            Dict mapping video_name -> Tensor[num_models, T]
        """
        outputs_by_video: Dict[str, torch.Tensor] = {}

        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            features = batch[0].type(torch.float).to(self.device)
            video_names = batch[-1]

            for model_idx, model in enumerate(self.models):
                model.eval()
                preds = model(features).detach().cpu()  # [bs, T]

                for b, vname in enumerate(video_names):
                    vname = str(vname)
                    pred_vec = preds[b]
                    if vname not in outputs_by_video:
                        outputs_by_video[vname] = torch.zeros(len(self.models), pred_vec.shape[0], dtype=torch.float32)
                    outputs_by_video[vname][model_idx] = pred_vec

            if batch_idx % 10 == 0:
                self.logger.info("Collected outputs for batch %d", batch_idx + 1)

        if not outputs_by_video:
            raise RuntimeError("No outputs collected; check the dataloader and max_batches setting.")

        return outputs_by_video

    def compute_variances(self, outputs_by_video: Dict[str, torch.Tensor], unbiased: bool = False) -> Dict[str, torch.Tensor]:
        """Compute epistemic variance across models for each video."""
        var_by_video: Dict[str, torch.Tensor] = {}
        for vname, tensor in outputs_by_video.items():
            # tensor: [num_models, T]
            var_by_video[vname] = tensor.var(dim=0, unbiased=unbiased)
        return var_by_video

    def _stabilize_weights(self,
                           var_by_video: Dict[str, torch.Tensor],
                           eps: float,
                           shrink_alpha: float,
                           gamma: float,
                           qclip: float,
                           wmin: float,
                           wmax: Optional[float]) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Convert per-video variance into normalized, clipped weights."""
        weights_by_video: Dict[str, torch.Tensor] = {}

        all_vars = torch.cat([v.flatten() for v in var_by_video.values()])

        for vname, var in var_by_video.items():
            var_shrunk = (1.0 - shrink_alpha) * var + shrink_alpha * var.mean()
            weights = torch.pow(var_shrunk + eps, -gamma)

            if qclip < 100.0:
                upper = torch.quantile(weights, qclip / 100.0)
                weights = torch.clamp(weights, max=upper)

            weights = weights / weights.mean().clamp(min=eps)
            if wmin > 0.0:
                weights = torch.clamp(weights, min=wmin)
            if wmax is not None and wmax > 0.0:
                weights = torch.clamp(weights, max=wmax)

            weights_by_video[vname] = weights

        all_weights = torch.cat([w.flatten() for w in weights_by_video.values()])

        def _safe_corr(x: torch.Tensor, y: torch.Tensor) -> float:
            if x.numel() < 2 or y.numel() < 2:
                return float("nan")
            x_centered = x - x.mean()
            y_centered = y - y.mean()
            denom = x_centered.norm() * y_centered.norm()
            if denom.item() == 0.0:
                return 0.0
            return float(torch.dot(x_centered, y_centered) / denom)

        weight_stats = {
            "mean": float(all_weights.mean().item()),
            "std": float(all_weights.std(unbiased=False).item()),
            "min": float(all_weights.min().item()),
            "max": float(all_weights.max().item()),
            "q10": float(torch.quantile(all_weights, 0.10).item()),
            "q50": float(torch.quantile(all_weights, 0.50).item()),
            "q90": float(torch.quantile(all_weights, 0.90).item()),
            "corr": _safe_corr(all_weights, all_vars),
        }
        return weights_by_video, weight_stats

    def compute_weights(self,
                        loader,
                        max_batches: Optional[int],
                        unbiased_var: bool,
                        eps: float,
                        shrink_alpha: float,
                        gamma: float,
                        qclip: float,
                        wmin: float,
                        wmax: Optional[float]) -> Tuple[Dict[str, torch.Tensor], Dict[str, float], Dict[str, float]]:
        """Full pipeline: forward models -> variance -> stabilized weights."""
        outputs = self.compute_model_outputs(loader, max_batches=max_batches)
        var_by_video = self.compute_variances(outputs, unbiased=unbiased_var)

        all_var = torch.cat([v.flatten() for v in var_by_video.values()])
        var_stats = {
            "mean": float(all_var.mean().item()),
            "std": float(all_var.std(unbiased=False).item()),
            "min": float(all_var.min().item()),
            "max": float(all_var.max().item()),
            "q10": float(torch.quantile(all_var, 0.10).item()),
            "q50": float(torch.quantile(all_var, 0.50).item()),
            "q90": float(torch.quantile(all_var, 0.90).item()),
        }

        weights_by_video, weight_stats = self._stabilize_weights(
            var_by_video=var_by_video,
            eps=eps,
            shrink_alpha=shrink_alpha,
            gamma=gamma,
            qclip=qclip,
            wmin=wmin,
            wmax=wmax if wmax and wmax > 0.0 else None,
        )

        self.logger.info(
            "Uncertainty weights ready | mean=%.4f std=%.4f min=%.4f max=%.4f corr=%.4f",
            weight_stats["mean"],
            weight_stats["std"],
            weight_stats["min"],
            weight_stats["max"],
            weight_stats["corr"],
        )

        return weights_by_video, var_stats, weight_stats
