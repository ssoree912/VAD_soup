#!/usr/bin/env python3
"""
Utility script for visualising pruning behaviour, Fisher statistics, soup searches,
and performance trade-offs for LANP-UVAD experiments.

Each sub-command operates on already-produced artifacts (checkpoints, pruning masks,
Fisher tensors, metadata CSV/YAML files) and saves publication-ready plots.

Example invocations:
    python tools/visualize_vad.py pruning-mask --mask path/to/best_auc.pkl.mask --out-dir figs/
    python tools/visualize_vad.py weight-hist --checkpoint best_auc.pkl --mask best_auc.pkl.mask --random-ratio 0.05
    python tools/visualize_vad.py fisher-distribution --fisher fisher.pt --param-names param_names.txt
    python tools/visualize_vad.py weight-vs-fisher --checkpoint best_auc.pkl --fisher fisher.pt --feature-dim 2048
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_fig(fig: matplotlib.figure.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2) Pruning visualisations
# ---------------------------------------------------------------------------


def plot_pruning_mask_heatmap(mask_path: Path, out_dir: Path, top_k: Optional[int] = None) -> None:
    mask_obj = torch.load(str(mask_path), map_location="cpu")
    if isinstance(mask_obj, torch.Tensor):
        raise ValueError("Mask file appears to be a single tensor. Expected a dict of named tensors.")

    entries: List[Tuple[str, float, np.ndarray]] = []
    channel_stats: List[Tuple[str, np.ndarray]] = []

    for name, tensor in mask_obj.items():
        mask = tensor.detach().cpu().float().numpy()
        keep_ratio = mask.mean()
        sparsity = 1.0 - keep_ratio
        entries.append((name, sparsity, mask.shape))

        if mask.ndim >= 2:
            out_dim = mask.shape[0]
            reshaped = mask.reshape(out_dim, -1)
            channel_sparsity = 1.0 - reshaped.mean(axis=1)
            channel_stats.append((name, channel_sparsity))

    # Sort layers by sparsity descending
    entries.sort(key=lambda x: x[1], reverse=True)
    if top_k is not None:
        entries = entries[:top_k]

    fig, ax = plt.subplots(figsize=(max(6, len(entries) * 0.6), 4))
    layer_names = [e[0] for e in entries]
    sparsities = [e[1] * 100 for e in entries]
    ax.bar(layer_names, sparsities, color="#4c78a8")
    ax.set_ylabel("Sparsity (%)")
    ax.set_title("Layer-wise sparsity")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=45, ha="right")
    _save_fig(fig, out_dir / "pruning_layer_sparsity.png")

    if channel_stats:
        fig, ax = plt.subplots(figsize=(8, max(4, len(channel_stats))))
        data = []
        ytick = []
        for idx, (name, ch_sparsity) in enumerate(channel_stats):
            data.append(ch_sparsity[None, :])
            ytick.append(f"{name} ({len(ch_sparsity)} ch)")
        heatmap = np.vstack(
            [
                np.pad(arr, ((0, 0), (0, max(1, max(x.shape[1] for x in data) - arr.shape[1]))), mode="constant")
                for arr in data
            ]
        )
        im = ax.imshow(heatmap, aspect="auto", cmap="magma_r", vmin=0.0, vmax=1.0)
        ax.set_yticks(np.arange(len(ytick)))
        ax.set_yticklabels(ytick)
        ax.set_xlabel("Output channel index")
        ax.set_title("Channel-wise sparsity")
        fig.colorbar(im, ax=ax, label="Sparsity")
        _save_fig(fig, out_dir / "pruning_channel_heatmap.png")


def plot_weight_hist_with_threshold(
    checkpoint_path: Path,
    mask_path: Path,
    out_path: Path,
    bins: int = 80,
    random_ratio: Optional[float] = None,
) -> None:
    state = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    mask_obj = torch.load(str(mask_path), map_location="cpu")

    all_abs = []
    pruned_abs = []
    for name, tensor in state.items():
        if name not in mask_obj:
            continue
        weight = tensor.detach().cpu().float().numpy()
        mask = mask_obj[name].detach().cpu().float().numpy()
        all_abs.append(np.abs(weight).reshape(-1))
        pruned_abs.append(np.abs(weight[mask == 0.0]))

    if not all_abs or not pruned_abs:
        raise RuntimeError("No overlapping tensors between checkpoint and mask.")

    all_abs_flat = np.concatenate(all_abs)
    pruned_abs_flat = np.concatenate(pruned_abs)
    tau = pruned_abs_flat.max() if pruned_abs_flat.size > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(all_abs_flat, bins=bins, alpha=0.6, label="|w|", color="#4c78a8")
    if pruned_abs_flat.size > 0:
        ax.hist(pruned_abs_flat, bins=bins, alpha=0.4, label="|w| (pruned)", color="#f58518")
    if np.isfinite(tau):
        ax.axvline(tau, color="#54a24b", linestyle="--", linewidth=2, label=f"tau = {tau:.4e}")
    if random_ratio is not None:
        ax.text(
            0.97,
            0.95,
            f"Random ratio: {random_ratio:.2%}",
            transform=ax.transAxes,
            ha="right",
            va="top",
        )
    ax.set_xlabel("|w|")
    ax.set_ylabel("Frequency")
    ax.set_title("Weight magnitude distribution")
    ax.legend()
    _save_fig(fig, out_path)


def plot_unprune_timeline(total_epochs: int, unprune_epoch: Optional[int], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 1.8))
    ax.set_title("Pruning → dense training timeline")
    ax.set_xlim(0, total_epochs)
    ax.set_ylim(0, 1)
    ax.set_yticks([])

    if unprune_epoch is None or unprune_epoch >= total_epochs:
        ax.barh(0.5, total_epochs, height=0.6, color="#f58518")
        ax.text(total_epochs / 2, 0.5, "Pruned training (masks active)", ha="center", va="center", color="white")
    else:
        ax.barh(0.5, unprune_epoch, height=0.6, color="#f58518")
        ax.barh(
            0.5,
            total_epochs - unprune_epoch,
            height=0.6,
            left=unprune_epoch,
            color="#4c78a8",
        )
        ax.axvline(unprune_epoch, color="black", linestyle="--")
        ax.text(
            unprune_epoch / 2,
            0.5,
            f"Pruned (0-{unprune_epoch})",
            ha="center",
            va="center",
            color="white",
        )
        ax.text(
            unprune_epoch + (total_epochs - unprune_epoch) / 2,
            0.5,
            f"Dense ({unprune_epoch}-{total_epochs})",
            ha="center",
            va="center",
            color="white",
        )

    ax.set_xlabel("Epoch")
    _save_fig(fig, out_path)


# ---------------------------------------------------------------------------
# 3) Fisher & soup visualisations
# ---------------------------------------------------------------------------


def _load_fisher(path: Path) -> Dict[str, torch.Tensor]:
    fisher = torch.load(str(path), map_location="cpu")
    if isinstance(fisher, dict):
        return {str(k): v.detach().cpu() for k, v in fisher.items()}
    if isinstance(fisher, list):
        return {f"param_{idx}": tensor.detach().cpu() for idx, tensor in enumerate(fisher)}
    raise ValueError(f"Unsupported Fisher format at {path}")


def plot_fisher_distribution(
    fisher_path: Path,
    out_path: Path,
    param_names_path: Optional[Path] = None,
    log_scale: bool = True,
) -> None:
    fisher_dict = _load_fisher(fisher_path)
    if param_names_path is not None:
        names = [line.strip() for line in param_names_path.read_text().splitlines() if line.strip()]
        if len(names) == len(fisher_dict):
            fisher_dict = {names[idx]: tensor for idx, tensor in enumerate(fisher_dict.values())}

    labels = []
    values = []
    for name, tensor in fisher_dict.items():
        if tensor.numel() == 0:
            continue
        arr = tensor.detach().float().view(-1).numpy()
        labels.append(name)
        values.append(arr)

    fig, ax = plt.subplots(figsize=(max(6, len(values) * 0.6), 4))
    ax.boxplot(values, vert=True, showfliers=False)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Fisher diagonal value")
    ax.set_title("Layer-wise Fisher distribution")
    if log_scale:
        ax.set_yscale("log")
    _save_fig(fig, out_path)


def plot_weight_vs_fisher(
    checkpoint_path: Path,
    fisher_path: Path,
    out_path: Path,
    feature_dim: int,
    dropout_rate: float,
    sample: Optional[int] = 50000,
) -> None:
    from model import AD_Model  # local import to avoid circular deps

    model = AD_Model(feature_dim, 512, dropout_rate)
    state = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)

    fisher_dict = _load_fisher(fisher_path)
    params = [p for p in model.parameters() if p.requires_grad and p.dim() > 1]
    named_params = {name: p for name, p in model.named_parameters() if p.requires_grad and p.dim() > 1}
    fisher_items = list(fisher_dict.items())

    pairs: List[Tuple[torch.Tensor, torch.nn.Parameter]] = []

    if all(name in named_params for name in fisher_dict.keys()):
        for name, fisher_tensor in fisher_items:
            pairs.append((fisher_tensor, named_params[name]))
    elif len(params) == len(fisher_items):
        for idx in range(len(params)):
            pairs.append((fisher_items[idx][1], params[idx]))
    else:
        raise RuntimeError(
            "Unable to match Fisher entries to model parameters. Provide --param-names when exporting Fisher."
        )

    weight_values = []
    fisher_values = []

    for fisher_tensor, param in pairs:
        w = param.detach().cpu().view(-1).numpy()
        f = fisher_tensor.detach().cpu().view(-1).numpy()
        if w.size != f.size:
            min_len = min(w.size, f.size)
            w = w[:min_len]
            f = f[:min_len]
        weight_values.append(np.abs(w))
        fisher_values.append(f)

    abs_w = np.concatenate(weight_values)
    fisher_flat = np.concatenate(fisher_values)

    if sample is not None and sample < abs_w.size:
        indices = np.random.choice(abs_w.size, sample, replace=False)
        abs_w = abs_w[indices]
        fisher_flat = fisher_flat[indices]

    corr = np.corrcoef(abs_w, fisher_flat)[0, 1] if abs_w.size > 1 else float("nan")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(abs_w, fisher_flat, s=6, alpha=0.4, color="#4c78a8")
    ax.set_xlabel("|w|")
    ax.set_ylabel("Fisher value")
    ax.set_title(f"|w| vs Fisher (r = {corr:.3f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _save_fig(fig, out_path)


def _barycentric_to_cartesian(coeffs: Sequence[float]) -> Tuple[float, float]:
    a, b, c = coeffs
    x = 0.5 * (2 * b + c)
    y = (math.sqrt(3) / 2) * c
    return x, y


def plot_soup_coefficients(results_path: Path, metric: str, out_path: Path) -> None:
    data = yaml.safe_load(results_path.read_text())
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    if not isinstance(data, list):
        raise ValueError("Expected a list of coefficient results.")

    first = data[0]
    num_models = len(first["coefficients"])

    if num_models == 2:
        xs = []
        scores = []
        for entry in data:
            coeff = entry["coefficients"][0]
            xs.append(coeff)
            scores.append(entry["score"][metric])
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(xs, scores, marker="o", color="#4c78a8")
        ax.set_xlabel("Coefficient for model 0")
        ax.set_ylabel(metric)
        ax.set_title(f"Grid sweep ({metric})")
    elif num_models == 3:
        coords = []
        values = []
        for entry in data:
            xy = _barycentric_to_cartesian(entry["coefficients"])
            coords.append(xy)
            values.append(entry["score"][metric])
        x, y = np.array(coords).T
        fig, ax = plt.subplots(figsize=(5, 5))
        sc = ax.scatter(x, y, c=values, cmap="viridis", s=80)
        ax.set_title(f"Coefficient simplex ({metric})")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(sc, ax=ax, label=metric)
    else:
        coeff_arrays = np.array([entry["coefficients"] for entry in data])
        topk = min(10, len(data))
        indices = np.argsort([entry["score"][metric] for entry in data])[::-1][:topk]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.imshow(coeff_arrays[indices], aspect="auto", cmap="viridis")
        ax.set_xlabel("Model index")
        ax.set_ylabel("Top-k combinations")
        ax.set_title(f"Top-{topk} coefficient heatmap ({metric})")

    _save_fig(fig, out_path)


def plot_fisher_vs_uniform(
    metrics_path: Path,
    out_path: Path,
    strategy_col: str = "strategy",
    roc_col: str = "roc_auc",
    pr_col: str = "pr_auc",
) -> None:
    rows = []
    if metrics_path.suffix.lower() in {".yml", ".yaml"}:
        data = yaml.safe_load(metrics_path.read_text())
        if isinstance(data, dict) and "rows" in data:
            data = data["rows"]
        rows = data
    else:
        with open(metrics_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    fisher_scores = []
    uniform_scores = []
    for row in rows:
        strategy = str(row[strategy_col]).lower()
        target = fisher_scores if "fisher" in strategy else uniform_scores
        target.append((float(row[roc_col]), float(row[pr_col])))

    fig, ax = plt.subplots(figsize=(5, 4))
    if uniform_scores:
        u = np.array(uniform_scores)
        ax.scatter(u[:, 0], u[:, 1], color="#4c78a8", label="Uniform")
    if fisher_scores:
        f = np.array(fisher_scores)
        ax.scatter(f[:, 0], f[:, 1], color="#f58518", label="Fisher-weighted")
    ax.set_xlabel("ROC-AUC")
    ax.set_ylabel("PR-AUC")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Uniform vs Fisher soups")
    _save_fig(fig, out_path)


# ---------------------------------------------------------------------------
# 5) Performance / efficiency trade-offs
# ---------------------------------------------------------------------------


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def plot_sparsity_vs_auc(csv_path: Path, out_path: Path) -> None:
    rows = _load_csv(csv_path)
    magnitude = [float(r.get("mag_pct", r.get("magnitude_pct", r.get("mag", 0)))) for r in rows]
    random_pct = [float(r.get("rand_pct", r.get("random_pct", r.get("rand", 0)))) for r in rows]
    roc = [float(r["roc_auc"]) for r in rows]
    pr = [float(r["pr_auc"]) for r in rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(magnitude, roc, c=random_pct, cmap="viridis", label="ROC-AUC")
    ax.set_xlabel("Magnitude pruning (%)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Sparsity vs ROC-AUC (color = random%)")
    fig.colorbar(sc, ax=ax, label="Random pruning (%)")
    _save_fig(fig, out_path.with_name(out_path.stem + "_roc.png"))

    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(magnitude, pr, c=random_pct, cmap="viridis", label="PR-AUC")
    ax.set_xlabel("Magnitude pruning (%)")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Sparsity vs PR-AUC (color = random%)")
    fig.colorbar(sc, ax=ax, label="Random pruning (%)")
    _save_fig(fig, out_path.with_name(out_path.stem + "_pr.png"))


def plot_unprune_ratio_vs_auc(csv_path: Path, out_path: Path) -> None:
    rows = _load_csv(csv_path)
    ratios = [float(r["unprune_ratio"]) for r in rows]
    roc = [float(r["roc_auc"]) for r in rows]
    pr = [float(r["pr_auc"]) for r in rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ratios, roc, marker="o", color="#4c78a8", label="ROC-AUC")
    ax.plot(ratios, pr, marker="s", color="#f58518", label="PR-AUC")
    ax.set_xlabel("Unprune ratio")
    ax.set_ylabel("AUC")
    ax.set_title("Unprune ratio sensitivity")
    ax.legend()
    ax.grid(alpha=0.3)
    _save_fig(fig, out_path)


def plot_efficiency_tradeoff(csv_path: Path, out_path: Path) -> None:
    rows = _load_csv(csv_path)
    settings = [r["setting"] for r in rows]
    roc = [float(r["roc_auc"]) for r in rows]
    pr = [float(r["pr_auc"]) for r in rows]
    params = [float(r.get("params_m", 0)) for r in rows]
    fps = [float(r.get("fps", 0)) for r in rows]
    vram = [float(r.get("vram_mb", 0)) for r in rows]

    width = 0.2
    x = np.arange(len(settings))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width, params, width, label="Params (M)", color="#4c78a8")
    ax.bar(x, fps, width, label="FPS", color="#f58518")
    ax.bar(x + width, vram, width, label="VRAM (MB)", color="#54a24b")
    ax.set_xticks(x)
    ax.set_xticklabels(settings, rotation=45, ha="right")
    ax.set_title("Efficiency comparison")
    ax.legend()
    _save_fig(fig, out_path.with_name(out_path.stem + "_bars.png"))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(roc, fps, color="#4c78a8", label="ROC-AUC vs FPS")
    for idx, name in enumerate(settings):
        ax.annotate(name, (roc[idx], fps[idx]), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("ROC-AUC")
    ax.set_ylabel("FPS")
    ax.grid(alpha=0.3)
    ax.legend()
    _save_fig(fig, out_path.with_name(out_path.stem + "_roc_fps.png"))


def plot_calibration_curve(
    predictions_path: Path,
    labels_path: Path,
    out_path: Path,
    num_bins: int = 15,
) -> None:
    preds = np.load(predictions_path) if predictions_path.suffix == ".npy" else np.loadtxt(predictions_path, delimiter=",")
    labels = (
        np.load(labels_path) if labels_path.suffix == ".npy" else np.loadtxt(labels_path, delimiter=",")
    )
    preds = preds.reshape(-1)
    labels = labels.reshape(-1)
    bins = np.linspace(0, 1, num_bins + 1)
    bin_ids = np.digitize(preds, bins) - 1
    bin_acc = []
    bin_conf = []
    for idx in range(num_bins):
        mask = bin_ids == idx
        if not mask.any():
            continue
        bin_acc.append(labels[mask].mean())
        bin_conf.append(preds[mask].mean())

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Ideal")
    ax.plot(bin_conf, bin_acc, marker="o", color="#4c78a8", label="Model")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Calibration curve")
    ax.legend()
    ax.grid(alpha=0.3)
    _save_fig(fig, out_path)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualisation helpers for LANP-UVAD.")
    sub = parser.add_subparsers(dest="command", required=True)

    # Pruning mask
    p_mask = sub.add_parser("pruning-mask", help="Plot layer/channel sparsity from a pruning mask file.")
    p_mask.add_argument("--mask", type=Path, required=True)
    p_mask.add_argument("--out-dir", type=Path, required=True)
    p_mask.add_argument("--top-k", type=int, default=None, help="Optional limit on the number of layers to display.")

    p_hist = sub.add_parser("weight-hist", help="Plot weight magnitude histogram with pruning threshold.")
    p_hist.add_argument("--checkpoint", type=Path, required=True)
    p_hist.add_argument("--mask", type=Path, required=True)
    p_hist.add_argument("--out", type=Path, required=True)
    p_hist.add_argument("--bins", type=int, default=80)
    p_hist.add_argument("--random-ratio", type=float, default=None)

    p_unprune = sub.add_parser("unprune-timeline", help="Visualise pruning/unpruning schedule.")
    p_unprune.add_argument("--total-epochs", type=int, required=True)
    p_unprune.add_argument("--unprune-epoch", type=int, default=None)
    p_unprune.add_argument("--out", type=Path, required=True)

    # Fisher & soup
    p_fisher = sub.add_parser("fisher-distribution", help="Plot Fisher diagonal distributions per layer.")
    p_fisher.add_argument("--fisher", type=Path, required=True)
    p_fisher.add_argument("--out", type=Path, required=True)
    p_fisher.add_argument("--param-names", type=Path, default=None, help="Optional text file mapping indices to names.")
    p_fisher.add_argument("--log-scale", action="store_true", default=False)

    p_wvf = sub.add_parser("weight-vs-fisher", help="Scatter |w| vs Fisher importance.")
    p_wvf.add_argument("--checkpoint", type=Path, required=True)
    p_wvf.add_argument("--fisher", type=Path, required=True)
    p_wvf.add_argument("--out", type=Path, required=True)
    p_wvf.add_argument("--feature-dim", type=int, required=True)
    p_wvf.add_argument("--dropout", type=float, default=0.7)
    p_wvf.add_argument("--sample", type=int, default=50000)

    p_coeff = sub.add_parser("soup-coefficients", help="Visualise soup coefficient search results.")
    p_coeff.add_argument("--results", type=Path, required=True, help="YAML list with coefficients and scores.")
    p_coeff.add_argument("--metric", type=str, default="roc_auc")
    p_coeff.add_argument("--out", type=Path, required=True)

    p_compare = sub.add_parser("fisher-vs-uniform", help="Compare Fisher-weighted vs uniform soups.")
    p_compare.add_argument("--metrics", type=Path, required=True)
    p_compare.add_argument("--out", type=Path, required=True)
    p_compare.add_argument("--strategy-col", type=str, default="strategy")
    p_compare.add_argument("--roc-col", type=str, default="roc_auc")
    p_compare.add_argument("--pr-col", type=str, default="pr_auc")

    # Performance
    p_sparsity = sub.add_parser("sparsity-vs-auc", help="Scatter ROC/PR vs pruning ratios.")
    p_sparsity.add_argument("--csv", type=Path, required=True)
    p_sparsity.add_argument("--out", type=Path, required=True)

    p_unprune_auc = sub.add_parser("unprune-vs-auc", help="Plot unprune ratio sensitivity.")
    p_unprune_auc.add_argument("--csv", type=Path, required=True)
    p_unprune_auc.add_argument("--out", type=Path, required=True)

    p_eff = sub.add_parser("efficiency", help="Compare speed/memory/params trade-offs.")
    p_eff.add_argument("--csv", type=Path, required=True)
    p_eff.add_argument("--out", type=Path, required=True)

    p_cal = sub.add_parser("calibration", help="Draw calibration curve/ECE proxy.")
    p_cal.add_argument("--predictions", type=Path, required=True)
    p_cal.add_argument("--labels", type=Path, required=True)
    p_cal.add_argument("--out", type=Path, required=True)
    p_cal.add_argument("--bins", type=int, default=15)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.command == "pruning-mask":
        _ensure_dir(args.out_dir)
        plot_pruning_mask_heatmap(args.mask, args.out_dir, args.top_k)
    elif args.command == "weight-hist":
        plot_weight_hist_with_threshold(args.checkpoint, args.mask, args.out, args.bins, args.random_ratio)
    elif args.command == "unprune-timeline":
        plot_unprune_timeline(args.total_epochs, args.unprune_epoch, args.out)
    elif args.command == "fisher-distribution":
        plot_fisher_distribution(args.fisher, args.out, args.param_names, args.log_scale)
    elif args.command == "weight-vs-fisher":
        plot_weight_vs_fisher(args.checkpoint, args.fisher, args.out, args.feature_dim, args.dropout, args.sample)
    elif args.command == "soup-coefficients":
        plot_soup_coefficients(args.results, args.metric, args.out)
    elif args.command == "fisher-vs-uniform":
        plot_fisher_vs_uniform(args.metrics, args.out, args.strategy_col, args.roc_col, args.pr_col)
    elif args.command == "sparsity-vs-auc":
        plot_sparsity_vs_auc(args.csv, args.out)
    elif args.command == "unprune-vs-auc":
        plot_unprune_ratio_vs_auc(args.csv, args.out)
    elif args.command == "efficiency":
        plot_efficiency_tradeoff(args.csv, args.out)
    elif args.command == "calibration":
        plot_calibration_curve(args.predictions, args.labels, args.out, args.bins)
    else:
        raise NotImplementedError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
