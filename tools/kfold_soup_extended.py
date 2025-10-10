"""
kfold_model_soup_extended.py
--------------------------------

This script implements an extended version of a k‑fold model soup
orchestration pipeline.  It generalizes the baseline script provided by
the user to more closely follow the methodology described by Suzuki et
al. (2024) for small‑dataset model soups【392160604737602†L580-L599】.  The
key enhancements include:

* Support for multiple hyperparameter configurations.  The script
  accepts a list of YAML config files via ``--hyperparam_configs``; each
  represents a different hyperparameter setting (e.g., learning rate,
  dropout).  For each config, the data are split into k folds and
  separate models are trained.

* Flexible generation of uniform soups across folds.  For each
  hyperparameter, the script averages the weights of all
  combinations of ``n`` models out of the ``k`` fold models
  (i.e. \( \binom{k}{n} \) combinations) rather than just taking
  the first ``n`` models.  This follows the uniform soup procedure
  described in the paper【392160604737602†L749-L765】.

* Calculation of average and minimum validation metrics (PR AUC and
  ROC AUC) across the folds that contribute to each soup.  These
  correspond to the ``AvgAcc`` and ``MinAcc`` criteria proposed in
  the paper【392160604737602†L771-L780】.

* Optional greedy soup construction.  When ``--greedy`` is specified,
  the script will attempt to build a soup incrementally by adding the
  model that most improves the minimum validation metric, akin to
  greedy soups【392160604737602†L821-L847】.  Greedy soups are
  constructed independently for each hyperparameter configuration.

Note that this script is designed for video anomaly detection (VAD)
experiments and relies on existing code for training (`main.py`),
loading checkpoints (`load_checkpoint`), averaging weights
(`average_state_dicts`), creating datasets (`CreateDataset`), and
computing PR AUC/ROC AUC (`calc_metrics`).  It assumes these modules
are available in the project.  The script orchestrates training and
evaluation, but does not run by itself unless `main.py` and the
associated utilities are present.

Usage example (shell):

```
python3 kfold_model_soup_extended.py \
  --hyperparam_configs config1.yaml config2.yaml config3.yaml \
  --folds 5 \
  --seed 42 \
  --train_flags "--max_epochs 10" \
  --soup_output ckpts/kfold_soup_extended \
  --greedy
```
"""

import argparse
import copy
import itertools
import tempfile
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import shlex
import yaml
from sklearn.model_selection import KFold
import torch
import numpy as np
from scipy.stats import entropy, wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
import sys  # noqa: E402
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.model_soup import average_state_dicts, load_checkpoint
from utils import get_logger, set_seeds, calc_metrics
from model import AD_Model
from data.dataset_loader import CreateDataset


def read_training_split(path: Path) -> List[str]:
    """Read a text file containing training sample identifiers (one per line)."""
    with open(path, 'r') as handle:
        return [line.strip() for line in handle if line.strip()]


def write_split(path: Path, lines: List[str]) -> None:
    """Write a list of sample identifiers to a split file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as handle:
        handle.write('\n'.join(lines) + '\n')


def load_config_dict(config_path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file into a Python dictionary."""
    with open(config_path, 'r') as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def save_config_dict(cfg: Dict[str, Any], path: Path) -> None:
    """Save a configuration dictionary as YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def run_training(fold_config: Path, extra_flags: List[str], ckpt_root: Path) -> Path:
    """Invoke main.py with the given configuration and return the directory of the new checkpoint.

    This function calls a separate training script (`main.py`) using the
    provided config and extra flags.  After training finishes, it locates
    the newly created checkpoint directory under ``ckpt_root`` by
    comparing before/after directory listings.  This assumes that
    ``main.py`` writes outputs under ``ckpt_path``/dataset_name.

    Args:
        fold_config: Path to the temporary YAML config for this fold.
        extra_flags: Additional command line flags passed to ``main.py``.
        ckpt_root: Directory where checkpoints are stored.

    Returns:
        Path to the newly created checkpoint directory.
    """
    existing = {p.name for p in ckpt_root.glob('*') if p.is_dir()}
    cmd = ['python3', 'main.py', '--load_config', str(fold_config)] + extra_flags
    subprocess.run(cmd, check=True)
    # Identify the new checkpoint directory
    candidates = [p for p in ckpt_root.glob('*') if p.is_dir() and p.name not in existing]
    if not candidates:
        raise RuntimeError('No new checkpoint directory found after training.')
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_args_namespace(config_dict: Dict[str, Any]) -> argparse.Namespace:
    """Convert a configuration dictionary into an argparse.Namespace object."""
    return argparse.Namespace(**copy.deepcopy(config_dict))


def build_logger(log_dir: Path, name: str) -> logging.Logger:
    """Create a logger that writes to a file under ``log_dir``."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'{name}.txt'
    logger = get_logger(str(log_path))
    return logger


def evaluate_checkpoint(checkpoint_path: Path, args_dict: Dict[str, Any], split_path: Path,
                        device: torch.device) -> Tuple[float, float]:
    """Evaluate a checkpoint on a validation split and return PR and ROC AUC.

    This helper loads the checkpoint, constructs the dataset loader for the
    given validation split (using the config in ``args_dict``), and
    computes PR AUC and ROC AUC.  It uses ``calc_metrics`` to compute
    these metrics from anomaly scores and labels.

    Args:
        checkpoint_path: Path to the model checkpoint (.pkl) file.
        args_dict: Configuration dictionary for dataset creation and model.
        split_path: Path to the validation split file (txt).
        device: CUDA or CPU device.

    Returns:
        Tuple of (PR AUC, ROC AUC).
    """
    cfg = copy.deepcopy(args_dict)
    # Set the testing split to this validation split
    cfg['testing_split'] = str(split_path)
    ns = load_args_namespace(cfg)
    ns.device = 'cuda' if device.type == 'cuda' else 'cpu'
    ns.gpu_id = device.index if device.type == 'cuda' and device.index is not None else 0
    ns.seed = cfg.get('seed', 1)
    set_seeds(ns.seed)
    logger = build_logger(Path(cfg['logger_path']) / cfg['dataset'], f'eval_{split_path.stem}')
    test_loader, _, _, _ = CreateDataset(ns, logger)

    model = AD_Model(ns.feature_dim, 512, ns.dropout_rate)
    if device.type == 'cuda':
        model.to(device)

    state_dict, _ = load_checkpoint(str(checkpoint_path))
    model.load_state_dict(state_dict)
    model.eval()

    total_scores = []
    total_labels = []
    with torch.no_grad():
        for features, label_frames, _ in test_loader:
            features = features.type(torch.float32).to(device)
            outputs = model(features)
            scores = outputs.squeeze().cpu().numpy()
            for score, label in zip(scores, label_frames[0]):
                total_scores.extend([score] * ns.segment_len)
                total_labels.extend(label.numpy().astype(int).tolist())

    prauc, rocauc = calc_metrics(total_scores, total_labels)
    return prauc, rocauc


def soup_state_dict(paths: List[Path]) -> Dict[str, torch.Tensor]:
    """Average the state dictionaries of multiple checkpoints."""
    state_dicts = [load_checkpoint(str(p))[0] for p in paths]
    return average_state_dicts(state_dicts)


def save_soup_state(state: Dict[str, torch.Tensor], reference_mask: Path, output_path: Path) -> None:
    """Save the averaged state and (optionally) a reference mask to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_path)
    if reference_mask and reference_mask.exists():
        shutil.copy2(reference_mask, output_path.with_suffix(output_path.suffix + '.mask'))


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 20, eps: float = 1e-6) -> float:
    hist_exp, bin_edges = np.histogram(expected, bins=bins)
    hist_act, _ = np.histogram(actual, bins=bin_edges)
    perc_exp = hist_exp / max(hist_exp.sum(), eps)
    perc_act = hist_act / max(hist_act.sum(), eps)
    perc_exp = np.clip(perc_exp, eps, None)
    perc_act = np.clip(perc_act, eps, None)
    psi = np.sum((perc_act - perc_exp) * np.log(perc_act / perc_exp))
    return float(psi)


def compute_distribution_metrics(train_values: np.ndarray, val_values: np.ndarray, bins: int = 20) -> Dict[str, float]:
    eps = 1e-6
    psi = population_stability_index(train_values, val_values, bins=bins, eps=eps)
    hist_train, edges = np.histogram(train_values, bins=bins)
    hist_val, _ = np.histogram(val_values, bins=edges)
    p_train = hist_train / max(hist_train.sum(), eps)
    p_val = hist_val / max(hist_val.sum(), eps)
    p_train = np.clip(p_train, eps, None)
    p_val = np.clip(p_val, eps, None)
    kl = float(entropy(p_train, p_val))
    js = float(jensenshannon(p_train, p_val) ** 2)
    wd = float(wasserstein_distance(train_values, val_values))
    return {
        'psi': psi,
        'kl_divergence': kl,
        'js_divergence': js,
        'wasserstein_distance': wd,
    }


def sample_rows(array: np.ndarray, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    if array.shape[0] <= max_samples:
        return array
    indices = rng.choice(array.shape[0], size=max_samples, replace=False)
    return array[indices]


def plot_histograms(train_values: np.ndarray, val_values: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.hist(train_values, bins=40, alpha=0.6, label='train', density=True)
    plt.hist(val_values, bins=40, alpha=0.6, label='validation', density=True)
    plt.xlabel('Feature norm')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def plot_pca(train_features: np.ndarray, val_features: np.ndarray, path: Path, sample_size: int,
             rng: np.random.Generator) -> None:
    if train_features.shape[0] == 0 or val_features.shape[0] == 0:
        return
    train_sample = sample_rows(train_features, sample_size, rng)
    val_sample = sample_rows(val_features, sample_size, rng)
    combined = np.vstack([train_sample, val_sample])
    if combined.shape[0] < 2:
        return
    pca = PCA(n_components=2)
    pca.fit(combined)
    train_proj = pca.transform(train_sample)
    val_proj = pca.transform(val_sample)
    plt.figure(figsize=(6, 5))
    plt.scatter(train_proj[:, 0], train_proj[:, 1], s=8, alpha=0.5, label='train')
    plt.scatter(val_proj[:, 0], val_proj[:, 1], s=8, alpha=0.5, label='validation')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def gather_dataset_arrays(dataset) -> Dict[str, np.ndarray]:
    features = []
    pseudo_labels = []
    reweights = []
    for info in dataset.video_info_dict.values():
        feat = info.get('feature')
        if isinstance(feat, np.ndarray):
            arr = feat
            if arr.ndim > 2:
                arr = arr.reshape(-1, arr.shape[-1])
            elif arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            features.append(arr.astype(np.float32))
        pseudo = info.get('pseudo_label')
        if isinstance(pseudo, np.ndarray):
            pseudo_labels.append(pseudo.reshape(-1))
        reweight = info.get('reweight')
        if isinstance(reweight, np.ndarray):
            reweights.append(reweight.reshape(-1))
    features_arr = np.vstack(features) if features else np.empty((0, 0))
    pseudo_arr = np.concatenate(pseudo_labels) if pseudo_labels else np.empty((0,))
    reweight_arr = np.concatenate(reweights) if reweights else np.empty((0,))
    return {
        'features': features_arr,
        'pseudo_labels': pseudo_arr,
        'reweights': reweight_arr,
    }


def gather_validation_labels(dataset) -> np.ndarray:
    labels = []
    for info in dataset.video_info_dict.values():
        label = info.get('label_test')
        if isinstance(label, np.ndarray):
            labels.append(label.reshape(-1))
    if labels:
        return np.concatenate(labels)
    return np.empty((0,))


def analyze_fold_data(fold_cfg: Dict[str, Any], device: torch.device, output_dir: Path,
                      max_samples: int, seed_offset: int = 0) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(**copy.deepcopy(fold_cfg))
    args.device = 'cuda' if device.type == 'cuda' else 'cpu'
    args.gpu_id = device.index if device.type == 'cuda' and device.index is not None else 0
    set_seeds(args.seed + seed_offset)
    logger = get_logger(str(output_dir / 'analysis.log'))
    test_loader, train_loader, _, _ = CreateDataset(args, logger)
    train_dataset = train_loader.dataset
    val_dataset = test_loader.dataset

    rng = np.random.default_rng(args.seed + seed_offset)

    train_arrays = gather_dataset_arrays(train_dataset)
    val_arrays = gather_dataset_arrays(val_dataset)

    train_features = train_arrays['features']
    val_features = val_arrays['features']

    train_norms = np.linalg.norm(train_features, axis=1) if train_features.size else np.empty((0,))
    val_norms = np.linalg.norm(val_features, axis=1) if val_features.size else np.empty((0,))

    metrics: Dict[str, Any] = {}
    if train_norms.size and val_norms.size:
        dist_metrics = compute_distribution_metrics(train_norms, val_norms)
        metrics.update({f'norm_{k}': v for k, v in dist_metrics.items()})
        metrics['train_norm_mean'] = float(train_norms.mean())
        metrics['val_norm_mean'] = float(val_norms.mean())
        hist_path = output_dir / 'feature_norm_hist.png'
        plot_histograms(train_norms, val_norms, hist_path)
        metrics['histogram_path'] = str(hist_path)
        if train_features.shape[1] >= 2:
            pca_path = output_dir / 'pca_projection.png'
            plot_pca(train_features, val_features, pca_path, max_samples, rng)
            metrics['pca_path'] = str(pca_path)

    train_pseudo = train_arrays['pseudo_labels']
    if train_pseudo.size:
        metrics['train_anomaly_ratio'] = float(train_pseudo.mean())
        metrics['train_pseudo_std'] = float(train_pseudo.std())

    train_reweight = train_arrays['reweights']
    if train_reweight.size:
        metrics['train_reweight_mean'] = float(train_reweight.mean())
        metrics['train_reweight_std'] = float(train_reweight.std())

    val_labels = gather_validation_labels(val_dataset)
    if val_labels.size:
        metrics['val_anomaly_ratio'] = float(val_labels.mean())

    metrics['train_feature_count'] = int(train_features.shape[0])
    metrics['val_feature_count'] = int(val_features.shape[0])

    return metrics


def build_fold_configs(base_cfg: Dict[str, Any], training_lines: List[str], k: int, seed: int,
                       hyperparam_index: int, work_dir: Path) -> List[Tuple[int, Dict[str, Any], Path, Path, Path]]:
    """Prepare fold configurations for a specific hyperparameter.

    Given a base configuration dictionary and the list of training sample
    identifiers, this function constructs and writes per‑fold
    training/validation split files and corresponding configuration
    dictionaries.  The hyperparameter index is used to namespace the
    working directory structure.

    Args:
        base_cfg: Base configuration dictionary.
        training_lines: List of sample identifiers from the base training split.
        k: Number of folds.
        seed: Random seed for shuffling.
        hyperparam_index: Index of the hyperparameter configuration.
        work_dir: Root directory for temporary files.

    Returns:
        A list of tuples for each fold: (fold_idx, fold_cfg, train_split_file, val_split_file, fold_config_path)
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    fold_configs = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(training_lines), 1):
        train_lines = [training_lines[i] for i in train_idx]
        val_lines = [training_lines[i] for i in val_idx]

        fold_dir = work_dir / f'hp_{hyperparam_index:02d}' / f'fold_{fold_idx:02d}'
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_split_file = fold_dir / 'train_split.txt'
        val_split_file = fold_dir / 'val_split.txt'
        write_split(train_split_file, train_lines)
        write_split(val_split_file, val_lines)

        fold_cfg = copy.deepcopy(base_cfg)
        fold_cfg['training_split'] = str(train_split_file)
        fold_cfg['testing_split'] = str(val_split_file)
        # Use a separate logger and checkpoint path per hyperparam and fold
        fold_cfg['logger_path'] = str(fold_dir / 'logs')
        fold_cfg['ckpt_path'] = str(fold_dir / 'ckpts')
        fold_cfg['seed'] = fold_cfg.get('seed', 1) + fold_idx

        fold_config_path = fold_dir / 'config.yaml'
        save_config_dict(fold_cfg, fold_config_path)

        fold_configs.append((fold_idx, fold_cfg, train_split_file, val_split_file, fold_config_path))
    return fold_configs


def synthesize_uniform_soups(fold_infos: List[Dict[str, Any]], min_size: int, max_size: int,
                             output_dir: Path, device: torch.device,
                             test_split: Path | None = None,
                             base_config: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """Generate uniform soups for a given hyperparameter configuration.

    For a list of fold information dictionaries (each containing at least
    ``ckpt``, ``val_split`` and ``config``), this function creates
    soups by averaging the weights of all combinations of ``n`` models
    (for n ranging from ``min_size`` to ``max_size``).  Each soup is
    evaluated on the validation splits of the models used to create it
    (as in AvgAcc and MinAcc).  The results are recorded in a list of
    dictionaries with relevant metadata.

    Args:
        fold_infos: List of fold info dictionaries for a single hyperparam.
        min_size: Minimum number of models in a soup (must be ≥ 2).
        max_size: Maximum number of models in a soup (≤ len(fold_infos)).
        output_dir: Directory in which to save the soup checkpoints.
        device: Device for evaluation.

    Returns:
        A list of result records.  Each record contains the soup path,
        the number of models, average PR/ROC AUC, and minimum PR/ROC AUC.
    """
    results = []
    num_models_total = len(fold_infos)
    max_size = min(max_size or num_models_total, num_models_total)
    for n in range(max(min_size, 2), max_size + 1):
        # Generate all combinations of n folds
        for combo_indices in itertools.combinations(range(num_models_total), n):
            selected_infos = [fold_infos[i] for i in combo_indices]
            ckpt_paths = [info['ckpt'] for info in selected_infos]
            masks = [info['mask'] for info in selected_infos if info['mask'] is not None]
            reference_mask = masks[-1] if masks else None
            avg_state = average_state_dicts([load_checkpoint(str(p))[0] for p in ckpt_paths])
            # Save soup checkpoint (unique name based on combination indices)
            combo_name = '_'.join(str(idx + 1) for idx in combo_indices)
            soup_path = output_dir / f'soup_{n}of{num_models_total}_{combo_name}.pkl'
            soup_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(avg_state, soup_path)
            if reference_mask is not None:
                shutil.copy2(reference_mask, soup_path.with_suffix('.pkl.mask'))
            # Evaluate on each selected fold's validation split
            pr_list = []
            roc_list = []
            for info in selected_infos:
                pr, roc = evaluate_checkpoint(soup_path, info['config'], info['val_split'], device)
                pr_list.append(pr)
                roc_list.append(roc)
            avg_pr = sum(pr_list) / len(pr_list)
            avg_roc = sum(roc_list) / len(roc_list)
            min_pr = min(pr_list)
            min_roc = min(roc_list)
            test_pr = None
            test_roc = None
            if test_split is not None and base_config is not None:
                test_pr, test_roc = evaluate_checkpoint(
                    soup_path,
                    base_config,
                    test_split,
                    device
                )
            results.append({
                'soup_path': str(soup_path),
                'num_models': n,
                'combo_indices': [idx + 1 for idx in combo_indices],
                'avg_pr_auc': float(avg_pr),
                'avg_roc_auc': float(avg_roc),
                'min_pr_auc': float(min_pr),
                'min_roc_auc': float(min_roc),
                'test_pr_auc': float(test_pr) if test_pr is not None else None,
                'test_roc_auc': float(test_roc) if test_roc is not None else None,
            })
    return results


def construct_greedy_soup(fold_infos: List[Dict[str, Any]], output_dir: Path,
                          device: torch.device,
                          test_split: Path | None = None,
                          base_config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Build a greedy soup across folds based on minimum validation performance.

    This function implements a greedy soup construction similar to
    Algorithm 1 in the model soup literature【392160604737602†L821-L847】.  It
    starts with the single model that achieves the best minimum validation
    metric and iteratively adds models if doing so improves the minimum
    performance across the selected validation splits.  The greedy soup is
    saved to disk and its validation metrics are returned.

    Args:
        fold_infos: List of fold info dicts (must contain ``ckpt``,
            ``val_split``, ``config``).
        output_dir: Directory to save the greedy soup.
        device: Device for evaluation.

    Returns:
        A dictionary summarizing the greedy soup: its path, the indices
        of included folds, and validation metrics.
    """
    # Number of candidate models
    num_models = len(fold_infos)
    # Keep track of selected fold indices and best minimum metric (ROC AUC)
    selected_indices: List[int] = []
    best_min_metric = 0.0
    # Compute individual validation metrics (PR AUC, ROC AUC) for each fold model
    individual_metrics: List[Tuple[float, float]] = []
    for idx, info in enumerate(fold_infos):
        pr, roc = evaluate_checkpoint(info['ckpt'], info['config'], info['val_split'], device)
        individual_metrics.append((pr, roc))
    # Sort candidate indices by descending ROC AUC (you may choose PR AUC instead)
    sorted_indices = sorted(range(num_models), key=lambda i: individual_metrics[i][1], reverse=True)
    soup_state: Dict[str, torch.Tensor] = None
    # Greedy selection: iteratively add models if they improve the minimum ROC AUC across selected validation splits
    for idx in sorted_indices:
        candidate_indices = selected_indices + [idx]
        candidate_infos = [fold_infos[i] for i in candidate_indices]
        # Average checkpoint states of candidate models
        ckpt_paths = [info['ckpt'] for info in candidate_infos]
        avg_state = average_state_dicts([load_checkpoint(str(p))[0] for p in ckpt_paths])
        # Save temporary soup state for evaluation
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        torch.save(avg_state, temp_path)
        # Evaluate candidate soup on validation splits of candidate models
        pr_list: List[float] = []
        roc_list: List[float] = []
        for info in candidate_infos:
            pr, roc = evaluate_checkpoint(temp_path, info['config'], info['val_split'], device)
            pr_list.append(pr)
            roc_list.append(roc)
        # Remove temporary file
        try:
            temp_path.unlink()
        except Exception:
            pass
        candidate_min_roc = min(roc_list)
        # Accept candidate if it improves the minimum ROC AUC
        if candidate_min_roc > best_min_metric:
            best_min_metric = candidate_min_roc
            selected_indices = candidate_indices
            soup_state = avg_state
    # If no model was selected, fall back to the best individual
    if not selected_indices:
        best_idx = sorted_indices[0]
        return {
            'soup_path': str(fold_infos[best_idx]['ckpt']),
            'selected_folds': [best_idx + 1],
            'avg_pr_auc': individual_metrics[best_idx][0],
            'avg_roc_auc': individual_metrics[best_idx][1],
            'min_pr_auc': individual_metrics[best_idx][0],
            'min_roc_auc': individual_metrics[best_idx][1],
        }
    # Save final greedy soup state
    soup_path = output_dir / 'greedy_soup.pkl'
    torch.save(soup_state, soup_path)
    # Evaluate final greedy soup on selected validation splits
    pr_list: List[float] = []
    roc_list: List[float] = []
    for idx in selected_indices:
        info = fold_infos[idx]
        pr, roc = evaluate_checkpoint(soup_path, info['config'], info['val_split'], device)
        pr_list.append(pr)
        roc_list.append(roc)
    avg_pr = sum(pr_list) / len(pr_list)
    avg_roc = sum(roc_list) / len(roc_list)
    min_pr = min(pr_list)
    min_roc = min(roc_list)
    result = {
        'soup_path': str(soup_path),
        'selected_folds': [idx + 1 for idx in selected_indices],
        'avg_pr_auc': float(avg_pr),
        'avg_roc_auc': float(avg_roc),
        'min_pr_auc': float(min_pr),
        'min_roc_auc': float(min_roc),
    }
    if test_split is not None and base_config is not None:
        test_pr, test_roc = evaluate_checkpoint(soup_path, base_config, test_split, device)
        result['test_pr_auc'] = float(test_pr)
        result['test_roc_auc'] = float(test_roc)
    return result


def main():
    parser = argparse.ArgumentParser(description='K‑fold model soup orchestrator with hyperparameter support.')
    parser.add_argument('--config', required=True, help='Base YAML config path')
    parser.add_argument('--hyperparam_configs', nargs='*', help='Optional additional YAML config files representing different hyperparameter settings.')
    parser.add_argument('--folds', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--seed', type=int, default=42, help='Seed for KFold shuffling')
    parser.add_argument('--train_flags', nargs='*', default=[], help='Extra flags passed to main.py for training')
    parser.add_argument('--work_dir', default='kfold_runs_extended', help='Directory to store temporary split/config files')
    parser.add_argument('--soup_output', default='ckpts/kfold_soup_extended', help='Directory to save soup checkpoints')
    parser.add_argument('--min_soup_size', type=int, default=2, help='Minimum number of models in soups')
    parser.add_argument('--max_soup_size', type=int, default=None, help='Maximum number of models in soups')
    parser.add_argument('--greedy', action='store_true', help='Enable greedy soup construction')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU index for evaluation')
    parser.add_argument('--test_split', type=str, default=None, help='Optional test split file to evaluate soups on')
    parser.add_argument('--analyze_folds', action='store_true', help='Compute per-fold data distribution diagnostics')
    parser.add_argument('--analysis_dir', type=str, default=None, help='Directory to save fold analysis artifacts (plots, metrics)')
    parser.add_argument('--analysis_max_samples', type=int, default=5000, help='Maximum samples per split used for analysis/visualization')
    args = parser.parse_args()

    # Flatten train_flags if nested quoting used
    train_flags = []
    for token in args.train_flags:
        if isinstance(token, str) and ' ' in token:
            train_flags.extend(shlex.split(token))
        else:
            train_flags.append(token)
    args.train_flags = train_flags

    base_config_path = Path(args.config).resolve()
    base_cfg = load_config_dict(base_config_path)
    dataset_name = base_cfg['dataset']

    # Determine hyperparameter config paths
    config_paths: List[Path] = [base_config_path]
    if args.hyperparam_configs:
        config_paths = [Path(p).resolve() for p in [args.config] + args.hyperparam_configs]

    # Read base training split
    train_split_path = Path(base_cfg['training_split']).resolve()
    training_lines = read_training_split(train_split_path)

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    all_results: List[Dict[str, Any]] = []
    analysis_results: List[Dict[str, Any]] = []
    test_split_path = Path(args.test_split).resolve() if args.test_split else None

    analysis_root = Path(args.analysis_dir).resolve() if args.analysis_dir else Path(args.soup_output).resolve() / 'analysis'

    for hp_index, cfg_path in enumerate(config_paths):
        # Load hyperparameter-specific config (if different from base)
        hp_cfg = load_config_dict(cfg_path)
        hp_dataset_name = hp_cfg.get('dataset', dataset_name)
        if hp_dataset_name != dataset_name:
            raise ValueError(f'Dataset mismatch: {hp_dataset_name} vs {dataset_name}')
        # Create fold configs for this hyperparam
        fold_configs = build_fold_configs(hp_cfg, training_lines, args.folds, args.seed, hp_index, work_dir)
        fold_infos = []
        for fold_idx, fold_cfg, train_file, val_file, fold_config_path in fold_configs:
            fold_ckpt_root = Path(fold_cfg['ckpt_path']).resolve() / dataset_name
            fold_ckpt_root.mkdir(parents=True, exist_ok=True)
            # Check if a best checkpoint already exists
            existing_best = []
            for ckpt_dir in fold_ckpt_root.glob('*'):
                candidate_best = ckpt_dir / 'best_auc.pkl'
                if candidate_best.exists():
                    existing_best.append(candidate_best)
            if existing_best:
                existing_best.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                best_ckpt_path = existing_best[0]
                # Use existing checkpoint
            else:
                # Train the fold
                ckpt_dir = run_training(fold_config_path, args.train_flags, fold_ckpt_root)
                best_ckpt_path = ckpt_dir / 'best_auc.pkl'
                if not best_ckpt_path.exists():
                    raise FileNotFoundError(f'Expected checkpoint not found: {best_ckpt_path}')
            mask_path = best_ckpt_path.with_suffix('.pkl.mask')
            fold_infos.append({
                'hyperparam_index': hp_index,
                'fold_idx': fold_idx,
                'ckpt': best_ckpt_path,
                'mask': mask_path if mask_path.exists() else None,
                'val_split': val_file,
                'config': fold_cfg,
            })
            if args.analyze_folds:
                fold_analysis_dir = analysis_root / f'hp_{hp_index:02d}' / f'fold_{fold_idx:02d}'
                analysis_metrics = analyze_fold_data(
                    fold_cfg,
                    device,
                    fold_analysis_dir,
                    args.analysis_max_samples,
                    seed_offset=fold_idx
                )
                analysis_entry = {
                    'hyperparam_index': hp_index,
                    'fold_idx': fold_idx,
                    'metrics': analysis_metrics,
                }
                fold_infos[-1]['analysis'] = analysis_metrics
                analysis_results.append(analysis_entry)
        # Directory to save soups for this hyperparameter
        hp_soup_dir = Path(args.soup_output).resolve() / f'hp_{hp_index:02d}'
        hp_soup_dir.mkdir(parents=True, exist_ok=True)
        # Uniform soups
        results_base_cfg = copy.deepcopy(hp_cfg)
        uniform_results = synthesize_uniform_soups(
            fold_infos,
            args.min_soup_size,
            args.max_soup_size or len(fold_infos),
            hp_soup_dir,
            device,
            test_split=test_split_path,
            base_config=results_base_cfg if test_split_path is not None else None
        )
        for rec in uniform_results:
            rec['hyperparam_index'] = hp_index
            rec['soup_type'] = 'uniform'
            all_results.append(rec)
        # Greedy soup if requested
        if args.greedy:
            greedy_result = construct_greedy_soup(
                fold_infos,
                hp_soup_dir,
                device,
                test_split=test_split_path,
                base_config=results_base_cfg if test_split_path is not None else None
            )
            greedy_result['hyperparam_index'] = hp_index
            greedy_result['soup_type'] = 'greedy'
            all_results.append(greedy_result)

    # Save consolidated results YAML
    soup_dir = Path(args.soup_output).resolve()
    soup_dir.mkdir(parents=True, exist_ok=True)
    results_path = soup_dir / 'kfold_soup_extended_results.yaml'
    with open(results_path, 'w') as handle:
        yaml.safe_dump(all_results, handle, sort_keys=False)
    if args.analyze_folds:
        analysis_path = soup_dir / 'kfold_fold_analysis.yaml'
        with open(analysis_path, 'w') as handle:
            yaml.safe_dump(analysis_results, handle, sort_keys=False)
        print(f'Fold analysis written to {analysis_path}')
    print(f'Extended k-fold soup results written to {results_path}')


if __name__ == '__main__':
    main()
