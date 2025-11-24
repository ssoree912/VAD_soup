"""Batch training and model soup orchestration.

This utility automates the workflow of:
1. Training multiple LANP-UVAD models with different seeds (and optional
   pruning random seeds).
2. Collecting the resulting checkpoints.
3. Building uniform and/or greedy model soups.
4. Optionally evaluating individual checkpoints and soups on the test split
   defined in the base configuration.

Example usage:

```
python3 tools/batch_model_soup.py \
  --config config/config_sh.yaml \
  --seeds 1 2 3 \
  --train_flags "--device cuda --gpu_id 0 --use_pruning --prune_magnitude_ratio 0.05 --prune_random_ratio 0.05" \
  --uniform_output ckpts/batch_soup/uniform.pkl \
  --greedy_output ckpts/batch_soup/greedy.pkl \
  --evaluate

python3 tools/batch_model_soup.py \
  --config config/config_sh.yaml \
  --seeds 1 \
  --prune_random_seeds 1001 1002 1003 \
  --train_flags "--device cuda --gpu_id 0 --use_pruning --prune_magnitude_ratio 0.05 --prune_random_ratio 0.05 --use_early_unprune --unprune_ratio 0.01" \
  --uniform_output ckpts/batch_soup/random_uniform.pkl \
  --evaluate
```

The script assumes the existence of ``main.py`` for training and relies on
utility functions from the repository such as ``average_state_dicts`` and
``CreateDataset``.  It produces a summary YAML file alongside the soup
checkpoints that captures individual and soup metrics.
"""

from __future__ import annotations

import argparse
import copy
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

import yaml

ROOT_DIR = Path(__file__).resolve().parent
# Repository root (two levels up)
PROJECT_ROOT = ROOT_DIR.parent.parent
import sys  # noqa: E402
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.soup.model_soup import average_state_dicts, load_checkpoint
from utils import get_logger, set_seeds, calc_metrics
from model import AD_Model
from data.dataset_loader import CreateDataset


def parse_train_flags(tokens: List[str]) -> List[str]:
    parsed: List[str] = []
    for token in tokens:
        if isinstance(token, str) and ' ' in token:
            parsed.extend(shlex.split(token))
        else:
            parsed.append(token)
    return parsed


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, 'r') as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def run_training(config_path: Path, extra_flags: List[str], ckpt_root: Path) -> Path:
    existing = {p.name for p in ckpt_root.glob('*') if p.is_dir()}
    cmd = ['python3', 'main.py', '--load_config', str(config_path)] + extra_flags
    subprocess.run(cmd, check=True)
    candidates = [p for p in ckpt_root.glob('*') if p.is_dir() and p.name not in existing]
    if not candidates:
        raise RuntimeError('No new checkpoint directory found after training.')
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def build_args_namespace(cfg: Dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(**copy.deepcopy(cfg))


def evaluate_checkpoint(checkpoint_path: Path, cfg: Dict[str, Any], device: torch.device) -> Tuple[float, float]:
    args = build_args_namespace(cfg)
    set_seeds(args.seed)
    args.device = 'cuda' if device.type == 'cuda' else 'cpu'
    args.gpu_id = device.index if device.type == 'cuda' and device.index is not None else 0
    log_dir = Path(cfg['logger_path']) / 'batch_eval' / cfg['dataset']
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(str(log_dir / 'batch_eval.txt'))
    test_loader, _, _, _ = CreateDataset(args, logger)

    model = AD_Model(args.feature_dim, 512, args.dropout_rate)
    if device.type == 'cuda':
        model.to(device)

    state_dict, _ = load_checkpoint(str(checkpoint_path))
    model.load_state_dict(state_dict)
    model.eval()

    total_scores: List[float] = []
    total_labels: List[int] = []
    with torch.no_grad():
        for features, label_frames, _ in test_loader:
            features = features.type(torch.float32).to(device)
            outputs = model(features)
            scores = outputs.squeeze().cpu().numpy()
            for score, label in zip(scores, label_frames[0]):
                total_scores.extend([score] * args.segment_len)
                total_labels.extend(label.numpy().astype(int).tolist())

    pr_auc, roc_auc = calc_metrics(total_scores, total_labels)
    return float(pr_auc), float(roc_auc)


def apply_mask_to_state(state: Dict[str, torch.Tensor], mask: Dict[str, torch.Tensor]) -> None:
    for key, tensor in mask.items():
        if key in state:
            state[key] = state[key] * tensor.to(state[key].dtype)


def save_state_and_mask(state: Dict[str, torch.Tensor], mask: Optional[Dict[str, torch.Tensor]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_path)
    mask_path = output_path.with_suffix(output_path.suffix + '.mask')
    if mask:
        torch.save(mask, mask_path)
    elif mask_path.exists():
        mask_path.unlink()


def average_state_and_mask(ckpt_paths: Iterable[Path]) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
    state_dicts: List[Dict[str, torch.Tensor]] = []
    reference_mask: Optional[Dict[str, torch.Tensor]] = None
    for path in ckpt_paths:
        state, mask = load_checkpoint(str(path))
        state_dicts.append(state)
        if mask is not None:
            reference_mask = mask
    avg_state = average_state_dicts(state_dicts)
    if reference_mask is not None:
        apply_mask_to_state(avg_state, reference_mask)
    return avg_state, reference_mask


def build_uniform_soup(ckpt_paths: List[Path], output_path: Path) -> Dict[str, torch.Tensor]:
    avg_state, reference_mask = average_state_and_mask(ckpt_paths)
    save_state_and_mask(avg_state, reference_mask, output_path)
    return avg_state


def construct_greedy_soup(ckpt_paths: List[Path], eval_cfg: Dict[str, Any], device: torch.device,
                          output_path: Path) -> Tuple[Dict[str, torch.Tensor], List[int], float, float, float, float]:
    if not ckpt_paths:
        raise ValueError('No checkpoints provided for greedy soup.')

    # Evaluate individual models
    metrics: List[Tuple[float, float]] = []
    for path in ckpt_paths:
        metrics.append(evaluate_checkpoint(path, eval_cfg, device))

    sorted_indices = sorted(range(len(ckpt_paths)), key=lambda i: metrics[i][1], reverse=True)
    selected: List[int] = []
    best_metric = -float('inf')
    soup_state: Optional[Dict[str, torch.Tensor]] = None
    soup_mask: Optional[Dict[str, torch.Tensor]] = None
    best_pr = 0.0
    best_roc = 0.0

    for idx in sorted_indices:
        candidates = selected + [idx]
        state, mask = average_state_and_mask(ckpt_paths[i] for i in candidates)
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            temp_path = Path(tmp.name)
        save_state_and_mask(state, mask, temp_path)
        pr, roc = evaluate_checkpoint(temp_path, eval_cfg, device)
        try:
            temp_path.unlink()
            mask_path = temp_path.with_suffix(temp_path.suffix + '.mask')
            if mask_path.exists():
                mask_path.unlink()
        except Exception:
            pass
        if roc > best_metric:
            best_metric = roc
            selected = candidates
            soup_state = state
            soup_mask = mask
            best_pr = pr
            best_roc = roc

    if not selected:
        idx = sorted_indices[0]
        soup_state, soup_mask = average_state_and_mask([ckpt_paths[idx]])
        save_state_and_mask(soup_state, soup_mask, output_path)
        pr, roc = evaluate_checkpoint(output_path, eval_cfg, device)
        selected = [idx]
        return soup_state, [i + 1 for i in selected], pr, roc, pr, roc

    # Final metrics with selected soup
    save_state_and_mask(soup_state, soup_mask, output_path)
    return soup_state, [i + 1 for i in selected], best_pr, best_roc, best_pr, best_roc


def main():
    parser = argparse.ArgumentParser(description='Batch train models and build soups.')
    parser.add_argument('--config', required=True, help='Base YAML config path')
    parser.add_argument('--seeds', nargs='+', type=int, help='List of seeds for training runs')
    parser.add_argument('--prune_random_seeds', nargs='*', type=int, help='Optional list of pruning random seeds')
    parser.add_argument('--train_flags', nargs='*', default=[], help='Extra flags passed to main.py during training')
    parser.add_argument('--device', default='cuda', help='Device string passed to main.py (default: cuda)')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU index for training/evaluation')
    parser.add_argument('--uniform_output', type=str, help='Path to save uniform soup checkpoint')
    parser.add_argument('--greedy_output', type=str, help='Path to save greedy soup checkpoint')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate individual and soup models on testing split')
    parser.add_argument('--results_yaml', type=str, help='Optional path to write summary YAML (default derived from output)')
    parser.add_argument('--checkpoint_paths', nargs='+', help='Existing checkpoint files to include in soups (skip training)')
    args = parser.parse_args()

    if not args.checkpoint_paths and not args.seeds:
        parser.error('Provide --seeds to trigger training runs or --checkpoint_paths to reuse existing models.')

    train_flags = parse_train_flags(args.train_flags)

    config_path = Path(args.config).resolve()
    base_cfg = load_config(config_path)
    dataset = base_cfg['dataset']
    train_split_path = Path(base_cfg['training_split']).resolve()
    ckpt_root = Path(base_cfg['ckpt_path']).resolve() / dataset
    ckpt_root.mkdir(parents=True, exist_ok=True)

    checkpoint_paths: List[Path] = []
    run_records: List[Dict[str, Any]] = []

    if args.checkpoint_paths:
        for path_str in args.checkpoint_paths:
            candidate = Path(path_str).expanduser().resolve()
            if not candidate.exists():
                raise FileNotFoundError(f'Checkpoint not found: {candidate}')
            checkpoint_paths.append(candidate)
            run_records.append({
                'checkpoint': str(candidate),
                'source': 'provided',
            })

    combinations: List[Tuple[int, Optional[int]]] = []
    if args.seeds:
        prune_list = args.prune_random_seeds if args.prune_random_seeds else [None]
        for seed in args.seeds:
            for prune_seed in prune_list:
                combinations.append((seed, prune_seed))

    for seed, prune_seed in combinations:
        extra_flags = train_flags.copy()
        extra_flags.extend(['--seed', str(seed)])
        if args.device:
            extra_flags.extend(['--device', args.device])
        extra_flags.extend(['--gpu_id', str(args.gpu_id)])
        if prune_seed is not None:
            extra_flags.extend(['--prune_random_seed', str(prune_seed)])

        ckpt_dir = run_training(config_path, extra_flags, ckpt_root)
        best_path = ckpt_dir / 'best_auc.pkl'
        if not best_path.exists():
            raise FileNotFoundError(f'best_auc.pkl not found in {ckpt_dir}')

        checkpoint_paths.append(best_path)
        record = {
            'seed': seed,
            'checkpoint': str(best_path),
        }
        if prune_seed is not None:
            record['prune_random_seed'] = prune_seed

        run_records.append(record)

    device = torch.device(f'cuda:{args.gpu_id}' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

    if not checkpoint_paths:
        raise RuntimeError('No checkpoints available for soup construction.')

    test_cfg = None
    val_cfg = None
    if args.evaluate:
        test_cfg = copy.deepcopy(base_cfg)
        test_cfg['logger_path'] = str(Path(test_cfg['logger_path']) / 'batch_eval')
        val_cfg = copy.deepcopy(base_cfg)
        val_cfg['logger_path'] = str(Path(val_cfg['logger_path']) / 'batch_eval_val')
        val_cfg['testing_split'] = str(train_split_path)

    individual_metrics = []
    if args.evaluate:
        for path in checkpoint_paths:
            test_pr, test_roc = evaluate_checkpoint(path, test_cfg, device)
            val_pr, val_roc = evaluate_checkpoint(path, val_cfg, device)
            individual_metrics.append({
                'checkpoint': str(path),
                'val_pr_auc': val_pr,
                'val_roc_auc': val_roc,
                'test_pr_auc': test_pr,
                'test_roc_auc': test_roc,
            })

    results: Dict[str, Any] = {
        'config': str(config_path),
        'runs': run_records,
    }

    if args.uniform_output:
        uniform_path = Path(args.uniform_output).resolve()
        uniform_state = build_uniform_soup(checkpoint_paths, uniform_path)
        soup_metrics = None
        if args.evaluate and test_cfg is not None:
            val_pr, val_roc = evaluate_checkpoint(uniform_path, val_cfg, device)
            test_pr, test_roc = evaluate_checkpoint(uniform_path, test_cfg, device)
            soup_metrics = {
                'val_pr_auc': val_pr,
                'val_roc_auc': val_roc,
                'test_pr_auc': test_pr,
                'test_roc_auc': test_roc,
            }
        results['uniform_soup'] = {
            'checkpoint': str(uniform_path),
            'metrics': soup_metrics,
        }

    if args.greedy_output:
        greedy_path = Path(args.greedy_output).resolve()
        greedy_eval_cfg = val_cfg if val_cfg is not None else base_cfg
        state, selected, avg_pr, avg_roc, min_pr, min_roc = construct_greedy_soup(
            checkpoint_paths,
            greedy_eval_cfg,
            device,
            greedy_path
        )
        greedy_metrics = None
        if args.evaluate and test_cfg is not None:
            val_pr, val_roc = evaluate_checkpoint(greedy_path, val_cfg, device)
            test_pr, test_roc = evaluate_checkpoint(greedy_path, test_cfg, device)
            greedy_metrics = {
                'val_pr_auc': val_pr,
                'val_roc_auc': val_roc,
                'test_pr_auc': test_pr,
                'test_roc_auc': test_roc,
            }
        results['greedy_soup'] = {
            'checkpoint': str(greedy_path),
            'selected_indices': selected,
            'avg_pr_auc': avg_pr,
            'avg_roc_auc': avg_roc,
            'min_pr_auc': min_pr,
            'min_roc_auc': min_roc,
            'metrics': greedy_metrics,
        }

    if individual_metrics:
        results['individual_metrics'] = individual_metrics

    results_path = Path(args.results_yaml).resolve() if args.results_yaml else None
    if not results_path:
        default_name = 'batch_soup_results.yaml'
        base_dir = Path(args.uniform_output or args.greedy_output or '.')
        if base_dir.suffix:
            base_dir = base_dir.parent
        results_path = base_dir / default_name

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as handle:
        yaml.safe_dump(results, handle, sort_keys=False)
    print(f'Batch soup summary written to {results_path}')


if __name__ == '__main__':
    main()
