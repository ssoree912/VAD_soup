#!/usr/bin/env python3
"""Run DWA pruning across seeds and build model soups."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from tools.batch_model_soup import (
    parse_train_flags,
    load_config,
    run_training,
    evaluate_checkpoint,
    build_uniform_soup,
    construct_greedy_soup,
)


def append_flag(flags: List[str], name: str, value: Optional[str] = None) -> None:
    flags.append(name)
    if value is not None:
        flags.append(value)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run DWA pruning for multiple seeds and build soups.')
    parser.add_argument('--config', required=True, help='Base YAML config path')
    parser.add_argument('--seeds', nargs='+', type=int, required=True, help='Seeds for independent training runs')
    parser.add_argument('--prune_random_seeds', nargs='*', type=int, help='Optional pruning random seeds')
    parser.add_argument('--train_flags', nargs='*', default=[], help='Additional flags forwarded to main.py')
    parser.add_argument('--device', default='cuda', help='Device passed to main.py (default: cuda)')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU index for training/evaluation')
    parser.add_argument('--uniform_output', type=str, required=True, help='Output path for uniform soup checkpoint')
    parser.add_argument('--greedy_output', type=str, required=True, help='Output path for greedy soup checkpoint')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate individual checkpoints and soups')
    parser.add_argument('--results_yaml', type=str, help='Optional consolidated results YAML path')
    parser.add_argument('--prune_ratio', type=float, default=0.10, help='Fraction of parameters pruned by magnitude (default: 0.10)')
    parser.add_argument('--dwa_alpha', type=float, default=0.1, help='Acceleration term coefficient alpha')
    parser.add_argument('--dwa_beta', type=float, default=1.0, help='Base scaling coefficient beta')
    args = parser.parse_args()

    train_flags = parse_train_flags(args.train_flags)

    config_path = Path(args.config).resolve()
    base_cfg = load_config(config_path)
    dataset = base_cfg['dataset']
    train_split_path = Path(base_cfg['training_split']).resolve()
    ckpt_root = Path(base_cfg['ckpt_path']).resolve() / dataset
    ckpt_root.mkdir(parents=True, exist_ok=True)

    prune_list = args.prune_random_seeds if args.prune_random_seeds else [None]

    checkpoint_paths: List[Path] = []
    run_records: List[Dict[str, Any]] = []

    for seed in args.seeds:
        for prune_seed in prune_list:
            extra_flags = list(train_flags)
            append_flag(extra_flags, '--seed', str(seed))
            if args.device:
                append_flag(extra_flags, '--device', args.device)
            append_flag(extra_flags, '--gpu_id', str(args.gpu_id))
            append_flag(extra_flags, '--use_pruning')
            append_flag(extra_flags, '--prune_magnitude_ratio', str(args.prune_ratio))
            append_flag(extra_flags, '--prune_random_ratio', '0.0')
            append_flag(extra_flags, '--pruning_strategy', 'dwa_kill_and_reactivate')
            append_flag(extra_flags, '--dwa_alpha', str(args.dwa_alpha))
            append_flag(extra_flags, '--dwa_beta', str(args.dwa_beta))
            if prune_seed is not None:
                append_flag(extra_flags, '--prune_random_seed', str(prune_seed))

            ckpt_dir = run_training(config_path, extra_flags, ckpt_root)
            best_path = ckpt_dir / 'best_auc.pkl'
            if not best_path.exists():
                raise FileNotFoundError(f'best_auc.pkl not found in {ckpt_dir}')

            checkpoint_paths.append(best_path)
            record: Dict[str, Any] = {
                'seed': seed,
                'checkpoint': str(best_path),
                'prune_ratio': args.prune_ratio,
                'dwa_alpha': args.dwa_alpha,
                'dwa_beta': args.dwa_beta,
            }
            if prune_seed is not None:
                record['prune_random_seed'] = prune_seed
            run_records.append(record)

    if not checkpoint_paths:
        raise RuntimeError('No checkpoints produced for soup construction.')

    device = torch.device(f'cuda:{args.gpu_id}' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

    test_cfg = None
    val_cfg = None
    if args.evaluate:
        test_cfg = copy.deepcopy(base_cfg)
        test_cfg['logger_path'] = str(Path(test_cfg['logger_path']) / 'dwa_soup_eval')
        val_cfg = copy.deepcopy(base_cfg)
        val_cfg['logger_path'] = str(Path(val_cfg['logger_path']) / 'dwa_soup_eval_val')
        val_cfg['testing_split'] = str(train_split_path)

    individual_metrics: List[Dict[str, Any]] = []
    if args.evaluate and test_cfg is not None and val_cfg is not None:
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
        'dwa_settings': {
            'prune_ratio': args.prune_ratio,
            'dwa_alpha': args.dwa_alpha,
            'dwa_beta': args.dwa_beta,
        },
    }

    uniform_path = Path(args.uniform_output).resolve()
    uniform_state = build_uniform_soup(checkpoint_paths, uniform_path)
    uniform_metrics = None
    if args.evaluate and test_cfg is not None and val_cfg is not None:
        val_pr, val_roc = evaluate_checkpoint(uniform_path, val_cfg, device)
        test_pr, test_roc = evaluate_checkpoint(uniform_path, test_cfg, device)
        uniform_metrics = {
            'val_pr_auc': val_pr,
            'val_roc_auc': val_roc,
            'test_pr_auc': test_pr,
            'test_roc_auc': test_roc,
        }
    results['uniform_soup'] = {
        'checkpoint': str(uniform_path),
        'metrics': uniform_metrics,
    }

    greedy_path = Path(args.greedy_output).resolve()
    greedy_eval_cfg = val_cfg if val_cfg is not None else base_cfg
    _, selected, avg_pr, avg_roc, min_pr, min_roc = construct_greedy_soup(
        checkpoint_paths,
        greedy_eval_cfg,
        device,
        greedy_path
    )
    greedy_metrics = None
    if args.evaluate and test_cfg is not None and val_cfg is not None:
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
        base_dir = uniform_path.parent
        results_path = base_dir / 'dwa_soup_results.yaml'

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as handle:
        import yaml  # local import to avoid mandatory dependency when only training
        yaml.safe_dump(results, handle, sort_keys=False)
    print(f'DWA soup summary written to {results_path}')


if __name__ == '__main__':
    main()
