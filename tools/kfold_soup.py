import argparse
import copy
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

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.model_soup import average_state_dicts, load_checkpoint
from utils import get_logger, set_seeds, calc_metrics
from model import AD_Model
from data.dataset_loader import CreateDataset


def read_training_split(path: Path) -> List[str]:
    with open(path, 'r') as handle:
        return [line.strip() for line in handle if line.strip()]


def write_split(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as handle:
        handle.write('\n'.join(lines) + '\n')


def load_config_dict(config_path: Path) -> Dict[str, Any]:
    with open(config_path, 'r') as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def save_config_dict(cfg: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def run_training(fold_config: Path, extra_flags: List[str], ckpt_root: Path) -> Path:
    existing = {p.name for p in ckpt_root.glob('*') if p.is_dir()}
    cmd = ['python3', 'main.py', '--load_config', str(fold_config)] + extra_flags
    completed = subprocess.run(cmd, check=True)
    _ = completed
    candidates = [p for p in ckpt_root.glob('*') if p.is_dir() and p.name not in existing]
    if not candidates:
        raise RuntimeError('No new checkpoint directory found after training.')
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_args_namespace(config_dict: Dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(**copy.deepcopy(config_dict))


def build_logger(log_dir: Path, name: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'{name}.txt'
    logger = get_logger(str(log_path))
    return logger


def evaluate_checkpoint(checkpoint_path: Path, args_dict: Dict[str, Any], split_path: Path,
                        device: torch.device) -> Tuple[float, float]:
    cfg = copy.deepcopy(args_dict)
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


def soup_state_dict(paths: List[Path]) -> torch.nn.Module:
    state_dicts = [load_checkpoint(str(p))[0] for p in paths]
    return average_state_dicts(state_dicts)


def save_soup_state(state: Dict[str, torch.Tensor], mask_path: Path, reference_mask: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_path)
    if reference_mask and reference_mask.exists():
        shutil.copy2(reference_mask, output_path.with_suffix(output_path.suffix + '.mask'))


def main():
    parser = argparse.ArgumentParser(description='K-fold model soup orchestrator for ShanghaiTech VAD.')
    parser.add_argument('--config', required=True, help='Base YAML config path')
    parser.add_argument('--folds', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--seed', type=int, default=42, help='Seed for KFold shuffling')
    parser.add_argument('--train_flags', nargs='*', default=[], help='Extra flags passed to main.py')
    parser.add_argument('--work_dir', default='kfold_runs', help='Directory to store temporary split/config files')
    parser.add_argument('--soup_output', default='ckpts/kfold_soup', help='Directory to save soup checkpoints')
    parser.add_argument('--min_soup_size', type=int, default=2, help='Minimum number of models in soups')
    parser.add_argument('--max_soup_size', type=int, default=None, help='Maximum number of models in soups')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU index for evaluation')
    args = parser.parse_args()

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

    train_split_path = Path(base_cfg['training_split']).resolve()
    training_lines = read_training_split(train_split_path)

    k = args.folds
    kf = KFold(n_splits=k, shuffle=True, random_state=args.seed)

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    ckpt_root = Path(base_cfg['ckpt_path']).resolve() / dataset_name
    ckpt_root.mkdir(parents=True, exist_ok=True)

    fold_infos = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(training_lines), 1):
        train_lines = [training_lines[i] for i in train_idx]
        val_lines = [training_lines[i] for i in val_idx]

        fold_dir = work_dir / f'fold_{fold_idx:02d}'
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_split_file = fold_dir / 'train_split.txt'
        val_split_file = fold_dir / 'val_split.txt'
        write_split(train_split_file, train_lines)
        write_split(val_split_file, val_lines)

        fold_cfg = copy.deepcopy(base_cfg)
        fold_cfg['training_split'] = str(train_split_file)
        fold_cfg['testing_split'] = str(val_split_file)
        fold_cfg['logger_path'] = str(fold_dir / 'logs')
        fold_cfg['ckpt_path'] = str(fold_dir / 'ckpts')
        fold_cfg['seed'] = fold_cfg.get('seed', 1) + fold_idx

        fold_config_path = fold_dir / 'config.yaml'
        save_config_dict(fold_cfg, fold_config_path)

        fold_ckpt_root = Path(fold_cfg['ckpt_path']).resolve() / dataset_name
        fold_ckpt_root.mkdir(parents=True, exist_ok=True)

        existing_best = []
        for ckpt_dir in fold_ckpt_root.glob('*'):
            candidate_best = ckpt_dir / 'best_auc.pkl'
            if candidate_best.exists():
                existing_best.append(candidate_best)

        if existing_best:
            existing_best.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            best_ckpt_path = existing_best[0]
            print(f'[Fold {fold_idx}] Reusing existing checkpoint: {best_ckpt_path.parent.name}')
        else:
            ckpt_dir = run_training(fold_config_path, args.train_flags, fold_ckpt_root)
            best_ckpt_path = ckpt_dir / 'best_auc.pkl'
            if not best_ckpt_path.exists():
                raise FileNotFoundError(f'Expected checkpoint not found: {best_ckpt_path}')
            print(f'[Fold {fold_idx}] Trained new checkpoint: {best_ckpt_path.parent.name}')

        mask_path = best_ckpt_path.with_suffix('.pkl.mask')

        fold_infos.append({
            'fold_idx': fold_idx,
            'train_split': train_split_file,
            'val_split': val_split_file,
            'config': fold_cfg,
            'ckpt': best_ckpt_path,
            'mask': mask_path if mask_path.exists() else None,
        })

    soup_dir = Path(args.soup_output).resolve()
    soup_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    max_soup = args.max_soup_size or len(fold_infos)
    records = []
    for soup_size in range(max(args.min_soup_size, 2), max_soup + 1):
        selected_folds = fold_infos[:soup_size]
        ckpt_paths = [info['ckpt'] for info in selected_folds]
        masks = [info['mask'] for info in selected_folds if info['mask'] is not None]
        reference_mask = masks[-1] if masks else None

        avg_state = average_state_dicts([load_checkpoint(str(p))[0] for p in ckpt_paths])
        soup_path = soup_dir / f'kfold_soup_{soup_size}of{k}.pkl'
        torch.save(avg_state, soup_path)
        if reference_mask is not None:
            shutil.copy2(reference_mask, soup_path.with_suffix('.pkl.mask'))

        fold_metrics = []
        for info in fold_infos:
            pr_auc, ro_auc = evaluate_checkpoint(soup_path, info['config'], info['val_split'], device)
            fold_metrics.append((pr_auc, ro_auc))

        avg_pr = sum(m[0] for m in fold_metrics) / len(fold_metrics)
        avg_roc = sum(m[1] for m in fold_metrics) / len(fold_metrics)
        min_pr = min(m[0] for m in fold_metrics)
        min_roc = min(m[1] for m in fold_metrics)

        records.append({
            'soup_path': str(soup_path),
            'num_models': int(soup_size),
            'avg_pr_auc': float(avg_pr),
            'avg_roc_auc': float(avg_roc),
            'min_pr_auc': float(min_pr),
            'min_roc_auc': float(min_roc),
        })

    results_path = soup_dir / 'kfold_soup_results.yaml'
    with open(results_path, 'w') as handle:
        yaml.safe_dump(records, handle, sort_keys=False)

    print(f'K-fold soups saved to {soup_dir}')
    print(f'Results written to {results_path}')


if __name__ == '__main__':
    main()
