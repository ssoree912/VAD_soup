import argparse
import csv
import logging
import os
import sys
from typing import List, Optional

import torch
import yaml

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import AD_Model
from data.dataset_loader import CreateDataset
from utils import calc_metrics, set_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate checkpoints and export frame-level metrics to CSV.'
    )
    parser.add_argument('--ckpts', nargs='+', required=True,
                        help='Checkpoint paths to evaluate.')
    parser.add_argument('--labels', nargs='*', default=None,
                        help='Optional labels for each checkpoint (matches order of --ckpts).')
    parser.add_argument('--config', required=True,
                        help='Configuration YAML used to instantiate the dataset/model.')
    parser.add_argument('--output', required=True,
                        help='CSV path to write evaluation results.')
    parser.add_argument('--device', default=None,
                        help='Override device for evaluation (cpu or cuda).')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU index when using CUDA.')
    return parser.parse_args()


def load_config(config_path: str) -> argparse.Namespace:
    with open(config_path, 'r') as handle:
        cfg = yaml.load(handle, Loader=yaml.FullLoader)
    return argparse.Namespace(**cfg)


def evaluate_model(model: AD_Model,
                   test_loader,
                   device: torch.device) -> dict:
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
                total_scores.extend([score] * test_loader.dataset.seg_len)
                total_labels.extend(label.detach().cpu().numpy().astype(int).tolist())

    prauc_frames, rocauc_frames = calc_metrics(total_scores, total_labels)
    return {
        'prauc': prauc_frames,
        'rocauc': rocauc_frames,
    }


def main():
    args = parse_args()
    logger = logging.getLogger('evaluate_ckpts')
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    cfg = load_config(args.config)

    if args.device:
        cfg.device = args.device
    if args.gpu_id is not None:
        cfg.gpu_id = args.gpu_id

    set_seeds(cfg.seed)

    if cfg.device == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{cfg.gpu_id}')
    elif cfg.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info('Using Apple Silicon GPU (MPS)')
    elif cfg.device in ['cuda', 'mps']:
        device = torch.device('cpu')
        logger.warning(f'{cfg.device.upper()} requested but not available. Falling back to CPU.')
    else:
        device = torch.device('cpu')

    logger.info('Preparing dataset with config: %s', args.config)
    test_loader, _, _, _ = CreateDataset(cfg, logger)

    ckpt_labels: Optional[List[str]] = args.labels
    if ckpt_labels and len(ckpt_labels) != len(args.ckpts):
        raise ValueError('--labels length must match number of checkpoints')

    results = []

    for idx, ckpt_path in enumerate(args.ckpts):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

        logger.info('Evaluating checkpoint (%d/%d): %s', idx + 1, len(args.ckpts), ckpt_path)
        state_dict = torch.load(ckpt_path, map_location=device)
        model = AD_Model(cfg.feature_dim, 512, cfg.dropout_rate)
        model.load_state_dict(state_dict)
        model.to(device)

        metrics = evaluate_model(model, test_loader, device)

        results.append({
            'label': ckpt_labels[idx] if ckpt_labels else '',
            'checkpoint_path': ckpt_path,
            'prauc_frames': metrics['prauc'],
            'rocauc_frames': metrics['rocauc'],
        })

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['label', 'checkpoint_path', 'prauc_frames', 'rocauc_frames'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    logger.info('Saved evaluation results to %s', args.output)


if __name__ == '__main__':
    main()

