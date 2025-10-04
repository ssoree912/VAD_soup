import argparse
import csv
import os
from typing import Dict, List, Optional

import yaml


CHECKPOINT_CANDIDATES = (
    ('best_auc.pkl', 'best_auc'),
    ('best_pr.pkl', 'best_pr'),
    ('last_epoch.pkl', 'last_epoch'),
)


def load_config(run_dir: str) -> Dict:
    config_path = os.path.join(run_dir, 'config_used.yaml')
    if not os.path.isfile(config_path):
        return {}
    with open(config_path, 'r') as handle:
        data = yaml.safe_load(handle) or {}
    return data


def collect_run_rows(run_dir: str) -> List[Dict]:
    config = load_config(run_dir)
    rows = []

    for filename, ckpt_type in CHECKPOINT_CANDIDATES:
        ckpt_path = os.path.join(run_dir, filename)
        if not os.path.isfile(ckpt_path):
            continue

        row = {
            'dataset': config.get('dataset'),
            'seed': config.get('seed'),
            'learning_rate': config.get('lr'),
            'checkpoint_type': ckpt_type,
            'checkpoint_path': ckpt_path,
            'run_directory': run_dir,
            'config_path': os.path.join(run_dir, 'config_used.yaml') if config else '',
        }
        rows.append(row)

    return rows


def collect_soup_rows(soup_paths: List[str], label_prefix: Optional[str] = None) -> List[Dict]:
    rows = []
    for soup_path in soup_paths:
        if not os.path.isfile(soup_path):
            raise FileNotFoundError(f'Soup checkpoint not found: {soup_path}')
        soup_dir = os.path.dirname(soup_path)
        soup_name = os.path.splitext(os.path.basename(soup_path))[0]
        ckpt_type = f'{label_prefix}_{soup_name}' if label_prefix else soup_name
        rows.append({
            'dataset': '',
            'seed': '',
            'learning_rate': '',
            'checkpoint_type': ckpt_type,
            'checkpoint_path': soup_path,
            'run_directory': soup_dir,
            'config_path': '',
        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export checkpoint metadata to CSV.')
    parser.add_argument('--run_dirs', nargs='+', required=True,
                        help='Run directories containing checkpoints and config_used.yaml.')
    parser.add_argument('--output', required=True, help='Output CSV file path.')
    parser.add_argument('--soup_ckpts', nargs='*', default=[],
                        help='Additional soup checkpoint paths to include.')
    parser.add_argument('--soup_label', default='soup',
                        help='Prefix label used for soup checkpoint entries.')
    return parser.parse_args()


def write_csv(rows: List[Dict], output_path: str) -> None:
    if not rows:
        raise ValueError('No checkpoint data collected; nothing to write.')

    fieldnames = ['dataset', 'seed', 'learning_rate', 'checkpoint_type',
                  'checkpoint_path', 'run_directory', 'config_path']

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()

    all_rows: List[Dict] = []

    for run_dir in args.run_dirs:
        if not os.path.isdir(run_dir):
            raise NotADirectoryError(f'Run directory not found: {run_dir}')
        all_rows.extend(collect_run_rows(run_dir))

    if args.soup_ckpts:
        all_rows.extend(collect_soup_rows(args.soup_ckpts, args.soup_label))

    write_csv(all_rows, args.output)


if __name__ == '__main__':
    main()

