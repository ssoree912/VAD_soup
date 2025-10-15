import argparse
import logging
import os
import sys
from collections import OrderedDict

import torch
import yaml

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import AD_Model
from data.dataset_loader import CreateDataset
from utils import set_seeds, calc_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Create a uniform model soup from LANP-UVAD checkpoints.')
    parser.add_argument('--ckpts', nargs='+', required=True, help='Checkpoint paths to be averaged.')
    parser.add_argument('--output', required=True, help='Path to save the averaged checkpoint.')
    parser.add_argument('--load_config', dest='config_file', help='Configuration yaml for evaluation (optional).')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the averaged checkpoint using the provided config.')
    parser.add_argument('--device', default=None, help='Override device for evaluation (cpu or cuda).')
    parser.add_argument('--gpu_id', type=int, default=None, help='GPU index to use when device is cuda.')
    return parser.parse_args()


def setup_logger():
    logger = logging.getLogger('model_soup')
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def load_config(config_path):
    with open(config_path, 'r') as handle:
        cfg = yaml.load(handle, Loader=yaml.FullLoader)
    return argparse.Namespace(**cfg)


def average_state_dicts(state_dicts):
    avg_state = OrderedDict()
    param_keys = state_dicts[0].keys()
    for key in param_keys:
        tensors = [sd[key] for sd in state_dicts]
        if torch.is_tensor(tensors[0]):
            if tensors[0].is_floating_point():
                stacked = torch.stack([t.to(torch.float32) for t in tensors], dim=0)
                mean_tensor = stacked.mean(dim=0)
                avg_state[key] = mean_tensor.to(tensors[0].dtype)
            else:
                stacked = torch.stack(tensors, dim=0)
                mean_tensor = stacked.float().mean(dim=0).round().to(tensors[0].dtype)
                avg_state[key] = mean_tensor
        else:
            avg_state[key] = tensors[0]
    return avg_state


def load_checkpoint(path):
    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, dict) and 'state_dict' in obj:
        state_dict = obj['state_dict']
        masks = obj.get('masks')
    else:
        state_dict = obj
        masks = None

    mask_path = path + '.mask'
    if os.path.exists(mask_path):
        masks = torch.load(mask_path, map_location='cpu')

    return state_dict, masks


def select_reference_mask(mask_list, checkpoint_paths, logger):
    selected_mask = None
    selected_path = None
    for path, mask in zip(checkpoint_paths, mask_list):
        if mask:
            selected_mask = mask
            selected_path = path
    if selected_mask is not None:
        logger.info('Using sparse mask from checkpoint: %s', selected_path)
    else:
        logger.info('No sparse masks found. Resulting soup will be dense.')
    return selected_mask


def build_model(args, state_dict, device):
    model = AD_Model(args.feature_dim, 512, args.dropout_rate)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def evaluate(model, args, device, logger):
    logger.info('Preparing datasets for evaluation...')
    test_loader, _, _, _ = CreateDataset(args, logger)

    total_scores = []
    total_labels = []

    model.eval()
    with torch.no_grad():
        for features, label_frames, video_name in test_loader:
            features = features.type(torch.float).to(device)
            label_frames = label_frames.type(torch.float)
            outputs = model(features)
            scores = outputs.squeeze().cpu().numpy()

            for score, label in zip(scores, label_frames[0]):
                total_scores.extend([score] * args.segment_len)
                total_labels.extend(label.detach().cpu().numpy().astype(int).tolist())

    prauc_frames, rocauc_frames = calc_metrics(total_scores, total_labels)
    logger.info('Soup evaluation — PR AUC: {:.2f}%, ROC AUC: {:.2f}%'.format(prauc_frames, rocauc_frames))
    return prauc_frames, rocauc_frames


def main():
    cli_args = parse_args()
    logger = setup_logger()

    if cli_args.evaluate and cli_args.config_file is None:
        raise ValueError('Evaluation requires --load_config to be provided.')

    logger.info('Loading checkpoints: %s', ', '.join(cli_args.ckpts))
    state_dicts = []
    masks = []
    for path in cli_args.ckpts:
        if not os.path.exists(path):
            raise FileNotFoundError('Checkpoint not found: {}'.format(path))
        state_dict, mask = load_checkpoint(path)
        state_dicts.append(state_dict)
        masks.append(mask)

    keys_reference = list(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        if list(sd.keys()) != keys_reference:
            raise ValueError('State dict keys do not match for checkpoint {}.'.format(cli_args.ckpts[state_dicts.index(sd)]))

    avg_state = average_state_dicts(state_dicts)

    reference_mask = select_reference_mask(masks, cli_args.ckpts, logger)
    if reference_mask is not None:
        for key, mask_tensor in reference_mask.items():
            if key in avg_state:
                avg_state[key] = avg_state[key] * mask_tensor.to(avg_state[key].dtype)

    output_dir = os.path.dirname(cli_args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(avg_state, cli_args.output)
    logger.info('Saved averaged checkpoint to %s', cli_args.output)

    mask_output_path = cli_args.output + '.mask'
    if reference_mask is not None:
        torch.save(reference_mask, mask_output_path)
        logger.info('Saved reference mask to %s', mask_output_path)
    elif os.path.exists(mask_output_path):
        os.remove(mask_output_path)
        logger.info('Removed existing mask file %s', mask_output_path)

    if not cli_args.evaluate:
        return

    args = load_config(cli_args.config_file)
    set_seeds(args.seed)

    eval_device = cli_args.device or getattr(args, 'device', None)
    eval_gpu_id = cli_args.gpu_id if cli_args.gpu_id is not None else getattr(args, 'gpu_id', 0)

    if eval_device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(eval_gpu_id))
    elif eval_device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info('Using Apple Silicon GPU (MPS)')
    elif eval_device in ['cuda', 'mps']:
        device = torch.device('cpu')
        logger.warning(f'{eval_device.upper()} requested but not available. Falling back to CPU.')
    else:
        device = torch.device('cpu')

    model = build_model(args, avg_state, device)
    val_results = evaluate(model, args, device, logger)

    results = {
        'soup_path': cli_args.output,
        'val_pr_auc': float(val_results[0]),
        'val_roc_auc': float(val_results[1]),
    }

    results_path = os.path.splitext(cli_args.output)[0] + '_results.yaml'
    try:
        import yaml
    except ImportError:
        yaml = None

    if yaml:
        with open(results_path, 'w') as handle:
            yaml.safe_dump(results, handle, sort_keys=False)
        logger.info('Saved evaluation metrics to %s', results_path)
    else:
        logger.warning('PyYAML not available; skipping YAML save of evaluation metrics.')


if __name__ == '__main__':
    main()
