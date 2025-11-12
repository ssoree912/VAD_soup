import torch

import os
import shutil
import sys
import json
import numpy as np
import yaml
import argparse
from typing import Dict, List, Optional

from utils import *
from model import AD_Model, Memory_module
from loss import Loss_bce
from data.dataset_loader import CreateDataset
from lanp.train import blend_scores, update_reweight
from post.score_fusion import fuse_scores


def _maybe_unpack_object_array(array):
    if isinstance(array, np.ndarray) and array.dtype == object:
        if array.shape == ():
            return array.item()
        if array.size == 1:
            return array.flat[0]
    return None


def load_score_dictionary(path: str) -> Dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    data: Dict[str, np.ndarray]
    if isinstance(payload, np.lib.npyio.NpzFile):
        data_dict = None
        preferred = ["snippet_scores", "scores", "data", "arr_0", "frame_scores"]
        for key in preferred:
            if key in payload.files:
                candidate = _maybe_unpack_object_array(payload[key])
                if isinstance(candidate, dict):
                    data_dict = candidate
                    break
        if data_dict is None:
            for key in payload.files:
                candidate = _maybe_unpack_object_array(payload[key])
                if isinstance(candidate, dict):
                    data_dict = candidate
                    break
        if data_dict is None:
            data = {k: payload[k] for k in payload.files}
        else:
            data = data_dict
    elif isinstance(payload, np.ndarray) and payload.dtype == object:
        data = payload.item()
    elif isinstance(payload, dict):
        data = payload
    else:
        raise ValueError(f"Unsupported ROI score container in {path}")

    score_map: Dict[str, np.ndarray] = {}
    for video, values in data.items():
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        score_map[video] = arr
    return score_map


def save_dict_payload(path: Optional[str], data: Dict[str, np.ndarray]):
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if path.endswith(".npz"):
        np.savez_compressed(path, data=np.array([data], dtype=object))
    else:
        np.save(path, data, allow_pickle=True)


def snippet_to_frame_scores(
    snippet_scores: Dict[str, np.ndarray],
    labels_dist: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    frame_map: Dict[str, np.ndarray] = {}
    for video, scores in snippet_scores.items():
        label_chunks = labels_dist.get(video)
        if label_chunks is None:
            continue
        frames: List[float] = []
        padded_scores = np.asarray(scores).reshape(-1)
        for idx, chunk in enumerate(label_chunks):
            if idx >= len(padded_scores):
                break
            repeat = len(chunk)
            frames.extend([float(padded_scores[idx])] * repeat)
        frame_map[video] = np.asarray(frames, dtype=np.float32)
    return frame_map


def export_eval_artifacts(
    snippet_scores: Dict[str, np.ndarray],
    labels_dist: Dict[str, np.ndarray],
    frame_path: Optional[str],
    snippet_path: Optional[str],
):
    if snippet_path:
        save_dict_payload(snippet_path, snippet_scores)
    if frame_path:
        frame_scores = snippet_to_frame_scores(snippet_scores, labels_dist)
        save_dict_payload(frame_path, frame_scores)


def append_metrics_json(path: Optional[str], record: Dict[str, object]):
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    existing: List[Dict[str, object]] = []
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                existing = json.load(fh)
                if not isinstance(existing, list):
                    existing = []
        except Exception:
            existing = []
    existing.append(record)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2)

def save_state_and_mask(model, path, pruning_handler=None):
    torch.save(model.state_dict(), path)
    mask_path = path + '.mask'
    if pruning_handler is not None:
        mask_dict = pruning_handler.export_masks()
        if mask_dict:
            torch.save(mask_dict, mask_path)
        elif os.path.exists(mask_path):
            os.remove(mask_path)
    else:
        if os.path.exists(mask_path):
            os.remove(mask_path)

def train(dataloader, model, optimizer_model, criterion, epoch, device, pruning_handler=None):
    with torch.set_grad_enabled(True):
        model.train()

        total_loss = 0.0
        total_samples = 0
        total_batches = 0

        for features, pseudo_labels, reweight, _, _ in dataloader:
            bs, nc, t, dim = features.shape
            features = features.type(torch.float).to(device)
            pseudo_labels = pseudo_labels.type(torch.float).to(device)
            reweight = reweight.type(torch.float).to(device)
            scores = model(features)
            loss_cls = criterion(scores, pseudo_labels, reweight)

            optimizer_model.zero_grad()
            loss_cls.backward()

            if pruning_handler is not None:
                pruning_handler.apply_gradients()

            optimizer_model.step()

            if pruning_handler is not None:
                pruning_handler.apply_weights()

            total_loss += loss_cls.item()
            total_samples += bs * nc * t
            total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)

        logger.info('Epoch: [{:.0f}/{:.0f}], '
                    'sample_num: {}, '
                    'loss_cls: {:.4f}.'.format(epoch, args.epochs, total_samples, avg_loss))

        return avg_loss

def test(model, test_loader, device, is_train_sample=False, roi_scores: Optional[Dict[str, np.ndarray]] = None,
         fusion_method: str = "none", fusion_alpha: float = 0.5):
    if is_train_sample:
        with torch.no_grad():
            model.eval()
            losses_dict={}
            for features, label_frames, video_name in test_loader:
                features = features.type(torch.float).to(device)
                label_frames = label_frames.type(torch.float).to(device)
                outputs = model(features)
                scores = outputs.squeeze().cpu().numpy()
        
                losses_dict[video_name[0]] = scores

        logger.info('Eval abnormal videos in training set. Finished!')

        return losses_dict
    else:
        with torch.no_grad():
            model.eval()
            scores_dist = {}
            labels_dist = {}
            for features, label_frames, video_name in test_loader:
                features = features.type(torch.float).to(device)
                label_frames = label_frames.type(torch.float).to(device)
                outputs = model(features)

                scores = outputs.squeeze().cpu().numpy()
                video_key = video_name[0]
                scores_dist[video_key] = scores
                labels_dist[video_key] = label_frames[0].detach().cpu().numpy().astype(int)

        if roi_scores is not None and fusion_method.lower() != "none":
            scores_dist = fuse_scores(scores_dist, roi_scores, fusion_method, fusion_alpha)

        total_scores = []
        total_labels = []
        for video, snippet_scores in scores_dist.items():
            label_chunks = labels_dist.get(video)
            if label_chunks is None:
                continue
            for score, label in zip(snippet_scores, label_chunks):
                total_scores.extend([score] * len(label))
                total_labels.extend(label.astype(int).tolist())

        total_score_frames = np.array(total_scores)
        total_label_frames = np.array(total_labels)

        prauc_frames, rocauc_frames = calc_metrics(total_score_frames, total_label_frames)
    
        logger.info('Testing: pr@ {:.2f}%, '
              'auc@ {:.2f}% \t'.format(prauc_frames, rocauc_frames))
        
        return scores_dist, prauc_frames, rocauc_frames, labels_dist

def save_run_config(args, ckpt_path, logger):
    config_dest = os.path.join(ckpt_path, 'config_used.yaml')
    config_src = getattr(args, 'config_file', None)

    if config_src and os.path.isfile(config_src):
        try:
            shutil.copy2(config_src, config_dest)
            logger.info('Copied config file to {}'.format(config_dest))
            return
        except (OSError, IOError) as err:
            logger.warning('Failed to copy config file ({}). Will dump runtime args instead.'.format(err))

    try:
        with open(config_dest, 'w') as handle:
            yaml.safe_dump({k: v for k, v in vars(args).items() if k != 'config_file'}, handle, sort_keys=True)
        logger.info('Saved run configuration to {}'.format(config_dest))
    except (OSError, IOError) as err:
        logger.warning('Failed to save run configuration: {}'.format(err))


def prepare_log_files(args):
    seed_tag = getattr(args, 'seed', None)
    if getattr(args, 'use_pruning', False):
        pruning_suffix = 'prunedEarly_' if getattr(args, 'use_early_unprune', False) else 'pruned_'
    else:
        pruning_suffix = ''
    if seed_tag is None:
        param_str = '{}_{}lr_{}_{}'.format(args.dataset, pruning_suffix, args.lr, get_timestamp())
    else:
        param_str = '{}_{}seed{}_lr_{}_{}'.format(args.dataset, pruning_suffix, seed_tag, args.lr, get_timestamp())
    param_str = param_str.replace('__', '_')

    ckpt_path = os.path.join(args.ckpt_path, args.dataset, param_str)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    logger_path = os.path.join(args.logger_path, args.dataset)
    if not os.path.exists(logger_path):
        os.makedirs(logger_path)

    logger = get_logger(logger_path+'/{}.txt'.format(param_str))
    logger.info('Train this model at time {}'.format(get_timestamp()))
    log_param(logger, args)
    logger.info(param_str)

    save_run_config(args, ckpt_path, logger)

    return logger, ckpt_path

def prepare_model(args, device):
    cls_model = AD_Model(args.feature_dim, 512, args.dropout_rate)

    if torch.cuda.is_available():
        cls_model.to(device)
        torch.backends.cudnn.benchmark = True

    return cls_model

def compute_weight_statistics(model):
    with torch.no_grad():
        l1_sum = 0.0
        l2_sum = 0.0
        max_abs = 0.0
        for param in model.parameters():
            data = param.detach().float()
            l1_sum += torch.sum(torch.abs(data)).item()
            l2_sum += torch.sum(data * data).item()
            max_abs = max(max_abs, torch.max(torch.abs(data)).item())

    return l1_sum, l2_sum ** 0.5, max_abs


class PruningHandler:
    def __init__(self, model, magnitude_ratio, random_ratio, logger=None, random_seed=None,
                 strategy: str = 'standard', dwa_alpha: float = 0.0, dwa_beta: float = 1.0,
                 dwa_acceleration: str = 'delta'):
        self.model = model
        self.magnitude_ratio = max(0.0, float(magnitude_ratio)) if magnitude_ratio is not None else 0.0
        self.random_ratio = max(0.0, float(random_ratio)) if random_ratio is not None else 0.0
        self.logger = logger
        self.entries = []
        self.total_params = 0
        self.pruned_params = 0
        self.random_seed = random_seed
        self.active = False
        self.released = False
        self.release_epoch = None
        self.stored_mask = {}
        self.strategy = (strategy or 'standard').lower()
        self.dwa_alpha = float(dwa_alpha)
        self.dwa_beta = float(dwa_beta)
        raw_mode = dwa_acceleration or 'delta'
        self.dwa_mode = raw_mode.lower()
        if self.dwa_mode not in ('delta', 'simple', 'none'):
            if self.logger:
                self.logger.warning('Unknown DWA acceleration mode "%s"; falling back to "delta".', raw_mode)
            self.dwa_mode = 'delta'
        self.magnitude_threshold = 0.0
        self._create_masks()

    def _gather_candidate_params(self):
        candidates = []
        for name, param in self.model.named_parameters():
            if (not param.requires_grad) or param.dim() <= 1:
                continue
            candidates.append((name, param))
        return candidates

    def _create_masks(self):
        candidates = self._gather_candidate_params()
        if not candidates:
            if self.logger:
                self.logger.warning('Pruning requested but no valid parameters were found.')
            return

        if self.magnitude_ratio <= 0.0 and self.random_ratio <= 0.0:
            if self.logger:
                self.logger.info('Pruning enabled but magnitude and random ratios are zero; skipping pruning.')
            return

        flat_weights = []
        param_slices = []
        cursor = 0
        for name, param in candidates:
            weight_flat = param.detach().float().abs().view(-1).cpu()
            flat_weights.append(weight_flat)
            numel = weight_flat.numel()
            param_slices.append((name, param, cursor, cursor + numel))
            cursor += numel

        all_abs = torch.cat(flat_weights, dim=0)
        total = all_abs.numel()
        self.total_params = total
        if total == 0:
            return

        global_mask = torch.ones(total, dtype=torch.bool)

        # Magnitude pruning
        k_mag = int(total * self.magnitude_ratio)
        if k_mag > 0:
            _, idx = torch.topk(all_abs, k_mag, largest=False)
            global_mask[idx] = False
            self.magnitude_threshold = float(all_abs[idx].max().item()) if idx.numel() > 0 else 0.0
        else:
            self.magnitude_threshold = 0.0

        # Random pruning on remaining weights
        remaining = torch.nonzero(global_mask, as_tuple=False).view(-1)
        k_rand = int(total * self.random_ratio)
        if k_rand > 0 and remaining.numel() > 0:
            k_rand = min(k_rand, remaining.numel())
            if self.random_seed is not None:
                generator = torch.Generator(device=remaining.device)
                generator.manual_seed(int(self.random_seed))
                perm = torch.randperm(remaining.numel(), generator=generator)[:k_rand]
            else:
                perm = torch.randperm(remaining.numel())[:k_rand]
            rand_idx = remaining[perm]
            global_mask[rand_idx] = False

        self.pruned_params = total - int(global_mask.sum().item())

        for name, param, start, end in param_slices:
            mask_flat = global_mask[start:end].to(param.device, dtype=param.dtype)
            mask_tensor = mask_flat.view_as(param).clone()
            mask_tensor.requires_grad = False
            original_values = param.data.detach().clone()
            original_pruned = original_values * (1 - mask_tensor)
            param.data.mul_(mask_tensor)
            self.entries.append({'name': name,
                                 'param': param,
                                 'mask': mask_tensor,
                                 'original_pruned': original_pruned})
            self.stored_mask[name] = mask_tensor.detach().cpu().clone().to(torch.float32)

        if self.logger:
            self.logger.info('Applied pruning: magnitude {:.2f}%, random {:.2f}%, pruned {}/{} params ({:.2f}%).'.
                             format(self.magnitude_ratio*100, self.random_ratio*100,
                                    self.pruned_params, self.total_params,
                                    (self.pruned_params / self.total_params) * 100 if self.total_params else 0.0))

        self.active = len(self.entries) > 0 and (self.magnitude_ratio > 0 or self.random_ratio > 0)

    def apply_gradients(self):
        if not self.active or self.released:
            return
        if self.strategy != 'dwa_kill_and_reactivate':
            for entry in self.entries:
                grad = entry['param'].grad
                if grad is not None:
                    grad.data.mul_(entry['mask'])
            return

        beta = self.dwa_beta
        alpha = self.dwa_alpha
        mode = 'delta' if self.dwa_mode == 'delta' else 'simple'
        for entry in self.entries:
            grad = entry['param'].grad
            if grad is None:
                continue
            param = entry['param']
            mask = entry['mask']
            tau = param.new_tensor(self.magnitude_threshold)
            abs_w = param.data.abs()
            diff = (abs_w - tau).abs()
            grad_alive = grad * mask * abs_w
            grad_dead = grad * (1 - mask) * diff

            if mode == 'delta':
                g_base = beta * (grad_alive + grad_dead)
                if alpha != 0.0:
                    sign_w = param.data.sign()
                    sign_g = grad.sign()
                    kill_cond = (abs_w > tau) & (sign_w == sign_g) & (mask > 0)
                    reactivate_cond = (abs_w <= tau) & (sign_w != sign_g)
                    delta = alpha * sign_w * kill_cond.to(param.dtype)
                    delta = delta - alpha * sign_w * reactivate_cond.to(param.dtype)
                    grad.data.copy_(g_base + delta)
                else:
                    grad.data.copy_(g_base)
            else:
                grad.data.copy_(beta * grad_alive + alpha * grad_dead)

    def apply_weights(self):
        if not self.active or self.released:
            return
        for entry in self.entries:
            entry['param'].data.mul_(entry['mask'])

    @property
    def sparsity(self):
        if self.total_params == 0:
            return 0.0
        return self.pruned_params / self.total_params

    def release(self, epoch=None):
        if not self.active or self.released:
            return

        for entry in self.entries:
            param = entry['param']
            mask = entry['mask']
            restored = entry.get('original_pruned')
            if restored is not None:
                param.data.add_(restored.to(param.device, dtype=param.dtype))
            if param.grad is not None:
                param.grad.data.mul_(mask)
        self.active = False
        self.released = True
        self.pruned_params = 0
        self.release_epoch = epoch
        self.entries = []
        if self.logger:
            msg_epoch = epoch if epoch is not None else 'N/A'
            self.logger.info('Pruning masks released at epoch {}. Model returned to full capacity.'.format(msg_epoch))

    def export_masks(self):
        if self.entries and self.active and not self.released:
            mask_dict = {}
            for entry in self.entries:
                mask_dict[entry['name']] = entry['mask'].detach().cpu().to(torch.float32)
            return mask_dict
        if self.stored_mask:
            return {k: v.clone() for k, v in self.stored_mask.items()}
        return None

    def has_active_masks(self):
        if self.entries and self.active and not self.released:
            return True
        return bool(self.stored_mask)

def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--load_config',
                               dest='config_file',
                               help='The yaml configuration file')
    config_args, remaining_argv = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument('--use_wandb', dest='use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--no_wandb', dest='use_wandb', action='store_false', help='Disable Weights & Biases logging')
    parser.set_defaults(use_wandb=False)
    parser.add_argument('--wandb_project', type=str, default=None, help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Weights & Biases entity (team) name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Weights & Biases run name')
    parser.add_argument('--wandb_group', type=str, default=None, help='Weights & Biases group name')
    parser.add_argument('--wandb_tags', nargs='*', default=None, help='Weights & Biases tags')
    parser.add_argument('--use_pruning', dest='use_pruning', action='store_true', help='Enable parameter pruning at training start')
    parser.add_argument('--no_pruning', dest='use_pruning', action='store_false', help='Disable parameter pruning')
    parser.set_defaults(use_pruning=False)
    parser.add_argument('--prune_magnitude_ratio', type=float, default=None, help='Fraction of parameters to prune by magnitude')
    parser.add_argument('--prune_random_ratio', type=float, default=None, help='Fraction of parameters to prune randomly in addition')
    parser.add_argument('--prune_random_seed', type=int, default=None, help='Seed to use for random pruning only')
    parser.add_argument('--pruning_strategy', type=str, default='standard',
                        choices=['standard', 'dwa_kill_and_reactivate'],
                        help='Pruning gradient strategy to apply when pruning is enabled')
    parser.add_argument('--dwa_alpha', type=float, default=0.0, help='Acceleration magnitude for DWA pruning strategy')
    parser.add_argument('--dwa_beta', type=float, default=1.0, help='Base scaling factor for DWA pruning strategy')
    parser.add_argument('--dwa_acceleration', type=str, default='delta',
                        choices=['delta', 'simple', 'none'],
                        help='DWA acceleration mode: delta for sign-boost, simple/none for base formula')
    parser.add_argument('--use_early_unprune', dest='use_early_unprune', action='store_true', help='Release pruning masks after a portion of training')
    parser.add_argument('--no_early_unprune', dest='use_early_unprune', action='store_false', help='Keep pruning masks active through entire training')
    parser.set_defaults(use_early_unprune=False)
    parser.add_argument('--unprune_ratio', type=float, default=None, help='Fraction of total epochs after which pruning masks are released (0~1)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default=None, help='Compute device to use (e.g., cuda or cpu)')
    parser.add_argument('--gpu_id', type=int, default=None, help='GPU index to use when device is cuda')
    parser.add_argument('--save_threshold_checkpoints', dest='save_threshold_checkpoints', action='store_true',
                        help='Save extra checkpoints whenever validation metrics pass thresholds')
    parser.add_argument('--no_save_threshold_checkpoints', dest='save_threshold_checkpoints', action='store_false',
                        help='Disable saving checkpoint files for every threshold hit (default)')
    parser.set_defaults(save_threshold_checkpoints=False)
    parser.add_argument('--roi_scores_path', type=str, default=None,
                        help='Path to npz/npy containing per-video ROI snippet scores.')
    parser.add_argument('--roi_score_fusion', type=str, default='none',
                        choices=['none', 'max', 'weighted'],
                        help='Fusion method for ROI vs frame scores during evaluation.')
    parser.add_argument('--roi_score_alpha', type=float, default=0.5,
                        help='Alpha for weighted ROI/frame fusion (ROI weight).')
    parser.add_argument('--roi_reweight_lambda', type=float, default=None,
                        help='Blend factor for ROI/global loss re-weighting (requires --roi_scores_path).')
    parser.add_argument('--save_snippet_scores_path', type=str, default=None,
                        help='Optional path to save snippet-level anomaly scores during evaluation.')
    parser.add_argument('--save_frame_scores_path', type=str, default=None,
                        help='Optional path to save frame-level anomaly scores during evaluation.')
    parser.add_argument('--save_memory_path', type=str, default=None,
                        help='Optional path to export the normal memory vectors after initialization.')
    parser.add_argument('--eval_only', action='store_true',
                        help='Skip training and only run evaluation/export once after loading data.')
    parser.add_argument('--metrics_json_path', type=str, default=None,
                        help='Optional JSON file to append ROC/PR metrics for each evaluation step.')

    if config_args.config_file:
        with open(config_args.config_file, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            if cfg is not None:
                parser.set_defaults(**cfg)

    args = parser.parse_args(remaining_argv)
    return args

if __name__ == '__main__':
    args = parse_args()

    if args.device=='cuda' and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(args.gpu_id))
    else:
        device = torch.device('cpu')

    set_seeds(args.seed)

    logger, ckpt_path = prepare_log_files(args)

    '''build model'''
    model = prepare_model(args, device)
    loss_criterion = Loss_bce()

    total_params = sum(p.numel() for p in model.parameters())
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError('Weights & Biases (wandb) is not installed. Please install wandb or disable logging.') from exc

        run_name = args.wandb_run_name if args.wandb_run_name else os.path.basename(ckpt_path)
        run_tags = args.wandb_tags if args.wandb_tags not in (None, []) else None
        wandb_run = wandb.init(project=args.wandb_project or 'lanp-uvad',
                               entity=args.wandb_entity if args.wandb_entity else None,
                               name=run_name,
                               group=args.wandb_group if args.wandb_group else None,
                               tags=run_tags)

        config_dict = {k: v for k, v in vars(args).items() if k != 'config_file'}
        config_dict['total_parameters'] = total_params
        config_dict['ckpt_path'] = ckpt_path
        wandb.config.update(config_dict, allow_val_change=True)
        wandb_run.summary['num_parameters'] = total_params
        wandb.watch(model, log='all', log_freq=100)

    pruning_handler = None
    if args.use_pruning:
        pruning_handler = PruningHandler(
            model,
            args.prune_magnitude_ratio,
            args.prune_random_ratio,
            logger,
            random_seed=getattr(args, 'prune_random_seed', None),
            strategy=getattr(args, 'pruning_strategy', 'standard'),
            dwa_alpha=getattr(args, 'dwa_alpha', 0.0),
            dwa_beta=getattr(args, 'dwa_beta', 1.0),
            dwa_acceleration=getattr(args, 'dwa_acceleration', 'delta'),
        )
        if args.use_wandb and wandb_run is not None and pruning_handler.total_params:
            wandb_run.summary['pruning/total_params'] = pruning_handler.total_params
            wandb_run.summary['pruning/pruned_params'] = pruning_handler.pruned_params
            wandb_run.summary['pruning/sparsity'] = pruning_handler.sparsity

    unprune_epoch = None
    if (pruning_handler is not None and pruning_handler.active and getattr(args, 'use_early_unprune', False)):
        ratio = args.unprune_ratio if args.unprune_ratio is not None else 0.3
        ratio = max(0.0, min(1.0, float(ratio)))
        unprune_epoch = max(1, int(np.ceil(args.epochs * ratio)))
        if unprune_epoch >= args.epochs:
            unprune_epoch = args.epochs
        logger.info('Early unprune scheduled after epoch {} (ratio {:.2f}).'.format(unprune_epoch, ratio))

    test_loader, train_loader, train_eval_loader, _ = CreateDataset(args, logger)
    roi_snippet_scores = None
    if args.roi_scores_path:
        roi_snippet_scores = load_score_dictionary(args.roi_scores_path)
        logger.info('Loaded ROI snippet scores from {} ({} videos).'.format(
            args.roi_scores_path, len(roi_snippet_scores)))
    
    '''load pretrained model'''
    if args.pretrained_path is not None:
        logger.info('load the pretrained model....')
        model.load_state_dict(torch.load(args.pretrained_ckpt))
        param_str_test = args.pretrained_ckpt.strip().split('/')[-2]
        scores_dict, _, _, _ = test(model, test_loader=test_loader, device=device)
        np.save('./test_results/{}/{}_test.npy'.format(args.dataset, param_str_test[-24:-5]), scores_dict)
        scores_dict = test(model=model, test_loader=train_eval_loader, device=device, is_train_sample=True)
        np.save('./test_results/{}/{}_train.npy'.format(args.dataset, param_str_test[-24:-5]), scores_dict)
        logger.info('load the pretrained model....finished!')

    optimizer_model = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.6)
     
    best_AUC, best_PR = 0, 0
    best_epoch_AUC, best_epoch_PR = 0, 0

    best_auc_path = os.path.join(ckpt_path, 'best_auc.pkl')
    best_pr_path = os.path.join(ckpt_path, 'best_pr.pkl')
    last_epoch_path = os.path.join(ckpt_path, 'last_epoch.pkl')
    
    memory = Memory_module(train_loader.dataset, device)
    updated_tag = memory.update_dataloader()
    logger.info(memory.logger_info)
    if args.save_memory_path:
        save_dict_payload(args.save_memory_path, {"normal_memory": memory.normal_memory.detach().cpu().numpy()})

    if args.roi_reweight_lambda is not None:
        if roi_snippet_scores is None:
            raise ValueError("--roi_reweight_lambda requires --roi_scores_path to be set.")
        dataset_train = train_loader.dataset
        global_scores = {video: info['reweight'] for video, info in dataset_train.video_info_dict.items()}
        blended_scores = blend_scores(global_scores, roi_snippet_scores, args.roi_reweight_lambda)
        update_reweight(dataset_train, blended_scores)
        logger.info('Applied ROI/global score blending to training reweights (lambda={:.2f}).'.format(
            args.roi_reweight_lambda))

    if args.eval_only:
        fusion_scores = roi_snippet_scores if (roi_snippet_scores is not None and args.roi_score_fusion != 'none') else None
        scores_dist, test_prauc, test_rocauc, labels_snapshot = test(
            model=model,
            test_loader=test_loader,
            device=device,
            roi_scores=fusion_scores,
            fusion_method=args.roi_score_fusion,
            fusion_alpha=args.roi_score_alpha,
        )
        logger.info('Eval-only: pr@ {:.2f}%, auc@ {:.2f}%'.format(test_prauc, test_rocauc))
        if args.save_frame_scores_path or args.save_snippet_scores_path:
            export_eval_artifacts(
                scores_dist,
                labels_snapshot,
                args.save_frame_scores_path,
                args.save_snippet_scores_path,
            )
        append_metrics_json(
            args.metrics_json_path,
            {
                "stage": "eval_only",
                "epoch": 0,
                "pr_auc": float(test_prauc),
                "roc_auc": float(test_rocauc),
            },
        )
        if args.use_wandb and wandb_run is not None:
            wandb.log({'eval_only/pr_auc': test_prauc, 'eval_only/roc_auc': test_rocauc}, step=0)
        sys.exit(0)
    
    if args.use_wandb and wandb_run is not None:
        initial_l1, initial_l2, initial_abs_max = compute_weight_statistics(model)
        log_payload = {'epoch': 0,
                       'model/weight_l1_norm': initial_l1,
                       'model/weight_l2_norm': initial_l2,
                       'model/weight_abs_max': initial_abs_max}
        if pruning_handler is not None:
            log_payload['pruning/sparsity'] = pruning_handler.sparsity
        wandb.log(log_payload, step=0)

    for epoch in range(1, args.epochs + 1):

        if (pruning_handler is not None and pruning_handler.active and getattr(args, 'use_early_unprune', False)
                and unprune_epoch is not None and epoch > unprune_epoch):
            pruning_handler.release(epoch)
            if args.use_wandb and wandb_run is not None:
                wandb.log({'epoch': epoch, 'pruning/released': 1}, step=epoch)
                wandb_run.summary['pruning/release_epoch'] = epoch
            logger.info('Early unprune applied before epoch {}.'.format(epoch))

        train_loss = train(train_loader, model, optimizer_model, loss_criterion, epoch, device, pruning_handler=pruning_handler)

        if args.use_wandb and wandb_run is not None:
            weight_l1, weight_l2, weight_abs_max = compute_weight_statistics(model)
            log_payload = {'epoch': epoch,
                           'train/loss': train_loss,
                           'model/weight_l1_norm': weight_l1,
                           'model/weight_l2_norm': weight_l2,
                           'model/weight_abs_max': weight_abs_max}
            if pruning_handler is not None:
                log_payload['pruning/sparsity'] = pruning_handler.sparsity
            wandb.log(log_payload, step=epoch)
            
        if epoch % args.test_freq == 0:
            fusion_scores = roi_snippet_scores if (roi_snippet_scores is not None and args.roi_score_fusion != 'none') else None
            scores_dist, test_prauc, test_rocauc, labels_snapshot = test(
                model=model,
                test_loader=test_loader,
                device=device,
                roi_scores=fusion_scores,
                fusion_method=args.roi_score_fusion,
                fusion_alpha=args.roi_score_alpha,
            )
            if args.save_frame_scores_path or args.save_snippet_scores_path:
                export_eval_artifacts(
                    scores_dist,
                    labels_snapshot,
                    args.save_frame_scores_path,
                    args.save_snippet_scores_path,
                )
            append_metrics_json(
                args.metrics_json_path,
                {
                    "stage": "val",
                    "epoch": epoch,
                    "pr_auc": float(test_prauc),
                    "roc_auc": float(test_rocauc),
                },
            )
            if args.use_wandb and wandb_run is not None:
                val_payload = {'epoch': epoch,
                               'val/pr_auc': test_prauc,
                               'val/roc_auc': test_rocauc}
                if pruning_handler is not None:
                    val_payload['pruning/sparsity'] = pruning_handler.sparsity
                wandb.log(val_payload, step=epoch)
            
            if test_rocauc > best_AUC:
                best_AUC = test_rocauc
                best_epoch_AUC = epoch
                save_state_and_mask(model, best_auc_path, pruning_handler)
                logger.info('Saved new best AUC checkpoint to {}'.format(best_auc_path))
                if args.use_wandb and wandb_run is not None:
                    wandb_run.summary['best_val_roc_auc'] = best_AUC
            if args.save_threshold_checkpoints and test_rocauc > args.th_auc * 100:
                threshold_path = os.path.join(ckpt_path, 'epoch_{}_test_auc_{:.2f}_pr_{:.2f}.pkl'.
                                              format(epoch, test_rocauc, test_prauc))
                save_state_and_mask(model, threshold_path, pruning_handler)
            if test_prauc > best_PR:
                best_PR = test_prauc
                best_epoch_PR = epoch
                save_state_and_mask(model, best_pr_path, pruning_handler)
                logger.info('Saved new best PR checkpoint to {}'.format(best_pr_path))
                if args.use_wandb and wandb_run is not None:
                    wandb_run.summary['best_val_pr_auc'] = best_PR
            if args.save_threshold_checkpoints and test_prauc > args.th_pr * 100:
                threshold_path = os.path.join(ckpt_path, 'epoch_{}_test_auc_{:.2f}_pr_{:.2f}.pkl'.format(epoch, test_rocauc, test_prauc))
                save_state_and_mask(model, threshold_path, pruning_handler)

            logger.info('best_AUC {:.2f} at epoch {}.\t best_PR {:.2f} at epoch {}.'.format(best_AUC, best_epoch_AUC, best_PR, best_epoch_PR))
            logger.info('============================')

    save_state_and_mask(model, last_epoch_path, pruning_handler)
    logger.info('Saved last epoch checkpoint to {}'.format(last_epoch_path))

    if pruning_handler is not None:
        if pruning_handler.released:
            logger.info('Final sparsity: {:.2f}% (masks released at epoch {}).'.format(pruning_handler.sparsity * 100, pruning_handler.release_epoch))
        else:
            logger.info('Final sparsity: {:.2f}% (pruning masks remained active).'.format(pruning_handler.sparsity * 100))

    if args.use_wandb and wandb_run is not None:
        wandb_run.summary['best_val_roc_auc'] = best_AUC
        wandb_run.summary['best_val_pr_auc'] = best_PR
        wandb_run.summary['best_auc_checkpoint'] = best_auc_path
        wandb_run.summary['best_pr_checkpoint'] = best_pr_path
        wandb_run.summary['last_epoch_checkpoint'] = last_epoch_path
        if pruning_handler is not None and pruning_handler.total_params:
            wandb_run.summary['pruning/total_params'] = pruning_handler.total_params
            wandb_run.summary['pruning/pruned_params'] = pruning_handler.pruned_params
            wandb_run.summary['pruning/sparsity'] = pruning_handler.sparsity
            if pruning_handler.release_epoch is not None:
                wandb_run.summary['pruning/release_epoch'] = pruning_handler.release_epoch
        wandb.finish()

    append_metrics_json(
        args.metrics_json_path,
        {
            "stage": "summary",
            "best_val_roc_auc": float(best_AUC),
            "best_val_pr_auc": float(best_PR),
            "best_epoch_auc": int(best_epoch_AUC),
            "best_epoch_pr": int(best_epoch_PR),
            "best_auc_checkpoint": best_auc_path,
            "best_pr_checkpoint": best_pr_path,
        },
    )
