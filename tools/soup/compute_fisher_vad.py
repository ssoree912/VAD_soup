#!/usr/bin/env python3
"""
Script to compute Fisher Information Matrix for VAD models.
"""

import argparse
import logging
import os
import sys
import yaml

import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import AD_Model
from data.dataset_loader import CreateDataset
from fisher_vad import FisherVAD, save_fisher_info
from utils import set_seeds


def parse_args():
    parser = argparse.ArgumentParser(description='Compute Fisher Information Matrix for VAD models.')
    parser.add_argument('--config', required=True, help='Configuration YAML file.')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint path.')
    parser.add_argument('--output', required=True, help='Output path to save Fisher information.')
    parser.add_argument('--device', default=None, help='Device to use (cpu or cuda).')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU index when using CUDA.')
    parser.add_argument('--max_batches', type=int, default=None, 
                       help='Maximum number of batches to process (for debugging).')
    return parser.parse_args()


def setup_logger():
    logger = logging.getLogger('compute_fisher_vad')
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def load_config(config_path: str) -> argparse.Namespace:
    with open(config_path, 'r') as handle:
        cfg = yaml.load(handle, Loader=yaml.FullLoader)
    return argparse.Namespace(**cfg)


def main():
    args = parse_args()
    logger = setup_logger()
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Override device settings if provided
    if args.device:
        cfg.device = args.device
    if args.gpu_id is not None:
        cfg.gpu_id = args.gpu_id
    
    # Set random seeds
    set_seeds(getattr(cfg, 'seed', 42))
    
    # Setup device
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
    
    logger.info(f'Using device: {device}')
    
    # Load model
    logger.info(f'Loading model from checkpoint: {args.checkpoint}')
    model = AD_Model(cfg.feature_dim, 512, cfg.dropout_rate)
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')
    
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    
    logger.info(f'Model loaded successfully. Total parameters: {sum(p.numel() for p in model.parameters())}')
    
    # Create dataset
    logger.info('Creating dataset...')
    _, train_loader, _, _ = CreateDataset(cfg, logger)
    
    # Limit batches if specified (for debugging)
    if args.max_batches is not None:
        logger.info(f'Limiting to {args.max_batches} batches for debugging')
        limited_data = []
        for i, batch in enumerate(train_loader):
            if i >= args.max_batches:
                break
            limited_data.append(batch)
        train_loader = limited_data
    
    # Initialize Fisher computation
    fisher_vad = FisherVAD(model, device, logger)
    
    # Compute Fisher Information
    logger.info('Starting Fisher Information computation...')
    fishers = fisher_vad.compute_fisher_for_model(train_loader)
    
    # Get variable information for saving
    variables = fisher_vad.get_mergeable_variables()
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save Fisher information
    save_fisher_info(fishers, variables, args.output, logger)
    
    # Print statistics
    total_elements = sum(fisher.numel() for fisher in fishers)
    logger.info(f'Fisher computation completed. Total elements: {total_elements}')
    
    # Compute some basic statistics
    all_values = torch.cat([fisher.flatten() for fisher in fishers])
    logger.info(f'Fisher statistics:')
    logger.info(f'  Mean: {all_values.mean().item():.6f}')
    logger.info(f'  Std: {all_values.std().item():.6f}')
    logger.info(f'  Min: {all_values.min().item():.6f}')
    logger.info(f'  Max: {all_values.max().item():.6f}')
    
    logger.info('Fisher computation completed successfully!')


if __name__ == '__main__':
    main()