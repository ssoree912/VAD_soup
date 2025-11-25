#!/usr/bin/env python3
"""
Script to perform Fisher Soup merging for VAD models.
"""

import argparse
import logging
import os
import sys
import yaml
from typing import List, Optional

import torch
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import AD_Model
from data.dataset_loader import CreateDataset
from fisher_soup_vad import FisherSoupVAD, load_models_and_fishers
from utils import set_seeds


def parse_args():
    parser = argparse.ArgumentParser(description='Fisher Soup merging for VAD models.')
    parser.add_argument('--config', required=True, help='Configuration YAML file.')
    parser.add_argument('--checkpoints', nargs='+', required=True, 
                       help='Model checkpoint paths to merge.')
    parser.add_argument('--fisher_paths', nargs='*', default=None,
                       help='Fisher information file paths (optional, must match checkpoints order).')
    parser.add_argument('--output', required=True, help='Output path to save the merged model.')
    parser.add_argument('--device', default=None, help='Device to use (cpu or cuda).')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU index when using CUDA.')
    
    # Merging strategy options
    parser.add_argument('--strategy', choices=['grid', 'random', 'uniform'], default='grid',
                       help='Coefficient search strategy.')
    parser.add_argument('--n_combinations', type=int, default=10,
                       help='Number of coefficient combinations to try.')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for coefficient generation.')
    
    # Fisher merging options
    parser.add_argument('--fisher_floor', type=float, default=1e-6,
                       help='Minimum Fisher value floor.')
    parser.add_argument('--no_favor_target', action='store_true',
                       help='Do not favor target model (apply fisher_floor to all models).')
    parser.add_argument('--no_normalize_fishers', action='store_true',
                       help='Do not normalize Fisher information.')
    
    # Evaluation options
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate the merged model.')
    
    return parser.parse_args()


def setup_logger():
    logger = logging.getLogger('fisher_soup_merge')
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


def generate_coefficients(strategy: str, n_models: int, n_combinations: int, 
                         random_seed: int, fisher_soup: FisherSoupVAD) -> List[List[float]]:
    """Generate coefficient combinations based on strategy."""
    if strategy == 'uniform':
        # Simple uniform averaging
        uniform_coeff = [1.0 / n_models] * n_models
        return [uniform_coeff]
    
    elif strategy == 'grid':
        if n_models == 2:
            return fisher_soup.create_pairwise_grid_coeffs(n_combinations)
        else:
            # For more than 2 models, use random sampling
            return fisher_soup.create_random_coeffs(n_models, n_combinations, random_seed)
    
    elif strategy == 'random':
        return fisher_soup.create_random_coeffs(n_models, n_combinations, random_seed)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


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
    set_seeds(args.random_seed)
    
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
    
    # Validate inputs
    if args.fisher_paths is not None and len(args.fisher_paths) != len(args.checkpoints):
        raise ValueError('Number of Fisher files must match number of checkpoints')
    
    # Load models and Fisher information
    logger.info(f'Loading {len(args.checkpoints)} models and Fisher information...')
    models, fishers, masks = load_models_and_fishers(
        checkpoint_paths=args.checkpoints,
        fisher_paths=args.fisher_paths,
        feature_dim=cfg.feature_dim,
        dropout_rate=cfg.dropout_rate,
        device=device,
        logger=logger
    )
    
    # Initialize Fisher Soup
    fisher_soup = FisherSoupVAD(device, logger)
    combined_mask = fisher_soup.combine_masks(masks)
    for model, mask in zip(models, masks):
        fisher_soup.apply_mask_to_model(model, mask)
    mask_stats = None
    if combined_mask is not None:
        total_mask_elems = sum(tensor.numel() for tensor in combined_mask.values())
        nonzero_mask = sum(int(tensor.sum().item()) for tensor in combined_mask.values())
        sparsity_ratio = 1.0 - (nonzero_mask / total_mask_elems if total_mask_elems else 0.0)
        logger.info(f'Combined pruning mask retains {nonzero_mask}/{total_mask_elems} weights '
                    f'({1 - sparsity_ratio:.2%} density).')
        mask_stats = {
            'retained_weights': nonzero_mask,
            'total_weights': total_mask_elems,
            'density': 1 - sparsity_ratio,
            'sparsity': sparsity_ratio,
        }
    
    # Generate coefficient combinations
    n_models = len(models)
    coefficients_set = generate_coefficients(
        strategy=args.strategy,
        n_models=n_models,
        n_combinations=args.n_combinations,
        random_seed=args.random_seed,
        fisher_soup=fisher_soup
    )
    
    logger.info(f'Generated {len(coefficients_set)} coefficient combinations using {args.strategy} strategy')
    
    # Prepare evaluation if requested
    test_loader = None
    if args.evaluate:
        logger.info('Preparing test dataset for evaluation...')
        test_loader, _, _, _ = CreateDataset(cfg, logger)
    
    # Search for optimal coefficients
    if test_loader is not None:
        results = fisher_soup.search_merging_coefficients(
            models=models,
            coefficients_set=coefficients_set,
            test_loader=test_loader,
            feature_dim=cfg.feature_dim,
            dropout_rate=cfg.dropout_rate,
            segment_len=cfg.segment_len,
            fishers=fishers,
            fisher_floor=args.fisher_floor,
            favor_target_model=not args.no_favor_target,
            normalize_fishers=not args.no_normalize_fishers,
            combined_mask=combined_mask,
            print_results=True
        )
        
        # Use best coefficients
        best_result = max(results, key=lambda x: x.score['roc_auc'])
        best_coefficients = best_result.coefficients
        logger.info(f'Using best coefficients: {best_coefficients}')
        
    else:
        # Use the first coefficient combination (or uniform if only one)
        best_coefficients = coefficients_set[0]
        logger.info(f'Using coefficients without evaluation: {best_coefficients}')
    
    # Create final merged model
    logger.info('Creating final merged model...')
    merged_models = fisher_soup.generate_merged_for_coeffs_set(
        models=models,
        coefficients_set=[best_coefficients],
        feature_dim=cfg.feature_dim,
        dropout_rate=cfg.dropout_rate,
        fishers=fishers,
        fisher_floor=args.fisher_floor,
        favor_target_model=not args.no_favor_target,
        normalize_fishers=not args.no_normalize_fishers,
        combined_mask=combined_mask
    )
    
    # Get the merged model
    _, final_model = next(merged_models)
    fisher_soup.apply_mask_to_model(final_model, combined_mask)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save the merged model
    final_state = final_model.state_dict()
    torch.save(final_state, args.output)
    logger.info(f'Saved Fisher Soup merged model to {args.output}')
    
    if combined_mask is not None:
        mask_output_path = args.output + '.mask'
        mask_cpu = {key: tensor.to('cpu') for key, tensor in combined_mask.items()}
        torch.save(mask_cpu, mask_output_path)
        logger.info(f'Saved combined pruning mask to {mask_output_path}')
    
    # Save metadata
    metadata = {
        'checkpoints': args.checkpoints,
        'fisher_paths': args.fisher_paths,
        'coefficients': best_coefficients,
        'strategy': args.strategy,
        'n_combinations': args.n_combinations,
        'fisher_floor': args.fisher_floor,
        'favor_target_model': not args.no_favor_target,
        'normalize_fishers': not args.no_normalize_fishers,
        'random_seed': args.random_seed,
        'mask_applied': combined_mask is not None,
    }
    metadata['mask_paths'] = [
        path + '.mask' if os.path.exists(path + '.mask') else None
        for path in args.checkpoints
    ]
    if mask_stats is not None:
        metadata['mask_stats'] = mask_stats
    
    if args.evaluate and test_loader is not None:
        metadata['evaluation_score'] = best_result.score
    
    metadata_path = os.path.splitext(args.output)[0] + '_metadata.yaml'
    with open(metadata_path, 'w') as f:
        yaml.safe_dump(metadata, f, sort_keys=False)
    logger.info(f'Saved metadata to {metadata_path}')
    
    # Final evaluation if requested
    if args.evaluate and test_loader is not None:
        logger.info('Performing final evaluation of merged model...')
        final_score = fisher_soup.evaluate_model(final_model, test_loader, cfg.segment_len)
        logger.info(f'Final merged model performance:')
        logger.info(f'  PR AUC: {final_score["pr_auc"]:.4f}')
        logger.info(f'  ROC AUC: {final_score["roc_auc"]:.4f}')
    
    logger.info('Fisher Soup merging completed successfully!')


if __name__ == '__main__':
    main()
