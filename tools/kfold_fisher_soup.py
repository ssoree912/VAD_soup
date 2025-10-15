#!/usr/bin/env python3
"""
K-Fold Fisher Soup for VAD models.
Computes Fisher information for all folds and performs Fisher Soup merging for all combinations.
"""

import os
import sys
import argparse
import logging
import torch
import yaml
from itertools import combinations
from pathlib import Path

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fisher_vad import FisherVAD
from fisher_soup_vad import FisherSoupVAD, load_models_and_fishers
from utils import set_seeds
from data.dataset_loader import CreateDataset
from model import AD_Model


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def find_fold_checkpoints(base_dir: str) -> dict:
    """Find all fold checkpoint files."""
    fold_ckpts = {}
    base_path = Path(base_dir)
    
    for fold_dir in sorted(base_path.glob("fold_*")):
        fold_num = fold_dir.name.split("_")[1]
        best_auc_files = list(fold_dir.glob("**/best_auc.pkl"))
        if best_auc_files:
            fold_ckpts[fold_num] = str(best_auc_files[0])
    
    return fold_ckpts


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_dataset(config: dict, logger: logging.Logger):
    """Prepare dataset loaders."""
    # Convert dict to object with attributes for dataset loader compatibility
    class Args:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    args = Args(config)
    test_loader, train_loader, train_eval_loader, _ = CreateDataset(args, logger)
    segment_len = config['segment_len']
    return train_loader, test_loader, segment_len


def compute_fisher_for_fold(ckpt_path: str, config_path: str, output_dir: str, logger: logging.Logger):
    """Compute Fisher information for a single fold."""
    fold_name = Path(ckpt_path).parents[3].name  # fold_XX
    fisher_output_path = os.path.join(output_dir, f"fisher_{fold_name}.pt")
    
    if os.path.exists(fisher_output_path):
        logger.info(f"Fisher info already exists for {fold_name}: {fisher_output_path}")
        return fisher_output_path
    
    logger.info(f"Computing Fisher information for {fold_name}")
    
    # Load config and set up device
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model first
    model = AD_Model(config['feature_dim'], 512, config['dropout_rate'])
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    
    # Create Fisher VAD instance
    fisher_vad = FisherVAD(model=model, device=device, logger=logger)
    
    # Prepare dataset (only need training loader for Fisher)
    train_loader, _, _ = prepare_dataset(config, logger)
    
    # Compute Fisher information
    fisher_info = fisher_vad.compute_fisher_for_model(train_loader)
    
    # Save Fisher information (list of tensors)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(fisher_info, fisher_output_path)
    logger.info(f"Saved Fisher info to: {fisher_output_path}")
    
    return fisher_output_path


def run_fisher_soup_combination(ckpt_paths: list, fisher_paths: list, combination: tuple, 
                               config_path: str, output_dir: str, logger: logging.Logger):
    """Run Fisher Soup for a specific combination of folds."""
    combo_name = "_".join([f"fold{fold}" for fold in combination])
    output_path = os.path.join(output_dir, f"fisher_soup_{combo_name}.pkl")
    
    if os.path.exists(output_path):
        logger.info(f"Fisher soup already exists: {output_path}")
        return
    
    logger.info(f"Running Fisher Soup for combination: {combo_name}")
    
    # Get checkpoints and Fisher files for this combination
    combo_ckpts = [ckpt_paths[fold] for fold in combination]
    combo_fishers = [fisher_paths[fold] for fold in combination]
    
    # Prepare config
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Create Fisher Soup VAD instance
    fisher_soup = FisherSoupVAD(device=device, logger=logger)
    
    # Load models and Fisher information
    models, fishers, masks = load_models_and_fishers(
        checkpoint_paths=combo_ckpts,
        fisher_paths=combo_fishers,
        feature_dim=config['feature_dim'],
        dropout_rate=config['dropout_rate'],
        device=device,
        logger=logger
    )
    
    # Combine masks if available
    combined_mask = fisher_soup.combine_masks(masks)
    for model, mask in zip(models, masks):
        fisher_soup.apply_mask_to_model(model, mask)
    
    # Generate coefficient combinations (using grid strategy)
    n_models = len(models)
    if n_models == 2:
        coefficients_set = fisher_soup.create_pairwise_grid_coeffs(10)
    else:
        coefficients_set = fisher_soup.create_random_coeffs(n_models, 10, seed=42)
    
    # Prepare test dataset for evaluation
    train_loader, test_loader, segment_len = prepare_dataset(config, logger)
    
    # Search for optimal merging coefficients
    results = fisher_soup.search_merging_coefficients(
        models=models,
        coefficients_set=coefficients_set,
        test_loader=test_loader,
        feature_dim=config['feature_dim'],
        dropout_rate=config['dropout_rate'],
        segment_len=segment_len,
        fishers=fishers,
        combined_mask=combined_mask,
        print_results=True
    )
    
    # Find best result and save the merged model
    best_result = max(results, key=lambda x: x.score['roc_auc'])
    logger.info(f"Best combination {combo_name} - Coefficients: {best_result.coefficients}, ROC AUC: {best_result.score['roc_auc']:.4f}")
    
    # Create final merged model with best coefficients
    best_model = fisher_soup.clone_model(models[0], config['feature_dim'], config['dropout_rate'])
    fisher_soup._merge_with_coeffs(
        output_model=best_model,
        models_to_merge=models,
        coefficients=best_result.coefficients,
        fishers=fishers
    )
    fisher_soup.apply_mask_to_model(best_model, combined_mask)
    
    # Save the merged model
    os.makedirs(output_dir, exist_ok=True)
    torch.save(best_model.state_dict(), output_path)
    logger.info(f"Saved Fisher soup model to: {output_path}")
    
    # Save results metadata
    results_path = output_path.replace('.pkl', '_results.pt')
    torch.save({
        'best_coefficients': best_result.coefficients,
        'best_score': best_result.score,
        'all_results': results,
        'fold_combination': combination,
        'checkpoint_paths': combo_ckpts,
        'fisher_paths': combo_fishers
    }, results_path)


def main():
    parser = argparse.ArgumentParser(description='K-Fold Fisher Soup for VAD models')
    parser.add_argument('--fold_base_dir', required=True, 
                       help='Base directory containing fold_XX subdirectories')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    parser.add_argument('--min_folds', type=int, default=2, help='Minimum number of folds to combine')
    parser.add_argument('--max_folds', type=int, default=5, help='Maximum number of folds to combine')
    parser.add_argument('--compute_fisher_only', action='store_true', 
                       help='Only compute Fisher information, skip soup merging')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting K-Fold Fisher Soup process")
    
    # Find all fold checkpoints
    fold_ckpts = find_fold_checkpoints(args.fold_base_dir)
    logger.info(f"Found {len(fold_ckpts)} folds: {list(fold_ckpts.keys())}")
    
    if len(fold_ckpts) < args.min_folds:
        logger.error(f"Not enough folds found. Need at least {args.min_folds}, found {len(fold_ckpts)}")
        return
    
    # Create output directories
    fisher_output_dir = os.path.join(args.output_dir, "fisher_info")
    soup_output_dir = os.path.join(args.output_dir, "fisher_soups")
    os.makedirs(fisher_output_dir, exist_ok=True)
    os.makedirs(soup_output_dir, exist_ok=True)
    
    # Step 1: Compute Fisher information for all folds
    logger.info("Step 1: Computing Fisher information for all folds")
    fisher_paths = {}
    for fold_num, ckpt_path in fold_ckpts.items():
        fisher_path = compute_fisher_for_fold(ckpt_path, args.config, fisher_output_dir, logger)
        fisher_paths[fold_num] = fisher_path
    
    if args.compute_fisher_only:
        logger.info("Fisher computation completed. Exiting as requested.")
        return
    
    # Step 2: Generate all combinations and run Fisher Soup
    logger.info("Step 2: Running Fisher Soup for all fold combinations")
    fold_numbers = sorted(fold_ckpts.keys())
    
    total_combinations = 0
    for k in range(args.min_folds, min(args.max_folds + 1, len(fold_numbers) + 1)):
        combos = list(combinations(fold_numbers, k))
        total_combinations += len(combos)
        logger.info(f"Will process {len(combos)} combinations of {k} folds")
        
        for combination in combos:
            try:
                run_fisher_soup_combination(
                    ckpt_paths=fold_ckpts,
                    fisher_paths=fisher_paths,
                    combination=combination,
                    config_path=args.config,
                    output_dir=soup_output_dir,
                    logger=logger
                )
            except Exception as e:
                logger.error(f"Failed to process combination {combination}: {e}")
                continue
    
    logger.info(f"Completed Fisher Soup for {total_combinations} combinations")
    logger.info(f"Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
