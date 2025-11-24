import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Sequence
import logging
import os
import sys
from collections import OrderedDict, namedtuple

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import AD_Model
from fisher_vad import FisherVAD
from utils import calc_metrics

MergeResult = namedtuple("MergeResult", ["coefficients", "score"])


class FisherSoupVAD:
    """Fisher Soup implementation for VAD models."""
    
    def __init__(self, device: torch.device, logger: Optional[logging.Logger] = None):
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
    
    def print_merge_result(self, result: MergeResult):
        """Print merge result in a readable format."""
        self.logger.info(f"Merging coefficients: {result.coefficients}")
        self.logger.info("Scores:")
        for name, value in result.score.items():
            self.logger.info(f"  {name}: {value:.4f}")
    
    def create_pairwise_grid_coeffs(self, n_weightings: int) -> List[Tuple[float, float]]:
        """Create pairwise grid coefficients for two models."""
        n_weightings -= 2
        denom = n_weightings + 1
        weightings = [((i + 1) / denom, 1 - (i + 1) / denom) for i in range(n_weightings)]
        weightings = [(0.0, 1.0)] + weightings + [(1.0, 0.0)]
        weightings.reverse()
        return weightings
    
    def create_random_coeffs(self, n_models: int, n_weightings: int, seed: Optional[int] = None) -> List[List[float]]:
        """Create random coefficients using Dirichlet distribution."""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Sample from Dirichlet distribution
        alpha = np.ones(n_models)
        coefficients = []
        for _ in range(n_weightings):
            coeff = np.random.dirichlet(alpha)
            coefficients.append(coeff.tolist())
        
        return coefficients
    
    def _merge_with_coeffs(self, 
                          output_model: AD_Model,
                          models_to_merge: List[AD_Model],
                          coefficients: Sequence[float],
                          fishers: Optional[List[List[torch.Tensor]]] = None,
                          fisher_floor: float = 1e-6,
                          favor_target_model: bool = True,
                          normalize_fishers: bool = True):
        """Merge models using Fisher-weighted averaging."""
        n_models = len(models_to_merge)
        assert len(coefficients) == n_models
        
        # Get parameters from all models
        output_params = list(output_model.parameters())
        models_params = [list(model.parameters()) for model in models_to_merge]
        
        # Ensure all models have the same number of parameters
        param_counts = [len(output_params)] + [len(params) for params in models_params]
        assert len(set(param_counts)) == 1, "All models must have the same number of parameters"
        
        if fishers is None:
            fishers = [1.0] * n_models
        else:
            assert len(fishers) == n_models
            
        # Normalize Fisher information if requested
        if normalize_fishers and fishers is not None and not isinstance(fishers[0], float):
            norm_constants = []
            for fisher_list in fishers:
                norm_const = torch.sqrt(sum(torch.sum(f ** 2) for f in fisher_list))
                norm_constants.append(norm_const)
        else:
            norm_constants = None
        
        # Map parameter indices to Fisher entries (only tensors with dim > 1 were recorded)
        param_to_fisher_idx: List[Optional[int]] = [None] * len(output_params)
        if fishers is not None and not isinstance(fishers[0], float):
            fisher_entry_count = 0
            for idx, param in enumerate(output_params):
                if param.dim() > 1:
                    param_to_fisher_idx[idx] = fisher_entry_count
                    fisher_entry_count += 1
            for idx, fisher_list in enumerate(fishers):
                if len(fisher_list) != fisher_entry_count:
                    raise ValueError(
                        f"Fisher list length ({len(fisher_list)}) for model {idx} does not match "
                        f"number of recorded parameters ({fisher_entry_count})."
                    )
        
        # Apply normalization to coefficients if needed
        if norm_constants is not None:
            coefficients = [w / n.item() for w, n in zip(coefficients, norm_constants)]
        
        # Merge each parameter
        for param_idx, output_param in enumerate(output_params):
            lhs_terms = []
            rhs_terms = []
            
            for model_idx, (model_params, coeff) in enumerate(zip(models_params, coefficients)):
                param = model_params[param_idx]
                
                # Get Fisher information for this parameter
                fisher_diag = 1.0
                if isinstance(fishers[model_idx], float):
                    fisher_diag = fishers[model_idx]
                elif param_to_fisher_idx[param_idx] is not None:
                    fisher_diag = fishers[model_idx][param_to_fisher_idx[param_idx]]
                
                # Apply fisher floor (except for target model if favor_target_model is True)
                if not favor_target_model or model_idx == 0:
                    if isinstance(fisher_diag, torch.Tensor):
                        fisher_diag = torch.clamp(fisher_diag, min=fisher_floor)
                    else:
                        fisher_diag = max(fisher_diag, fisher_floor)
                
                # Compute weighted terms
                weight = coeff * fisher_diag
                lhs_terms.append(weight)
                rhs_terms.append(weight * param.data)
            
            # Combine terms
            if isinstance(lhs_terms[0], torch.Tensor):
                lhs_sum = torch.stack(lhs_terms).sum(dim=0)
                rhs_sum = torch.stack(rhs_terms).sum(dim=0)
            else:
                lhs_sum = sum(lhs_terms)
                rhs_sum = sum(rhs_terms)
            
            # Update output parameter - ensure broadcasting compatibility
            if isinstance(lhs_sum, torch.Tensor) and isinstance(rhs_sum, torch.Tensor):
                # Check if shapes are compatible for division
                if lhs_sum.shape != rhs_sum.shape:
                    # Handle shape mismatch by ensuring element-wise operations
                    if lhs_sum.numel() == 1:
                        # lhs_sum is scalar, expand to match rhs_sum shape
                        lhs_sum = lhs_sum.expand_as(rhs_sum)
                    elif rhs_sum.numel() == 1:
                        # rhs_sum is scalar, expand to match lhs_sum shape  
                        rhs_sum = rhs_sum.expand_as(lhs_sum)
                    else:
                        # Try broadcasting, if it fails use element-wise division with proper reshaping
                        try:
                            result = rhs_sum / lhs_sum
                        except RuntimeError as e:
                            if "doesn't match the broadcast shape" in str(e):
                                # Ensure both tensors have same shape as output parameter
                                param_shape = output_param.shape
                                if lhs_sum.shape != param_shape:
                                    lhs_sum = lhs_sum.view(param_shape)
                                if rhs_sum.shape != param_shape:
                                    rhs_sum = rhs_sum.view(param_shape)
                                result = rhs_sum / lhs_sum
                            else:
                                raise e
                        output_param.data.copy_(result)
                        continue
                        
                # Ensure shapes match for division
                try:
                    result = rhs_sum / lhs_sum
                    output_param.data.copy_(result)
                except RuntimeError as e:
                    if "doesn't match the broadcast shape" in str(e):
                        # Ensure both tensors have same shape as output parameter
                        param_shape = output_param.shape
                        if lhs_sum.shape != param_shape:
                            # Check if total elements match before reshaping
                            if lhs_sum.numel() == output_param.numel():
                                lhs_sum = lhs_sum.view(param_shape)
                            else:
                                # Use element-wise division with proper broadcasting
                                lhs_sum = lhs_sum.view(-1)[:output_param.numel()].view(param_shape)
                        if rhs_sum.shape != param_shape:
                            # Check if total elements match before reshaping
                            if rhs_sum.numel() == output_param.numel():
                                rhs_sum = rhs_sum.view(param_shape)
                            else:
                                # Use element-wise division with proper broadcasting
                                rhs_sum = rhs_sum.view(-1)[:output_param.numel()].view(param_shape)
                        result = rhs_sum / lhs_sum
                        output_param.data.copy_(result)
                    else:
                        raise e
            else:
                output_param.data.copy_(rhs_sum / lhs_sum)
    
    def clone_model(self, model: AD_Model, feature_dim: int, dropout_rate: float) -> AD_Model:
        """Create a deep copy of the model."""
        new_model = AD_Model(feature_dim, 512, dropout_rate)
        new_model.load_state_dict(model.state_dict())
        new_model.to(self.device)
        return new_model

    def combine_masks(self, masks: Sequence[Optional[Dict[str, torch.Tensor]]]) -> Optional[Dict[str, torch.Tensor]]:
        """Combine multiple pruning masks using OR logic."""
        combined: Dict[str, torch.Tensor] = {}
        has_mask = False
        for mask in masks:
            if mask is None:
                continue
            has_mask = True
            for key, tensor in mask.items():
                bool_tensor = (tensor != 0).to(torch.bool)
                if key not in combined:
                    combined[key] = bool_tensor.clone()
                else:
                    combined[key] = torch.logical_or(combined[key], bool_tensor)
        if not has_mask:
            return None
        return combined

    def apply_mask_to_model(self, model: AD_Model, mask: Optional[Dict[str, torch.Tensor]]):
        """Apply a pruning mask in-place to the model parameters."""
        if mask is None:
            return
        param_dict = dict(model.named_parameters())
        with torch.no_grad():
            for name, mask_tensor in mask.items():
                if name in param_dict:
                    param = param_dict[name]
                    mask_on_device = mask_tensor.to(param.device, dtype=param.dtype)
                    param.data.mul_(mask_on_device)
    
    def generate_merged_for_coeffs_set(self,
                                     models: List[AD_Model],
                                     coefficients_set: Sequence[Sequence[float]],
                                     feature_dim: int,
                                     dropout_rate: float,
                                     fishers: Optional[List[List[torch.Tensor]]] = None,
                                     fisher_floor: float = 1e-6,
                                     favor_target_model: bool = True,
                                     normalize_fishers: bool = True,
                                     combined_mask: Optional[Dict[str, torch.Tensor]] = None):
        """Generate merged models for a set of coefficients."""
        for coefficients in coefficients_set:
            # Start from a fresh clone for each coefficient set
            output_model = self.clone_model(models[0], feature_dim, dropout_rate)
            self._merge_with_coeffs(
                output_model=output_model,
                models_to_merge=models,
                coefficients=coefficients,
                fishers=fishers,
                fisher_floor=fisher_floor,
                favor_target_model=favor_target_model,
                normalize_fishers=normalize_fishers
            )
            self.apply_mask_to_model(output_model, combined_mask)
            yield coefficients, output_model
    
    def evaluate_model(self, model: AD_Model, test_loader, segment_len: int) -> Dict[str, float]:
        """Evaluate a single model and return metrics."""
        total_scores = []
        total_labels = []
        
        model.eval()
        with torch.no_grad():
            for features, label_frames, _ in test_loader:
                features = features.type(torch.float).to(self.device)
                label_frames = label_frames.type(torch.float)
                outputs = model(features)
                scores = outputs.squeeze().cpu().numpy()
                
                for score, label in zip(scores, label_frames[0]):
                    total_scores.extend([score] * segment_len)
                    total_labels.extend(label.detach().cpu().numpy().astype(int).tolist())
        
        prauc_frames, rocauc_frames = calc_metrics(total_scores, total_labels)
        return {
            'pr_auc': prauc_frames,
            'roc_auc': rocauc_frames
        }
    
    def search_merging_coefficients(self,
                                  models: List[AD_Model],
                                  coefficients_set: Sequence[Sequence[float]],
                                  test_loader,
                                  feature_dim: int,
                                  dropout_rate: float,
                                  segment_len: int,
                                  fishers: Optional[List[List[torch.Tensor]]] = None,
                                  fisher_floor: float = 1e-6,
                                  favor_target_model: bool = True,
                                  normalize_fishers: bool = True,
                                  combined_mask: Optional[Dict[str, torch.Tensor]] = None,
                                  print_results: bool = True) -> List[MergeResult]:
        """Search for optimal merging coefficients."""
        self.logger.info(f"Searching merging coefficients with {len(coefficients_set)} combinations...")
        
        merged_models = self.generate_merged_for_coeffs_set(
            models=models,
            coefficients_set=coefficients_set,
            feature_dim=feature_dim,
            dropout_rate=dropout_rate,
            fishers=fishers,
            fisher_floor=fisher_floor,
            favor_target_model=favor_target_model,
            normalize_fishers=normalize_fishers,
            combined_mask=combined_mask
        )
        
        results = []
        for coeffs, merged_model in merged_models:
            score = self.evaluate_model(merged_model, test_loader, segment_len)
            result = MergeResult(coefficients=coeffs, score=score)
            results.append(result)
            
            if print_results:
                self.print_merge_result(result)
        
        # Find best result
        best_result = max(results, key=lambda x: x.score['roc_auc'])
        self.logger.info(f"Best result - Coefficients: {best_result.coefficients}, ROC AUC: {best_result.score['roc_auc']:.4f}")
        
        return results


def load_models_and_fishers(checkpoint_paths: List[str], 
                          fisher_paths: Optional[List[str]],
                          feature_dim: int,
                          dropout_rate: float,
                          device: torch.device,
                          logger: Optional[logging.Logger] = None) -> Tuple[List[AD_Model], Optional[List[List[torch.Tensor]]], List[Optional[Dict[str, torch.Tensor]]]]:
    """Load models and their Fisher information."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    models = []
    fishers = None
    masks: List[Optional[Dict[str, torch.Tensor]]] = []
    
    # Load models
    for i, ckpt_path in enumerate(checkpoint_paths):
        logger.info(f"Loading model {i+1}/{len(checkpoint_paths)}: {ckpt_path}")
        model = AD_Model(feature_dim, 512, dropout_rate)
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        models.append(model)
        
        mask_path = ckpt_path + ".mask"
        if os.path.exists(mask_path):
            raw_mask_dict = torch.load(mask_path, map_location='cpu')
            bool_mask = {key: (tensor != 0).to(torch.bool) for key, tensor in raw_mask_dict.items()}
            masks.append(bool_mask)
        else:
            masks.append(None)
    
    # Load Fisher information if provided
    if fisher_paths is not None:
        assert len(fisher_paths) == len(checkpoint_paths), "Number of Fisher files must match number of checkpoints"
        fishers = []
        
        for i, fisher_path in enumerate(fisher_paths):
            logger.info(f"Loading Fisher info {i+1}/{len(fisher_paths)}: {fisher_path}")
            fisher_dict = torch.load(fisher_path, map_location=device)

            if isinstance(fisher_dict, list):
                fisher_list = fisher_dict
            elif isinstance(fisher_dict, dict):
                if "fisher_list" in fisher_dict:
                    fisher_list = fisher_dict["fisher_list"]
                else:
                    def _key_sort_key(key: str):
                        suffix = key.split('_')[-1]
                        return (0, int(suffix)) if suffix.isdigit() else (1, key)
                    sorted_keys = sorted(fisher_dict.keys(), key=_key_sort_key)
                    fisher_list = [fisher_dict[key] for key in sorted_keys]
            else:
                raise ValueError(f"Unsupported Fisher data format in {fisher_path}")

            fishers.append(fisher_list)
    
    return models, fishers, masks
