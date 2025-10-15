import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import AD_Model


class FisherVAD:
    """Fisher Information computation for VAD models."""
    
    def __init__(self, model: AD_Model, device: torch.device, logger: Optional[logging.Logger] = None):
        self.model = model
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
    def get_mergeable_variables(self) -> List[torch.nn.Parameter]:
        """Get model parameters that can be merged (excluding non-trainable parameters)."""
        mergeable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.dim() > 1:
                mergeable_params.append(param)
        return mergeable_params
    
    def _compute_fisher_for_batch(self, batch_data: Dict, variables: List[torch.nn.Parameter]) -> List[torch.Tensor]:
        """Compute Fisher Information for a single batch."""
        features, pseudo_labels, reweight, _, _ = batch_data
        bs, nc, t, dim = features.shape
        
        features = features.type(torch.float).to(self.device)
        pseudo_labels = pseudo_labels.type(torch.float).to(self.device)
        
        batch_fishers = []
        for param in variables:
            batch_fishers.append(torch.zeros_like(param, device=self.device))
        
        # Process each sample in the batch
        for b in range(bs):
            for crop in range(nc):
                sample_features = features[b:b+1, crop:crop+1]  # [1, 1, t, dim]
                sample_label = pseudo_labels[b, crop:crop+1]  # [1, t]
                
                # Compute gradients for this sample
                sample_fishers = self._compute_fisher_single_sample(
                    sample_features, sample_label, variables
                )
                
                # Accumulate Fisher information
                for i, fisher in enumerate(sample_fishers):
                    batch_fishers[i] += fisher
        
        # Average over batch size and crops
        total_samples = bs * nc
        for fisher in batch_fishers:
            fisher /= total_samples
            
        return batch_fishers
    
    def _compute_fisher_single_sample(self, features: torch.Tensor, 
                                    labels: torch.Tensor, 
                                    variables: List[torch.nn.Parameter]) -> List[torch.Tensor]:
        """Compute Fisher Information for a single sample using the expectation over model outputs."""
        with torch.enable_grad():
            # Forward pass
            outputs = self.model(features)  # [1, t]
            outputs = outputs.squeeze(0)  # [t]
            
            sample_fishers = []
            for param in variables:
                sample_fishers.append(torch.zeros_like(param, device=self.device))
            
            # Process fewer time steps to reduce memory
            time_steps = min(outputs.shape[0], 10)  # Limit to 10 time steps
            step_size = max(1, outputs.shape[0] // time_steps)
            
            # Compute Fisher for sampled time steps
            for i in range(0, outputs.shape[0], step_size):
                if i >= time_steps * step_size:
                    break
                    
                logit = outputs[i]
                prob = torch.sigmoid(logit)
                
                # Simplified Fisher computation: only use expected gradients
                # For binary classification, use prob as weight
                log_prob = F.logsigmoid(logit) if prob > 0.5 else F.logsigmoid(-logit)
                
                try:
                    grads = torch.autograd.grad(
                        outputs=log_prob, 
                        inputs=variables, 
                        retain_graph=True,
                        create_graph=False
                    )
                    
                    # Accumulate Fisher information (gradient squared)
                    for j, grad in enumerate(grads):
                        if grad is not None:
                            sample_fishers[j] += grad ** 2
                            
                except RuntimeError:
                    # Skip if gradient computation fails
                    continue
                
                # Clear gradients
                for param in variables:
                    if param.grad is not None:
                        param.grad.zero_()
            
            # Average over time steps
            for fisher in sample_fishers:
                fisher /= time_steps
                
        return sample_fishers
    
    def compute_fisher_for_model(self, dataloader, max_batches=100) -> List[torch.Tensor]:
        """Compute Fisher Information Matrix for the entire model using the given dataloader."""
        self.logger.info("Computing Fisher Information Matrix for VAD model...")
        
        variables = self.get_mergeable_variables()
        self.logger.info(f"Found {len(variables)} mergeable parameters")
        
        # Initialize Fisher accumulators
        fishers = []
        for param in variables:
            fishers.append(torch.zeros_like(param, device=self.device))
        
        self.model.eval()
        n_batches = 0
        
        with torch.no_grad():
            # Set requires_grad=True for Fisher computation
            for param in variables:
                param.requires_grad_(True)
        
        for batch_idx, batch_data in enumerate(dataloader):
            if batch_idx >= max_batches:
                self.logger.info(f"Reached maximum batches limit ({max_batches})")
                break
                
            try:
                batch_fishers = self._compute_fisher_for_batch(batch_data, variables)
                
                # Accumulate Fisher information
                for i, batch_fisher in enumerate(batch_fishers):
                    fishers[i] += batch_fisher.detach()
                
                n_batches += 1
                
                # Clear cache more frequently
                if batch_idx % 5 == 0:
                    torch.cuda.empty_cache()
                    self.logger.info(f"Processed batch {batch_idx + 1}")
                    
            except torch.cuda.OutOfMemoryError:
                self.logger.warning(f"CUDA OOM at batch {batch_idx}, clearing cache and continuing...")
                torch.cuda.empty_cache()
                continue
        
        # Average over all batches
        for fisher in fishers:
            fisher /= n_batches
        
        self.logger.info(f"Fisher computation completed. Processed {n_batches} batches.")
        return fishers


def save_fisher_info(fishers: List[torch.Tensor], variables: List[torch.nn.Parameter], 
                    save_path: str, logger: Optional[logging.Logger] = None):
    """Save Fisher Information to disk."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    fisher_dict = {}
    for i, (fisher, param) in enumerate(zip(fishers, variables)):
        param_name = f"param_{i}"
        # Find the actual parameter name in the model
        for name, p in param.named_parameters() if hasattr(param, 'named_parameters') else []:
            if p is param:
                param_name = name
                break
        
        fisher_dict[param_name] = fisher.cpu()
    
    torch.save(fisher_dict, save_path)
    logger.info(f"Saved Fisher Information to {save_path}")


def load_fisher_info(load_path: str, device: torch.device, 
                    logger: Optional[logging.Logger] = None) -> Dict[str, torch.Tensor]:
    """Load Fisher Information from disk."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    fisher_dict = torch.load(load_path, map_location=device)
    logger.info(f"Loaded Fisher Information from {load_path}")
    return fisher_dict
