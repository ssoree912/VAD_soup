#!/usr/bin/env python3
"""
Comprehensive visualization script for VAD experiments.
This script generates all visualizations for pruning, Fisher soup, and performance analysis.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import pandas as pd

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import AD_Model

# Configure matplotlib for better plots
plt.style.use('default')
sns.set_palette("husl")
matplotlib_backend = 'Agg'
plt.switch_backend(matplotlib_backend)

FLOW_DEFINITIONS = {
    "baseline_soup": [
        {"name": "Dense Training", "start": 0.0, "duration": 3.0, "color": "#1f77b4"},
        {"name": "Checkpoint Selection", "start": 3.0, "duration": 1.0, "color": "#9467bd"},
        {"name": "Uniform Soup Averaging", "start": 4.0, "duration": 1.5, "color": "#ff7f0e"},
        {"name": "Evaluation", "start": 5.5, "duration": 1.0, "color": "#2ca02c"},
    ],
    "pruning_soup": [
        {"name": "Dense Warm-up", "start": 0.0, "duration": 2.0, "color": "#1f77b4"},
        {"name": "Apply Pruning Masks", "start": 2.0, "duration": 0.5, "color": "#d62728"},
        {"name": "Sparse Training", "start": 2.5, "duration": 2.5, "color": "#ff7f0e"},
        {"name": "Sparse Checkpoints", "start": 5.0, "duration": 0.8, "color": "#9467bd"},
        {"name": "Soup Averaging (Sparse)", "start": 5.8, "duration": 1.2, "color": "#17becf"},
        {"name": "Evaluation", "start": 7.0, "duration": 1.0, "color": "#2ca02c"},
    ],
    "unpruning_soup": [
        {"name": "Dense Warm-up", "start": 0.0, "duration": 1.5, "color": "#1f77b4"},
        {"name": "Apply Pruning Masks", "start": 1.5, "duration": 0.3, "color": "#d62728"},
        {"name": "Sparse Training", "start": 1.8, "duration": 2.2, "color": "#ff7f0e"},
        {"name": "Unprune (Release Masks)", "start": 4.0, "duration": 0.2, "color": "#8c564b", "type": "marker"},
        {"name": "Dense Fine-tuning", "start": 4.2, "duration": 1.8, "color": "#bcbd22"},
        {"name": "Fisher Computation", "start": 6.0, "duration": 0.8, "color": "#9467bd"},
        {"name": "Fisher-weighted Soup", "start": 6.8, "duration": 1.2, "color": "#17becf"},
        {"name": "Evaluation", "start": 8.0, "duration": 1.0, "color": "#2ca02c"},
    ],
}


class VADVisualizer:
    """Comprehensive visualizer for VAD experiments."""
    
    def __init__(self, base_path: str, output_dir: str):
        self.base_path = Path(base_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _save_fig(self, fig: plt.Figure, name: str, subdir: str = ""):
        """Save figure to output directory."""
        if subdir:
            save_dir = self.output_dir / subdir
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = self.output_dir
            
        save_path = save_dir / f"{name}.png"
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        self.logger.info(f"Saved {save_path}")
        
    def create_individual_soup_pipelines(self):
        """Create individual training pipeline flowcharts for different soup types."""
        self.logger.info("Creating individual soup training pipelines...")
        
        try:
            # Define the three soup pipelines
            pipelines = {
                'baseline_soup': {
                    'title': 'Baseline Soup Training Pipeline',
                    'stages': [
                        ('Dense\nTraining\n(Multiple\nModels)', 0, 6, '#3498db'),
                        ('Fisher\nComputation', 6, 7, '#9b59b6'),
                        ('Fisher\nSoup\nMerge', 7, 8, '#8e44ad'),
                        ('Final\nModel', 8, 9, '#2c3e50')
                    ],
                    'color_scheme': '#2980b9'
                },
                'pruning_soup': {
                    'title': 'Pruning Soup Training Pipeline',
                    'stages': [
                        ('Dense\nTraining\n(Multiple\nModels)', 0, 3, '#3498db'),
                        ('Pruning\nApplied', 3, 4, '#f39c12'),
                        ('Pruned\nTraining\n(Multiple\nModels)', 4, 7, '#e67e22'),
                        ('Fisher\nComputation', 7, 8, '#9b59b6'),
                        ('Fisher\nSoup\nMerge', 8, 9, '#8e44ad'),
                        ('Final\nModel', 9, 10, '#2c3e50')
                    ],
                    'color_scheme': '#e67e22'
                },
                'unprune_soup': {
                    'title': 'Unprune Soup Training Pipeline', 
                    'stages': [
                        ('Dense\nTraining\n(1% epoch)', 0, 0.5, '#3498db'),
                        ('Unprune\n(30%)', 0.5, 1, '#27ae60'),
                        ('Pruning\nApplied', 1, 2, '#f39c12'),
                        ('Pruned\nTraining', 2, 5, '#e67e22'),
                        ('Dense\nFine-tuning', 5, 7, '#2ecc71'),
                        ('Fisher\nComputation', 7, 8, '#9b59b6'),
                        ('Fisher\nSoup\nMerge', 8, 9, '#8e44ad'),
                        ('Final\nModel', 9, 10, '#2c3e50')
                    ],
                    'color_scheme': '#27ae60'
                }
            }
            
            # Create subplots for all three pipelines
            fig, axes = plt.subplots(3, 1, figsize=(14, 12))
            fig.suptitle('Individual Training Pipeline Flowcharts', fontsize=16, fontweight='bold')
            
            for idx, (pipeline_name, pipeline_info) in enumerate(pipelines.items()):
                ax = axes[idx]
                
                # Draw pipeline stages
                for stage_name, start, end, color in pipeline_info['stages']:
                    # Draw rectangle for stage
                    rect = plt.Rectangle((start, 0.3), end - start, 0.4, 
                                       facecolor=color, alpha=0.8, edgecolor='black', linewidth=1)
                    ax.add_patch(rect)
                    
                    # Add stage text
                    text_x = start + (end - start) / 2
                    ax.text(text_x, 0.5, stage_name, ha='center', va='center', 
                           fontsize=10, fontweight='bold', color='white', wrap=True)
                
                # Draw arrows between stages
                for i in range(len(pipeline_info['stages']) - 1):
                    _, _, end1, _ = pipeline_info['stages'][i]
                    _, start2, _, _ = pipeline_info['stages'][i + 1]
                    
                    # Arrow from end of current stage to start of next
                    ax.arrow(end1, 0.5, start2 - end1 - 0.1, 0, 
                            head_width=0.05, head_length=0.05, fc='black', ec='black')
                
                # Configure subplot
                ax.set_xlim(-0.5, 12.5)
                ax.set_ylim(0, 1)
                ax.set_title(pipeline_info['title'], fontsize=14, fontweight='bold', 
                           color=pipeline_info['color_scheme'], pad=15)
                ax.set_xlabel('Training Progress', fontsize=12)
                ax.set_ylabel('Pipeline Stages', fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.set_yticks([])
                
                # Add stage markers on x-axis
                stage_positions = []
                stage_labels = []
                for stage_name, start, end, _ in pipeline_info['stages']:
                    stage_positions.append(start + (end - start) / 2)
                    stage_labels.append(f'{start}-{end}')
                
                ax.set_xticks(stage_positions[::2])  # Show every other tick to avoid crowding
                ax.set_xticklabels(stage_labels[::2], rotation=45)
            
            plt.tight_layout()
            self._save_fig(fig, "individual_soup_pipelines", "overview")
            
            # Create separate detailed diagrams for each pipeline
            for pipeline_name, pipeline_info in pipelines.items():
                self._create_detailed_pipeline(pipeline_name, pipeline_info)
                
        except Exception as e:
            self.logger.warning(f"Error creating individual soup pipelines: {e}")
            
    def _create_detailed_pipeline(self, pipeline_name, pipeline_info):
        """Create a detailed pipeline diagram for a specific soup type."""
        try:
            fig, ax = plt.subplots(figsize=(16, 6))
            
            # Draw pipeline stages with more detail
            y_pos = 0.5
            stage_height = 0.3
            
            for i, (stage_name, start, end, color) in enumerate(pipeline_info['stages']):
                stage_width = end - start
                
                # Main stage rectangle
                rect = plt.Rectangle((start, y_pos - stage_height/2), stage_width, stage_height,
                                   facecolor=color, alpha=0.8, edgecolor='black', linewidth=2)
                ax.add_patch(rect)
                
                # Stage text
                text_x = start + stage_width / 2
                ax.text(text_x, y_pos, stage_name, ha='center', va='center',
                       fontsize=11, fontweight='bold', color='white', wrap=True)
                
                # Duration text below
                ax.text(text_x, y_pos - stage_height/2 - 0.15, f'Duration: {stage_width}',
                       ha='center', va='center', fontsize=9, style='italic')
                
                # Draw arrows between stages
                if i < len(pipeline_info['stages']) - 1:
                    next_start = pipeline_info['stages'][i + 1][1]
                    arrow_start = start + stage_width
                    arrow_length = next_start - arrow_start
                    
                    if arrow_length > 0:
                        ax.arrow(arrow_start, y_pos, arrow_length - 0.1, 0,
                                head_width=0.08, head_length=0.08, fc='black', ec='black', linewidth=2)
            
            # Configure plot
            ax.set_xlim(-0.5, max(end for _, _, end, _ in pipeline_info['stages']) + 0.5)
            ax.set_ylim(0, 1)
            ax.set_title(f'{pipeline_info["title"]} - Detailed View', 
                        fontsize=16, fontweight='bold', color=pipeline_info['color_scheme'], pad=20)
            ax.set_xlabel('Training Progress', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_yticks([])
            
            # Add timeline markers
            timeline_positions = list(range(0, int(max(end for _, _, end, _ in pipeline_info['stages'])) + 1))
            ax.set_xticks(timeline_positions)
            ax.set_xticklabels([f'T{i}' for i in timeline_positions])
            
            plt.tight_layout()
            filename = f"{pipeline_name}_detailed_pipeline"
            self._save_fig(fig, filename, "overview")
            
        except Exception as e:
            self.logger.warning(f"Error creating detailed pipeline for {pipeline_name}: {e}")
        
    def analyze_pruning_masks(self):
        """Analyze and visualize pruning masks."""
        self.logger.info("Analyzing pruning masks...")
        
        mask_files = list(self.base_path.glob("**/*.mask"))
        if not mask_files:
            self.logger.warning("No mask files found")
            return
            
        # Group masks by category
        categories = {'baseline_soup': [], 'pruning_soup': [], 'unpruning_soup': []}
        
        for mask_file in mask_files:
            path_str = str(mask_file)
            flow = self._determine_flow(mask_file)
            if flow in categories:
                categories[flow].append(mask_file)
                
        # Analyze sparsity for each category
        sparsity_data = []
        
        for category, files in categories.items():
            for mask_file in files:
                try:
                    mask_dict = torch.load(mask_file, map_location='cpu')
                    if isinstance(mask_dict, dict):
                        total_params = 0
                        pruned_params = 0
                        
                        for name, mask in mask_dict.items():
                            mask_array = mask.detach().cpu().float().numpy()
                            total_params += mask_array.size
                            pruned_params += (mask_array == 0).sum()
                            
                        sparsity = pruned_params / total_params if total_params > 0 else 0
                        sparsity_data.append({
                            'category': category,
                            'file': mask_file.name,
                            'sparsity': float(sparsity * 100),
                            'total_params': int(total_params),
                            'pruned_params': int(pruned_params)
                        })
                except Exception as e:
                    self.logger.warning(f"Error processing {mask_file}: {e}")
                    
        if sparsity_data:
            # Plot sparsity comparison
            df = pd.DataFrame(sparsity_data)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Sparsity by category
            sns.boxplot(data=df, x='category', y='sparsity', ax=ax1)
            ax1.set_title('Sparsity Distribution by Category')
            ax1.set_ylabel('Sparsity (%)')
            
            # Individual sparsity values
            sns.stripplot(data=df, x='category', y='sparsity', ax=ax2, size=8, alpha=0.7)
            ax2.set_title('Individual Model Sparsity')
            ax2.set_ylabel('Sparsity (%)')
            
            self._save_fig(fig, "sparsity_analysis", "pruning")
            
            # Save sparsity statistics
            stats_file = self.output_dir / "pruning" / "sparsity_stats.json"
            stats_file.parent.mkdir(parents=True, exist_ok=True)
            with open(stats_file, 'w') as f:
                json.dump(sparsity_data, f, indent=2)
                
    def visualize_layer_sparsity(self, mask_file: Path):
        """Visualize layer-wise sparsity for a specific mask."""
        try:
            mask_dict = torch.load(mask_file, map_location='cpu')
            if not isinstance(mask_dict, dict):
                return
                
            layer_sparsity = []
            for name, mask in mask_dict.items():
                mask_array = mask.detach().cpu().float().numpy()
                sparsity = (mask_array == 0).mean() * 100
                layer_sparsity.append({'layer': name, 'sparsity': sparsity})
                
            if layer_sparsity:
                df = pd.DataFrame(layer_sparsity)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                bars = ax.bar(range(len(df)), df['sparsity'], color='skyblue', alpha=0.7)
                ax.set_xlabel('Layer')
                ax.set_ylabel('Sparsity (%)')
                ax.set_title(f'Layer-wise Sparsity - {mask_file.name}')
                ax.set_xticks(range(len(df)))
                ax.set_xticklabels(df['layer'], rotation=45, ha='right')
                
                # Add value labels on bars
                for bar, sparsity in zip(bars, df['sparsity']):
                    height = bar.get_height()
                    ax.annotate(f'{sparsity:.1f}%',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=8)
                
                filename = f"layer_sparsity_{mask_file.stem}"
                self._save_fig(fig, filename, "pruning")
                
        except Exception as e:
            self.logger.warning(f"Error visualizing layer sparsity for {mask_file}: {e}")
            
    def analyze_fisher_information(self):
        """Analyze Fisher information matrices."""
        self.logger.info("Analyzing Fisher information...")
        
        fisher_files = list(self.base_path.glob("**/fisher_cache/*.pt"))
        if not fisher_files:
            self.logger.warning("No Fisher files found")
            return
            
        fisher_stats = []
        
        for fisher_file in fisher_files:
            try:
                fisher_data = torch.load(fisher_file, map_location='cpu')
                
                if isinstance(fisher_data, dict):
                    if 'fisher_list' in fisher_data:
                        fisher_list = fisher_data['fisher_list']
                    else:
                        fisher_list = list(fisher_data.values())
                elif isinstance(fisher_data, list):
                    fisher_list = fisher_data
                else:
                    continue
                    
                # Compute statistics
                all_values = []
                for fisher_tensor in fisher_list:
                    if torch.is_tensor(fisher_tensor):
                        values = fisher_tensor.detach().cpu().float().numpy().flatten()
                        all_values.extend(values)
                        
                if all_values:
                    all_values = np.array(all_values)
                    stats = {
                        'file': fisher_file.name,
                        'category': self._determine_flow(fisher_file),
                        'mean': float(np.mean(all_values)),
                        'std': float(np.std(all_values)),
                        'min': float(np.min(all_values)),
                        'max': float(np.max(all_values)),
                        'median': float(np.median(all_values)),
                        'total_elements': len(all_values)
                    }
                    fisher_stats.append(stats)
                    
            except Exception as e:
                self.logger.warning(f"Error processing Fisher file {fisher_file}: {e}")
                
        if fisher_stats:
            # Plot Fisher statistics
            df = pd.DataFrame(fisher_stats)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Mean Fisher values
            sns.boxplot(data=df, x='category', y='mean', ax=axes[0,0])
            axes[0,0].set_title('Mean Fisher Information by Category')
            axes[0,0].set_yscale('log')
            
            # Standard deviation
            sns.boxplot(data=df, x='category', y='std', ax=axes[0,1])
            axes[0,1].set_title('Fisher Information Std by Category')
            axes[0,1].set_yscale('log')
            
            # Distribution range
            df['range'] = df['max'] - df['min']
            sns.boxplot(data=df, x='category', y='range', ax=axes[1,0])
            axes[1,0].set_title('Fisher Information Range by Category')
            axes[1,0].set_yscale('log')
            
            # Total elements
            sns.barplot(data=df, x='category', y='total_elements', ax=axes[1,1])
            axes[1,1].set_title('Total Fisher Elements by Category')
            
            plt.tight_layout()
            self._save_fig(fig, "fisher_statistics", "fisher")
            
            # Save Fisher statistics
            stats_file = self.output_dir / "fisher" / "fisher_stats.json"
            stats_file.parent.mkdir(parents=True, exist_ok=True)
            with open(stats_file, 'w') as f:
                json.dump(fisher_stats, f, indent=2)
                
        # Visualize weight vs Fisher correlation for multiple models
        self.visualize_multiple_weight_fisher_correlations()
        
        # Create combined visualization with baseline, pruned, unpruned models
        self.visualize_combined_weight_fisher_correlations()
                
    def visualize_weight_vs_fisher(self, checkpoint_file: Path, fisher_file: Path, feature_dim: int = 2048):
        """Visualize weight magnitude vs Fisher information correlation."""
        try:
            # Load model
            model = AD_Model(feature_dim, 512, 0.6)
            state_dict = torch.load(checkpoint_file, map_location='cpu')
            model.load_state_dict(state_dict)
            
            # Load Fisher information
            fisher_data = torch.load(fisher_file, map_location='cpu')
            if isinstance(fisher_data, dict) and 'fisher_list' in fisher_data:
                fisher_list = fisher_data['fisher_list']
            elif isinstance(fisher_data, list):
                fisher_list = fisher_data
            else:
                return
                
            # Get model parameters (only those with dim > 1)
            model_params = [p for p in model.parameters() if p.requires_grad and p.dim() > 1]
            
            if len(model_params) != len(fisher_list):
                self.logger.warning(f"Parameter count mismatch: {len(model_params)} vs {len(fisher_list)}")
                return
                
            # Collect weight magnitudes and Fisher values
            weights = []
            fishers = []
            
            for param, fisher_tensor in zip(model_params, fisher_list):
                w = param.detach().cpu().abs().numpy().flatten()
                f = fisher_tensor.detach().cpu().numpy().flatten()
                
                # Ensure same size
                min_size = min(len(w), len(f))
                weights.extend(w[:min_size])
                fishers.extend(f[:min_size])
                
            if weights and fishers:
                weights = np.array(weights)
                fishers = np.array(fishers)
                
                # Sample for visualization if too many points
                if len(weights) > 50000:
                    idx = np.random.choice(len(weights), 50000, replace=False)
                    weights = weights[idx]
                    fishers = fishers[idx]
                    
                # Create scatter plot
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Log scale scatter
                ax.scatter(weights, fishers, alpha=0.5, s=1)
                ax.set_xlabel('|Weight|')
                ax.set_ylabel('Fisher Information')
                ax.set_xscale('log')
                ax.set_yscale('log')
                
                # Compute correlation
                # Filter out zero and inf values for correlation
                valid_mask = (weights > 0) & (fishers > 0) & np.isfinite(weights) & np.isfinite(fishers)
                if np.sum(valid_mask) > 1:
                    corr = np.corrcoef(np.log(weights[valid_mask]), np.log(fishers[valid_mask]))[0, 1]
                    ax.set_title(f'Weight vs Fisher Correlation (r = {corr:.3f})\n{checkpoint_file.name}')
                else:
                    ax.set_title(f'Weight vs Fisher\n{checkpoint_file.name}')
                    
                ax.grid(True, alpha=0.3)
                
                filename = f"weight_fisher_corr_{checkpoint_file.stem}"
                self._save_fig(fig, filename, "fisher")
                
        except Exception as e:
            self.logger.warning(f"Error in weight vs Fisher visualization: {e}")
            
    def visualize_multiple_weight_fisher_correlations(self):
        """Visualize weight vs Fisher correlation for multiple best_auc models."""
        try:
            # Find all best_auc.pkl files and filter for shanghaitech
            all_best_auc_files = list(self.base_path.glob("**/best_auc.pkl"))
            shanghaitech_files = [f for f in all_best_auc_files if "shanghaitech" in str(f)]
            
            if not shanghaitech_files:
                self.logger.warning("No shanghaitech best_auc.pkl files found")
                return
                
            # Select diverse models from different categories
            baseline_models = [f for f in shanghaitech_files if "baseline" in str(f) and "kfold" not in str(f)]
            pruned_models = [f for f in shanghaitech_files if "pruned_" in str(f) and "unpruned" not in str(f) and "kfold" not in str(f)]
            unpruned_models = [f for f in shanghaitech_files if "unpruned" in str(f) and "kfold" not in str(f)]
            
            # Select one from each category
            selected_models = []
            if baseline_models:
                selected_models.append(baseline_models[0])
                self.logger.info(f"Selected baseline: {baseline_models[0]}")
            if pruned_models:
                selected_models.append(pruned_models[0])
                self.logger.info(f"Selected pruned: {pruned_models[0]}")
            if unpruned_models:
                selected_models.append(unpruned_models[0])
                self.logger.info(f"Selected unpruned: {unpruned_models[0]}")
                
            # Fill remaining slots with kfold models if needed
            if len(selected_models) < 5:
                kfold_models = [f for f in shanghaitech_files if "kfold" in str(f)]
                remaining_slots = 5 - len(selected_models)
                selected_models.extend(kfold_models[:remaining_slots])
                
            fisher_cache_files = list(self.base_path.glob("**/fisher_cache/*.pt"))
            
            if not selected_models or not fisher_cache_files:
                self.logger.warning("No suitable models or Fisher cache files found")
                return
            
            for i, checkpoint_file in enumerate(selected_models):
                try:
                    self.logger.info(f"Processing model {i+1}: {checkpoint_file}")
                    # Try to find corresponding Fisher file
                    fisher_file = None
                    for f_file in fisher_cache_files:
                        if any(part in str(f_file) for part in str(checkpoint_file).split('/')):
                            fisher_file = f_file
                            break
                    
                    if not fisher_file and fisher_cache_files:
                        # Use first available Fisher file as fallback
                        fisher_file = fisher_cache_files[0]
                        
                    if fisher_file:
                        self.logger.info(f"Using Fisher file: {fisher_file}")
                        self.visualize_weight_vs_fisher_single(checkpoint_file, fisher_file, i)
                        
                except Exception as e:
                    self.logger.warning(f"Error processing {checkpoint_file}: {e}")
                    
        except Exception as e:
            self.logger.warning(f"Error in multiple weight vs Fisher visualization: {e}")
            
    def visualize_weight_vs_fisher_single(self, checkpoint_file: Path, fisher_file: Path, model_idx: int, feature_dim: int = 2048):
        """Visualize weight magnitude vs Fisher information correlation for a single model."""
        try:
            # Load model
            model = AD_Model(feature_dim, 512, 0.6)
            state_dict = torch.load(checkpoint_file, map_location='cpu')
            model.load_state_dict(state_dict)
            
            # Load Fisher information
            fisher_data = torch.load(fisher_file, map_location='cpu')
            if isinstance(fisher_data, dict) and 'fisher_list' in fisher_data:
                fisher_list = fisher_data['fisher_list']
            elif isinstance(fisher_data, list):
                fisher_list = fisher_data
            else:
                return
                
            # Get model parameters (only those with dim > 1)
            model_params = [p for p in model.parameters() if p.requires_grad and p.dim() > 1]
            
            if len(model_params) != len(fisher_list):
                # Try to match by size if counts don't match
                min_len = min(len(model_params), len(fisher_list))
                model_params = model_params[:min_len]
                fisher_list = fisher_list[:min_len]
                
            # Collect weight magnitudes and Fisher values
            weights = []
            fishers = []
            
            for param, fisher_tensor in zip(model_params, fisher_list):
                w = param.detach().cpu().abs().numpy().flatten()
                f = fisher_tensor.detach().cpu().numpy().flatten()
                
                # Ensure same size
                min_size = min(len(w), len(f))
                weights.extend(w[:min_size])
                fishers.extend(f[:min_size])
                
            if weights and fishers:
                weights = np.array(weights)
                fishers = np.array(fishers)
                
                # Sample for visualization if too many points
                if len(weights) > 50000:
                    idx = np.random.choice(len(weights), 50000, replace=False)
                    weights = weights[idx]
                    fishers = fishers[idx]
                    
                # Create scatter plot
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Log scale scatter
                ax.scatter(weights, fishers, alpha=0.5, s=1)
                ax.set_xlabel('|Weight|')
                ax.set_ylabel('Fisher Information')
                ax.set_xscale('log')
                ax.set_yscale('log')
                
                # Compute correlation
                # Filter out zero and inf values for correlation
                valid_mask = (weights > 0) & (fishers > 0) & np.isfinite(weights) & np.isfinite(fishers)
                if np.sum(valid_mask) > 1:
                    corr = np.corrcoef(np.log(weights[valid_mask]), np.log(fishers[valid_mask]))[0, 1]
                    # Extract meaningful path parts
                    path_parts = str(checkpoint_file).split('/')
                    relevant_path = '/'.join([p for p in path_parts if p in ['baseline_soup', 'random_soup', 'kfold_soup', 'baseline', 'pruned', 'unpruned', 'shanghaitech']])
                    ax.set_title(f'Weight vs Fisher Correlation (r = {corr:.3f})\nModel {model_idx+1}: {relevant_path}')
                else:
                    path_parts = str(checkpoint_file).split('/')
                    relevant_path = '/'.join([p for p in path_parts if p in ['baseline_soup', 'random_soup', 'kfold_soup', 'baseline', 'pruned', 'unpruned', 'shanghaitech']])
                    ax.set_title(f'Weight vs Fisher\nModel {model_idx+1}: {relevant_path}')
                    
                ax.grid(True, alpha=0.3)
                
                filename = f"weight_fisher_corr_model_{model_idx+1}_{checkpoint_file.stem}"
                self._save_fig(fig, filename, "fisher")
                
        except Exception as e:
            self.logger.warning(f"Error in single weight vs Fisher visualization: {e}")
            
    def visualize_combined_weight_fisher_correlations(self):
        """Create combined visualization with baseline, pruned, and unpruned models in one plot."""
        try:
            # Find all best_auc.pkl files and filter for shanghaitech
            all_best_auc_files = list(self.base_path.glob("**/best_auc.pkl"))
            shanghaitech_files = [f for f in all_best_auc_files if "shanghaitech" in str(f)]
            
            if not shanghaitech_files:
                self.logger.warning("No shanghaitech best_auc.pkl files found")
                return
                
            # Select one model from each category
            baseline_models = [f for f in shanghaitech_files if "baseline" in str(f) and "kfold" not in str(f)]
            pruned_models = [f for f in shanghaitech_files if "pruned_" in str(f) and "unpruned" not in str(f) and "kfold" not in str(f)]
            unpruned_models = [f for f in shanghaitech_files if "unpruned" in str(f) and "kfold" not in str(f)]
            
            models_to_plot = []
            if baseline_models:
                models_to_plot.append(('Baseline', baseline_models[0], '#1f77b4'))  # Blue
            if pruned_models:
                models_to_plot.append(('Pruned', pruned_models[0], '#ff7f0e'))     # Orange
            if unpruned_models:
                models_to_plot.append(('Early pruned', unpruned_models[0], '#2ca02c'))  # Green
                
            if not models_to_plot:
                self.logger.warning("No suitable models found for combined visualization")
                return
                
            fisher_cache_files = list(self.base_path.glob("**/fisher_cache/*.pt"))
            if not fisher_cache_files:
                self.logger.warning("No Fisher cache files found")
                return
                
            # Create combined plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            all_correlations = []
            
            for model_type, checkpoint_file, color in models_to_plot:
                try:
                    self.logger.info(f"Processing {model_type}: {checkpoint_file}")
                    
                    # Find corresponding Fisher file or use fallback
                    fisher_file = None
                    for f_file in fisher_cache_files:
                        if any(part in str(f_file) for part in str(checkpoint_file).split('/')):
                            fisher_file = f_file
                            break
                    
                    if not fisher_file:
                        fisher_file = fisher_cache_files[0]  # Use first available as fallback
                        
                    # Load model
                    model = AD_Model(2048, 512, 0.6)
                    state_dict = torch.load(checkpoint_file, map_location='cpu')
                    model.load_state_dict(state_dict)
                    
                    # Load Fisher information
                    fisher_data = torch.load(fisher_file, map_location='cpu')
                    if isinstance(fisher_data, dict) and 'fisher_list' in fisher_data:
                        fisher_list = fisher_data['fisher_list']
                    elif isinstance(fisher_data, list):
                        fisher_list = fisher_data
                    else:
                        continue
                        
                    # Get model parameters
                    model_params = [p for p in model.parameters() if p.requires_grad and p.dim() > 1]
                    
                    if len(model_params) != len(fisher_list):
                        min_len = min(len(model_params), len(fisher_list))
                        model_params = model_params[:min_len]
                        fisher_list = fisher_list[:min_len]
                        
                    # Collect weight magnitudes and Fisher values
                    weights = []
                    fishers = []
                    
                    for param, fisher_tensor in zip(model_params, fisher_list):
                        w = param.detach().cpu().abs().numpy().flatten()
                        f = fisher_tensor.detach().cpu().numpy().flatten()
                        
                        min_size = min(len(w), len(f))
                        weights.extend(w[:min_size])
                        fishers.extend(f[:min_size])
                        
                    if weights and fishers:
                        weights = np.array(weights)
                        fishers = np.array(fishers)
                        
                        # Sample for visualization
                        if len(weights) > 20000:
                            idx = np.random.choice(len(weights), 20000, replace=False)
                            weights = weights[idx]
                            fishers = fishers[idx]
                            
                        # Create scatter plot
                        ax.scatter(weights, fishers, alpha=0.6, s=2, color=color, label=model_type)
                        
                        # Compute correlation
                        valid_mask = (weights > 0) & (fishers > 0) & np.isfinite(weights) & np.isfinite(fishers)
                        if np.sum(valid_mask) > 1:
                            corr = np.corrcoef(np.log(weights[valid_mask]), np.log(fishers[valid_mask]))[0, 1]
                            all_correlations.append(f"{model_type}: r = {corr:.3f}")
                            
                except Exception as e:
                    self.logger.warning(f"Error processing {model_type} model: {e}")
                    
            # Configure plot
            ax.set_xlabel('|Weight|', fontsize=12)
            ax.set_ylabel('Fisher Information', fontsize=12)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            
            # Add correlation info to title
            corr_text = '\n'.join(all_correlations)
            ax.set_title(f'Weight vs Fisher Information Correlation\nComparison: Baseline, Pruned, and Early Pruned Models\n{corr_text}', 
                        fontsize=14, pad=20)
            
            plt.tight_layout()
            self._save_fig(fig, "combined_weight_fisher_correlation", "fisher")
            
            self.logger.info("Saved combined weight vs Fisher correlation plot")
            
        except Exception as e:
            self.logger.warning(f"Error in combined weight vs Fisher visualization: {e}")
            
    def create_soup_performance_comparison(self):
        """Compare performance of different soup strategies."""
        self.logger.info("Creating soup performance comparison...")
        
        # Look for soup results
        soup_files = list(self.base_path.glob("**/fisher_soup_merged.pkl"))
        
        performance_data = []
        
        for soup_file in soup_files:
            category = self._determine_flow(soup_file)
            performance_data.append({
                'model': f"{category}_fisher_soup",
                'type': 'Fisher Soup',
                'category': category,
                'file': str(soup_file)
            })
            
        # Add individual model performance (if available)
        individual_models = list(self.base_path.glob("**/best_auc.pkl"))
        for model_file in individual_models:
            category = self._determine_flow(model_file)
            performance_data.append({
                'model': f"{category}_individual",
                'type': 'Individual',
                'category': category,
                'file': str(model_file)
            })
            
        if performance_data:
            # Create comparison plot
            df = pd.DataFrame(performance_data)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Group by category and type
            categories = df['category'].unique()
            types = df['type'].unique()
            
            x = np.arange(len(categories))
            width = 0.35
            
            for i, model_type in enumerate(types):
                type_data = df[df['type'] == model_type]
                counts = [len(type_data[type_data['category'] == cat]) for cat in categories]
                ax.bar(x + i * width, counts, width, label=model_type)
                
            ax.set_xlabel('Model Category')
            ax.set_ylabel('Number of Models')
            ax.set_title('Model Count by Category and Type')
            ax.set_xticks(x + width / 2)
            ax.set_xticklabels(categories)
            ax.legend()
            
            self._save_fig(fig, "model_overview", "performance")
            
            # Save performance data
            perf_file = self.output_dir / "performance" / "model_summary.json"
            perf_file.parent.mkdir(parents=True, exist_ok=True)
            with open(perf_file, 'w') as f:
                json.dump(performance_data, f, indent=2)
                
    def create_efficiency_analysis(self):
        """Analyze efficiency trade-offs between different approaches."""
        self.logger.info("Creating efficiency analysis...")
        
        # Estimate model complexity
        efficiency_data = []
        
        # Analyze different model types
        model_types = {
            'baseline_soup': self.base_path.glob("**/baseline*/best_auc.pkl"),
            'pruning_soup': self.base_path.glob("**/pruned_*/best_auc.pkl"),
            'unpruning_soup': self.base_path.glob("**/unpruned_*/best_auc.pkl"),
            'fisher_soup': self.base_path.glob("**/fisher_soup_merged.pkl")
        }
        
        for model_type, files in model_types.items():
            for file_path in files:
                try:
                    # Load model to get parameter count
                    state_dict = torch.load(file_path, map_location='cpu')
                    
                    total_params = sum(p.numel() for p in state_dict.values() if torch.is_tensor(p))
                    
                    # Check for mask
                    mask_file = Path(str(file_path) + '.mask')
                    active_params = total_params
                    sparsity = 0.0
                    
                    if mask_file.exists():
                        mask_dict = torch.load(mask_file, map_location='cpu')
                        if isinstance(mask_dict, dict):
                            total_mask_elements = 0
                            zero_elements = 0
                            for mask in mask_dict.values():
                                mask_array = mask.detach().cpu().float().numpy()
                                total_mask_elements += mask_array.size
                                zero_elements += (mask_array == 0).sum()
                            if total_mask_elements > 0:
                                sparsity = zero_elements / total_mask_elements
                                active_params = int(total_params * (1 - sparsity))
                                
                    efficiency_data.append({
                        'model_type': model_type,
                        'file': file_path.name,
                        'total_params': total_params,
                        'active_params': active_params,
                        'sparsity': sparsity * 100,
                        'compression_ratio': total_params / max(active_params, 1)
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Error analyzing {file_path}: {e}")
                    
        if efficiency_data:
            df = pd.DataFrame(efficiency_data)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Parameter count comparison
            sns.boxplot(data=df, x='model_type', y='total_params', ax=axes[0,0])
            axes[0,0].set_title('Total Parameter Count by Model Type')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # Active parameters
            sns.boxplot(data=df, x='model_type', y='active_params', ax=axes[0,1])
            axes[0,1].set_title('Active Parameter Count by Model Type')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # Sparsity distribution
            sns.boxplot(data=df, x='model_type', y='sparsity', ax=axes[1,0])
            axes[1,0].set_title('Sparsity Distribution by Model Type')
            axes[1,0].tick_params(axis='x', rotation=45)
            
            # Compression ratio
            sns.boxplot(data=df, x='model_type', y='compression_ratio', ax=axes[1,1])
            axes[1,1].set_title('Compression Ratio by Model Type')
            axes[1,1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            self._save_fig(fig, "efficiency_analysis", "performance")
            
            # Save efficiency data
            eff_file = self.output_dir / "performance" / "efficiency_stats.json"
            eff_file.parent.mkdir(parents=True, exist_ok=True)
            with open(eff_file, 'w') as f:
                json.dump(efficiency_data, f, indent=2)
                
    def _draw_flow_timeline(self, flow_name: str, stages: List[Dict[str, object]]):
        fig, ax = plt.subplots(figsize=(12, 6))
        max_extent = 0.0
        for idx, stage in enumerate(stages):
            stage_type = stage.get("type", "bar")
            start = float(stage["start"])
            duration = float(stage.get("duration", 0.0))
            color = stage.get("color", "#1f77b4")
            max_extent = max(max_extent, start + duration)
            if stage_type == "marker":
                ax.axvline(x=start, color=color, linewidth=3, alpha=0.85)
                ax.text(start, idx + 0.1, stage["name"], rotation=90, va="bottom", ha="center", fontsize=9)
            else:
                ax.barh(idx, duration, left=start, color=color, alpha=0.75, height=0.6)
                ax.text(
                    start + duration / 2,
                    idx,
                    stage["name"],
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                    weight="bold",
                )
        ax.set_xlabel("Relative Training Progress")
        ax.set_xlim(-0.2, max_extent + 0.5)
        ax.set_yticks([])
        ax.grid(axis="x", alpha=0.3)
        title = {
            "baseline_soup": "Baseline Dense → Uniform Soup Flow",
            "pruning_soup": "Pruning-Only Soup Flow",
            "unpruning_soup": "Pruning → Unpruning → Fisher Soup Flow",
        }.get(flow_name, flow_name)
        ax.set_title(title)
        self._save_fig(fig, f"{flow_name}_timeline", "overview")

    def create_training_timeline_visualization(self):
        """Create visualization of the training pipeline for each soup flow."""
        self.logger.info("Creating training timeline visualizations...")
        for flow_name, stages in FLOW_DEFINITIONS.items():
            self._draw_flow_timeline(flow_name, stages)
        
    def _determine_flow(self, file_path: Path) -> str:
        """Determine soup flow for path."""
        path_str = str(file_path).lower()
        if 'baseline' in path_str:
            return 'baseline_soup'
        if 'unpruned' in path_str or 'unprund' in path_str:
            return 'unpruning_soup'
        if 'pruned' in path_str:
            return 'pruning_soup'
        elif 'fisher' in path_str:
            return 'fisher_soup'
        else:
            return 'unknown'
            
    def generate_all_visualizations(self):
        """Generate all visualizations."""
        self.logger.info("Starting comprehensive visualization generation...")
        
        # Create overview
        self.create_training_timeline_visualization()
        
        # Create individual soup training pipelines
        self.create_individual_soup_pipelines()
        
        # Analyze pruning
        self.analyze_pruning_masks()
        
        # Visualize specific mask examples
        mask_files = list(self.base_path.glob("**/unpruned_*/best_auc.pkl.mask"))[:3]
        for mask_file in mask_files:
            self.visualize_layer_sparsity(mask_file)
            
        # Analyze Fisher information
        self.analyze_fisher_information()
        
        # Weight vs Fisher analysis
        checkpoint_files = list(self.base_path.glob("**/unpruned_*/best_auc.pkl"))[:2]
        fisher_files = list(self.base_path.glob("**/fisher_cache/fisher_unpruned_*.pt"))[:2]
        
        for checkpoint_file, fisher_file in zip(checkpoint_files, fisher_files):
            self.visualize_weight_vs_fisher(checkpoint_file, fisher_file)
            
        # Performance analysis
        self.create_soup_performance_comparison()
        self.create_efficiency_analysis()
        
        self.logger.info(f"All visualizations saved to: {self.output_dir}")
        
        # Create summary report
        self._create_summary_report()
        
    def _create_summary_report(self):
        """Create a summary report of all analyses."""
        report_path = self.output_dir / "summary_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("VAD Model Comprehensive Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Generated Visualizations:\n")
            f.write("- Baseline / Pruning / Unpruning Soup Timelines\n")
            f.write("- Pruning Mask Analysis\n")
            f.write("- Layer-wise Sparsity Analysis\n")
            f.write("- Fisher Information Statistics\n")
            f.write("- Weight vs Fisher Correlation\n")
            f.write("- Soup Performance Comparison\n")
            f.write("- Efficiency Analysis\n\n")
            
            f.write("Directory Structure:\n")
            for subdir in ["overview", "pruning", "fisher", "performance"]:
                subdir_path = self.output_dir / subdir
                if subdir_path.exists():
                    f.write(f"- {subdir}/\n")
                    for file in subdir_path.glob("*.png"):
                        f.write(f"  * {file.name}\n")
                    f.write("\n")
                    
        self.logger.info(f"Summary report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive VAD visualization")
    parser.add_argument("--base_path", type=str, 
                       default="/Users/hwangsolhee/Desktop/mlpr/VAD_soup/ckpts",
                       help="Base path to checkpoint directory")
    parser.add_argument("--output_dir", type=str,
                       default="/Users/hwangsolhee/Desktop/mlpr/VAD_soup/visualizations",
                       help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    visualizer = VADVisualizer(args.base_path, args.output_dir)
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()
