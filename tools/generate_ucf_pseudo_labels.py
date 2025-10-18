#!/usr/bin/env python3
"""
Generate pseudo label scores for UCF-Crime dataset using normality propagation.
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import logging

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.normprop import normality_propagation


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def load_video_list(file_path):
    """Load video list from text file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Video list file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        video_list = [line.strip() for line in f.readlines() if line.strip()]
    
    return video_list


def load_features(feature_path, video_name, feature_name_end='_res.npy'):
    """Load feature file for a video."""
    # Remove extension and add feature suffix
    video_base = os.path.splitext(video_name)[0]
    feature_file = os.path.join(feature_path, video_base + feature_name_end)
    
    if not os.path.exists(feature_file):
        logging.warning(f"Feature file not found: {feature_file}")
        return None
    
    try:
        features = np.load(feature_file)
        return features
    except Exception as e:
        logging.error(f"Error loading {feature_file}: {e}")
        return None


def generate_pseudo_labels_for_video(features, video_name, logger, k=8):
    """Generate pseudo labels for a single video using normality propagation."""
    if features is None or len(features) == 0:
        logger.warning(f"No features available for {video_name}")
        return None
    
    try:
        # Apply normality propagation with UCF-Crime specific settings
        pseudo_scores = normality_propagation(
            features, 
            k=k, 
            mask=None, 
            is_ucf=True  # Use Euclidean similarity for UCF-Crime
        )
        
        return pseudo_scores
        
    except Exception as e:
        logger.error(f"Error in normality propagation for {video_name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Generate pseudo label scores for UCF-Crime')
    parser.add_argument('--feature_path', required=True, 
                       help='Path to extracted features directory')
    parser.add_argument('--train_list', required=True,
                       help='Path to Anomaly_Train.txt file')
    parser.add_argument('--output_path', required=True,
                       help='Output path for pseudo_label_scores_ucf.npy')
    parser.add_argument('--feature_name_end', default='_res.npy',
                       help='Feature file name suffix')
    parser.add_argument('--k', type=int, default=8,
                       help='Window size for normality propagation')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting UCF-Crime pseudo label generation")
    
    # Load training video list
    logger.info(f"Loading training video list from: {args.train_list}")
    train_videos = load_video_list(args.train_list)
    logger.info(f"Found {len(train_videos)} training videos")
    
    # Generate pseudo labels for each video
    pseudo_scores_dict = {}
    successful_videos = 0
    
    logger.info("Generating pseudo labels...")
    for video_name in tqdm(train_videos, desc="Processing videos"):
        # Load features
        features = load_features(args.feature_path, video_name, args.feature_name_end)
        
        if features is not None:
            # Generate pseudo labels
            pseudo_scores = generate_pseudo_labels_for_video(
                features, video_name, logger, args.k
            )
            
            if pseudo_scores is not None:
                # Store with video name as key (without extension)
                video_key = os.path.splitext(video_name)[0]
                pseudo_scores_dict[video_key] = pseudo_scores
                successful_videos += 1
            else:
                logger.warning(f"Failed to generate pseudo labels for {video_name}")
        else:
            logger.warning(f"Failed to load features for {video_name}")
    
    logger.info(f"Successfully processed {successful_videos}/{len(train_videos)} videos")
    
    # Save pseudo scores dictionary
    if pseudo_scores_dict:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        np.save(args.output_path, pseudo_scores_dict)
        logger.info(f"Saved pseudo label scores to: {args.output_path}")
        
        # Print statistics
        total_scores = sum(len(scores) for scores in pseudo_scores_dict.values())
        avg_length = total_scores / len(pseudo_scores_dict)
        logger.info(f"Statistics:")
        logger.info(f"  - Total videos: {len(pseudo_scores_dict)}")
        logger.info(f"  - Total segments: {total_scores}")
        logger.info(f"  - Average segments per video: {avg_length:.2f}")
        
        # Sample some videos to show score ranges
        sample_videos = list(pseudo_scores_dict.keys())[:5]
        logger.info(f"Sample pseudo score ranges:")
        for video in sample_videos:
            scores = pseudo_scores_dict[video]
            logger.info(f"  - {video}: [{scores.min():.4f}, {scores.max():.4f}]")
            
    else:
        logger.error("No pseudo labels were generated successfully!")
        return 1
    
    logger.info("UCF-Crime pseudo label generation completed successfully")
    return 0


if __name__ == "__main__":
    exit(main())