#!/usr/bin/env python3
"""
Complete setup script for UCF-Crime dataset.
Downloads required files, extracts features, and generates pseudo labels.
"""

import os
import sys
import argparse
import logging
import subprocess
import urllib.request
from pathlib import Path


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def download_file(url, output_path, logger):
    """Download a file from URL."""
    try:
        logger.info(f"Downloading {url} to {output_path}")
        urllib.request.urlretrieve(url, output_path)
        logger.info(f"Downloaded successfully: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def download_ucf_annotations(data_dir, logger):
    """Download UCF-Crime annotation files."""
    base_url = "https://www.crcv.ucf.edu/projects/real-world/"
    files_to_download = [
        "Temporal_Anomaly_Annotation_New.txt",
        "Anomaly_Train.txt", 
        "Anomaly_Test.txt"
    ]
    
    os.makedirs(data_dir, exist_ok=True)
    
    for filename in files_to_download:
        url = base_url + filename
        output_path = os.path.join(data_dir, filename)
        
        if os.path.exists(output_path):
            logger.info(f"File already exists: {output_path}")
            continue
            
        if not download_file(url, output_path, logger):
            logger.error(f"Failed to download {filename}")
            return False
    
    return True


def download_model(model_dir, logger):
    """Download pre-trained ResNeXt-101 model."""
    model_url = "https://www.dropbox.com/s/u8aelclp24v1ek2/resnext-101-kinetics.pth"
    model_path = os.path.join(model_dir, "resnext-101-kinetics.pth")
    
    if os.path.exists(model_path):
        logger.info(f"Model already exists: {model_path}")
        return model_path
    
    os.makedirs(model_dir, exist_ok=True)
    
    if download_file(model_url, model_path, logger):
        return model_path
    else:
        return None


def clone_3d_resnets_repo(repo_dir, logger):
    """Clone 3D-ResNets-PyTorch repository."""
    repo_url = "https://github.com/kenshohara/3D-ResNets-PyTorch.git"
    
    if os.path.exists(repo_dir):
        logger.info(f"Repository already exists: {repo_dir}")
        return True
    
    try:
        logger.info(f"Cloning {repo_url} to {repo_dir}")
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
        logger.info("Repository cloned successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone repository: {e}")
        return False


def run_feature_extraction(video_dir, output_dir, model_path, repo_dir, logger):
    """Run feature extraction script."""
    script_path = "tools/extract_ucf_features.py"
    
    if not os.path.exists(video_dir):
        logger.error(f"Video directory not found: {video_dir}")
        logger.info("Please download UCF-Crime videos and specify the correct path")
        return False
    
    # Add 3D-ResNets repo to Python path temporarily
    env = os.environ.copy()
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{repo_dir}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = repo_dir
    
    cmd = [
        "python", script_path,
        "--video_dir", video_dir,
        "--output_dir", output_dir,
        "--model_path", model_path,
        "--segment_len", "16",
        "--overlap", "8"
    ]
    
    try:
        logger.info("Starting feature extraction...")
        logger.info(f"Command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env=env)
        logger.info("Feature extraction completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Feature extraction failed: {e}")
        return False


def run_pseudo_label_generation(feature_dir, train_list, output_path, logger):
    """Run pseudo label generation script."""
    script_path = "tools/generate_ucf_pseudo_labels.py"
    
    cmd = [
        "python", script_path,
        "--feature_path", feature_dir,
        "--train_list", train_list,
        "--output_path", output_path,
        "--k", "8"
    ]
    
    try:
        logger.info("Starting pseudo label generation...")
        logger.info(f"Command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logger.info("Pseudo label generation completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Pseudo label generation failed: {e}")
        return False


def verify_setup(data_dir, logger):
    """Verify that all required files are present."""
    required_files = [
        "Temporal_Anomaly_Annotation_New.txt",
        "Anomaly_Train.txt",
        "Anomaly_Test.txt",
        "pseudo_label_scores_ucf.npy"
    ]
    
    missing_files = []
    for filename in required_files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)
    
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False
    
    # Check if features directory exists and has files
    features_dir = os.path.join(data_dir, "features")
    if not os.path.exists(features_dir):
        logger.error(f"Features directory not found: {features_dir}")
        return False
    
    feature_files = list(Path(features_dir).glob("*.npy"))
    if len(feature_files) == 0:
        logger.error("No feature files found in features directory")
        return False
    
    logger.info(f"Setup verification passed! Found {len(feature_files)} feature files")
    return True


def main():
    parser = argparse.ArgumentParser(description='Setup UCF-Crime dataset for VAD training')
    parser.add_argument('--video_dir', 
                       help='Directory containing UCF-Crime videos (required for feature extraction)')
    parser.add_argument('--data_dir', default='./data/ucf-crime',
                       help='Output directory for UCF-Crime data')
    parser.add_argument('--model_dir', default='./models',
                       help='Directory to store pre-trained models')
    parser.add_argument('--repo_dir', default='./3D-ResNets-PyTorch',
                       help='Directory for 3D-ResNets-PyTorch repository')
    parser.add_argument('--skip_features', action='store_true',
                       help='Skip feature extraction (if features already exist)')
    parser.add_argument('--skip_download', action='store_true',
                       help='Skip downloading annotation files')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info("Starting UCF-Crime dataset setup")
    
    # Step 1: Download annotation files
    if not args.skip_download:
        logger.info("Step 1: Downloading UCF-Crime annotation files...")
        if not download_ucf_annotations(args.data_dir, logger):
            logger.error("Failed to download annotation files")
            return 1
    else:
        logger.info("Step 1: Skipping annotation download")
    
    # Step 2: Download pre-trained model
    logger.info("Step 2: Downloading pre-trained ResNeXt-101 model...")
    model_path = download_model(args.model_dir, logger)
    if not model_path:
        logger.error("Failed to download pre-trained model")
        return 1
    
    # Step 3: Clone 3D-ResNets repository
    logger.info("Step 3: Setting up 3D-ResNets-PyTorch repository...")
    if not clone_3d_resnets_repo(args.repo_dir, logger):
        logger.error("Failed to setup 3D-ResNets repository")
        return 1
    
    # Step 4: Extract features (if video directory provided)
    features_dir = os.path.join(args.data_dir, "features")
    if not args.skip_features and args.video_dir:
        logger.info("Step 4: Extracting features from videos...")
        if not run_feature_extraction(args.video_dir, features_dir, model_path, args.repo_dir, logger):
            logger.error("Failed to extract features")
            return 1
    else:
        if args.skip_features:
            logger.info("Step 4: Skipping feature extraction")
        else:
            logger.warning("Step 4: Skipping feature extraction (no video directory provided)")
            logger.info("Please run feature extraction manually when you have the videos")
    
    # Step 5: Generate pseudo labels
    train_list = os.path.join(args.data_dir, "Anomaly_Train.txt")
    pseudo_scores_path = os.path.join(args.data_dir, "pseudo_label_scores_ucf.npy")
    
    if os.path.exists(features_dir) and os.path.exists(train_list):
        logger.info("Step 5: Generating pseudo label scores...")
        if not run_pseudo_label_generation(features_dir, train_list, pseudo_scores_path, logger):
            logger.error("Failed to generate pseudo labels")
            return 1
    else:
        logger.warning("Step 5: Skipping pseudo label generation (features or train list not available)")
    
    # Step 6: Verify setup
    logger.info("Step 6: Verifying setup...")
    if verify_setup(args.data_dir, logger):
        logger.info("UCF-Crime dataset setup completed successfully!")
        logger.info(f"Data directory: {args.data_dir}")
        logger.info("You can now run: python main.py --load_config config/config_ucf.yaml")
    else:
        logger.error("Setup verification failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())