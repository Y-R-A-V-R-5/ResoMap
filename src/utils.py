"""
ResoMap Utilities

This module provides reusable utilities for managing datasets in ResoMap experiments.
It includes:

1. move_tree_with_progress:
   Recursively move a directory tree from a source to a destination with a tqdm progress bar.

2. download_and_prepare_dataset:
   Download a Kaggle dataset via KaggleHub and move it to a canonical location,
   using the KaggleHub cache as a temporary transport layer.

Design Notes:
- All paths are explicit; no assumptions about project structure.
- Progress bars reflect actual file operations.
- Purely filesystem + KaggleHub logic; no ML dependencies.
"""

import os
import shutil
from tqdm import tqdm
import kagglehub
from pathlib import Path
import yaml


def move_tree_with_progress(src: str, dst: str, desc: str = "[ResoMap] Moving files") -> None:
    """
    Recursively move a directory tree from src to dst with a tqdm progress bar.

    Parameters
    ----------
    src : str
        Source directory path
    dst : str
        Destination directory path
    desc : str
        Description displayed in the progress bar
    """

    # Collect all files first to have deterministic progress
    file_list = [
        os.path.relpath(os.path.join(root, file), src)
        for root, _, files in os.walk(src)
        for file in files
    ]

    # Ensure destination directories exist
    for rel_path in file_list:
        os.makedirs(os.path.dirname(os.path.join(dst, rel_path)), exist_ok=True)

    # Move files with progress bar
    for rel_path in tqdm(file_list, desc=desc, unit="file"):
        shutil.move(os.path.join(src, rel_path), os.path.join(dst, rel_path))


def download_and_prepare_dataset(
    kaggle_id: str,
    target_dir: str,
    cleanup_cache: bool = False,
) -> str:
    """
    Download a Kaggle dataset via KaggleHub and move it to a specified directory.

    Parameters
    ----------
    kaggle_id : str
        Kaggle dataset identifier, e.g., 'vinline/unified-dataset-for-skin-cancer-classification'
    target_dir : str
        Absolute path where the dataset should be stored permanently
    cleanup_cache : bool, optional
        If True, delete the KaggleHub temporary cache after moving the dataset

    Returns
    -------
    str
        Absolute path to the prepared dataset directory
    """

    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Fast path: use existing dataset if present
    if os.path.exists(target_dir) and os.listdir(target_dir):
        print(f"[ResoMap] Using existing dataset at: {target_dir}")
        return os.path.abspath(target_dir)

    # Download dataset via KaggleHub (temporary cache)
    print(f"[ResoMap] Downloading dataset from KaggleHub: {kaggle_id}")
    cached_path = kagglehub.dataset_download(kaggle_id)

    # Handle single-root nested datasets
    contents = os.listdir(cached_path)
    if len(contents) == 1 and os.path.isdir(os.path.join(cached_path, contents[0])):
        cached_path = os.path.join(cached_path, contents[0])

    # Move dataset to canonical location with progress
    print(f"[ResoMap] Moving dataset to canonical location: {target_dir}")
    move_tree_with_progress(cached_path, target_dir, desc="[ResoMap] Moving dataset")

    # Optional cleanup of empty cache folder
    if cleanup_cache and os.path.exists(cached_path):
        print("[ResoMap] Cleaning KaggleHub cache folder")
        shutil.rmtree(cached_path, ignore_errors=True)

    return os.path.abspath(target_dir)


def load_config(path: Path) -> dict:
    """
    Load ResoMap configuration with automatic modular config loading.
    
    Loads the main config.yaml and automatically merges modular configs:
    - sweep.yaml: Experiment grid (models, resolutions)
    - training.yaml: Training hyperparameters
    - system.yaml: GPU/system settings
    - data.yaml: Dataset and augmentation
    - explainability.yaml: Interpretation methods
    - mlflow.yaml: Experiment tracking
    
    Parameters
    ----------
    path : Path
        Path to main config.yaml file
        
    Returns
    -------
    dict
        Merged configuration dictionary with keys:
        'sweep', 'training', 'system', 'data', 'explainability', 'mlflow'
    """
    # Load main config
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    # Get config directory
    config_dir = path.parent
    
    # Load modular configs
    modular_configs = {
        'sweep': 'sweep.yaml',
        'training': 'training.yaml',
        'system': 'system.yaml',
        'data': 'data.yaml',
        'explainability': 'explainability.yaml',
        'mlflow': 'mlflow.yaml'
    }
    
    for key, filename in modular_configs.items():
        config_path = config_dir / filename
        if config_path.exists():
            with open(config_path, "r") as f:
                config[key] = yaml.safe_load(f)
        else:
            # Fallback: config might already have this section (backward compatibility)
            if key not in config:
                print(f"Warning: {filename} not found and '{key}' not in config.yaml")
    
    return config