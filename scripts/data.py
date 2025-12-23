"""
scripts/data.py - Dataset Preparation Script for ResoMap Experiments

This script handles:
1. Loading project configuration from a YAML file.
2. Resolving file paths relative to the project root.
3. Downloading and moving datasets from Kaggle via `download_and_prepare_dataset`.
4. Optional force re-download of dataset if it already exists.

Usage:
    python scripts/data.py [--force]

Outputs:
- Downloads/moves dataset to a canonical location specified in the configuration.
- Prints the absolute path to the prepared dataset.
"""

import sys
from pathlib import Path
import yaml
import argparse
import shutil

# Add project root to Python path to import local modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import download_and_prepare_dataset, load_config

def resolve_path(relative_path: str) -> Path:
    """
    Resolve a path relative to the project root.

    Parameters
    ----------
    relative_path : str
        Relative path from the project root.

    Returns
    -------
    Path
        Absolute Path object.
    """
    return PROJECT_ROOT / relative_path

def prepare_dataset(force: bool = False) -> str:
    """
    Download and move dataset to the canonical project directory as specified in config.
    Can force re-download even if dataset already exists.

    Parameters
    ----------
    force : bool, optional
        If True, delete existing dataset folder and re-download.

    Returns
    -------
    str
        Absolute path to the prepared dataset.
    """
    # Load project config
    config = load_config(resolve_path("configs/config.yaml"))
    data_cfg = config.get("data", {})

    # Target dataset path
    target_path = resolve_path(data_cfg["raw_path"])

    # Force removal if requested
    if force and target_path.exists():
        print(f"[ResoMap] Force re-download enabled. Removing existing dataset at: {target_path}")
        shutil.rmtree(target_path, ignore_errors=True)

    # Download/move dataset using utility
    dataset_path = download_and_prepare_dataset(
        kaggle_id=data_cfg["kaggle_id"],
        target_dir=target_path,
        cleanup_cache=data_cfg.get("cleanup_kaggle_cache", False)
    )

    print(f"[ResoMap] Dataset ready at: {dataset_path}")
    return dataset_path

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Prepare dataset for ResoMap experiments.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if dataset exists."
    )
    args = parser.parse_args()

    # Prepare dataset
    prepare_dataset(force=args.force)