"""
data.py - Data Loading Utilities for ResoMap Experiments

This module provides generic, reusable utilities for preparing image datasets
for training, validation, and testing in ResoMap experiments. It is built on 
PyTorch's torchvision and DataLoader abstractions.

Main Functionality:
1. get_data_loaders:
   - Loads images from train/val/test subdirectories of a dataset folder.
   - Applies resizing, optional data augmentation, and converts images to tensors.
   - Returns PyTorch DataLoaders for easy batching and shuffling.

Design Notes:
- Assumes the dataset directory has the following structure:
      data_dir/
          train/
          val/
          test/
- Optional augmentation is applied only to training data via horizontal flip
  and small rotations (configurable by the `augment` flag).
- Transformations are applied consistently across datasets.
- Error checking ensures required directories exist.
- Uses a flexible number of worker threads for efficient data loading.

Example Usage:
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir="/path/to/data",
        resolution=224,
        batch_size=32,
        num_workers=4,
        augment=True
    )
"""

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loaders(
    data_dir: str,
    resolution: int,
    batch_size: int,
    num_workers: int = 4,
    augment: bool = False
):
    """
    Create PyTorch DataLoaders for train, validation, and test datasets.

    Parameters
    ----------
    data_dir : str
        Path to the dataset directory containing 'train', 'val', and 'test' subfolders.
    resolution : int
        Desired image resolution (images will be resized to resolution x resolution).
    batch_size : int
        Batch size for DataLoader.
    num_workers : int, optional
        Number of subprocesses to use for data loading (default: 4).
    augment : bool, optional
        If True, apply simple data augmentations (horizontal flip, small rotation).

    Returns
    -------
    tuple
        (train_loader, val_loader, test_loader) DataLoader objects.
    """

    # Build list of transformations
    transform_list = [transforms.Resize((resolution, resolution)), transforms.ToTensor()]

    # Add augmentation if requested
    if augment:
        transform_list.insert(1, transforms.RandomHorizontalFlip())
        transform_list.insert(2, transforms.RandomRotation(10))  # rotate +/-10 degrees

    transform = transforms.Compose(transform_list)

    # Helper function to safely create dataset
    def load_dataset(subset: str):
        path = os.path.join(data_dir, subset)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected directory not found: {path}")
        return datasets.ImageFolder(path, transform=transform)

    # Load datasets
    train_ds = load_dataset("train")
    val_ds = load_dataset("val")
    test_ds = load_dataset("test")

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader