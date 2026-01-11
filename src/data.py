"""
data.py - Enhanced Data Loading Utilities for ResoMap GPU Experiments

This module provides comprehensive utilities for preparing image datasets
with GPU-optimized loading and advanced augmentation strategies.

Main Functionality:
1. get_data_loaders:
   - Loads images from train/val/test subdirectories
   - Applies resolution-aware augmentation
   - GPU-optimized loading with pin_memory and prefetching
   - Supports normalization with ImageNet statistics

2. ResolutionAwareAugmentation:
   - Adaptive augmentation based on input resolution
   - Stronger augmentation for higher resolutions
   - Preserves important features at low resolutions

Design Notes:
- Assumes dataset structure: data_dir/{train,val,test}/
- Optimized for GPU training with persistent workers
- Resolution-specific augmentation strategies
- Supports advanced transforms like color jitter, random crops
"""

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple, Optional


class ResolutionAwareAugmentation:
    """
    Adaptive augmentation strategy based on input resolution.
    
    Higher resolutions get stronger augmentation to prevent overfitting
    and help the model learn robust features. Lower resolutions get
    minimal augmentation to preserve important details.
    """
    
    def __init__(self, resolution: int, augment_config: Optional[dict] = None):
        """
        Initialize resolution-aware augmentation.
        
        Parameters
        ----------
        resolution : int
            Input image resolution
        augment_config : dict, optional
            Augmentation configuration from config.yaml
        """
        self.resolution = resolution
        self.config = augment_config or {}
        
    def get_train_transforms(self) -> transforms.Compose:
        """Get training transforms based on resolution."""
        transform_list = []
        
        # Base resize
        transform_list.append(transforms.Resize((self.resolution, self.resolution)))
        
        # Resolution-dependent random crop
        if self.resolution >= 256 and self.config.get('random_crop', {}).get('enabled', False):
            scale = self.config['random_crop'].get('scale', [0.8, 1.0])
            transform_list.append(transforms.RandomResizedCrop(
                self.resolution, 
                scale=scale,
                interpolation=transforms.InterpolationMode.BILINEAR
            ))
        
        # Horizontal flip
        if self.config.get('horizontal_flip', True):
            transform_list.append(transforms.RandomHorizontalFlip())
        
        # Rotation (scale with resolution)
        rotation_deg = self.config.get('rotation', 10)
        if self.resolution >= 128:
            transform_list.append(transforms.RandomRotation(rotation_deg))
        
        # Color jitter (stronger for higher resolutions)
        color_config = self.config.get('color_jitter', {})
        if color_config.get('enabled', False):
            brightness = color_config.get('brightness', 0.2)
            contrast = color_config.get('contrast', 0.2)
            saturation = color_config.get('saturation', 0.2)
            hue = color_config.get('hue', 0.1)
            
            # Scale intensity with resolution
            if self.resolution < 128:
                brightness *= 0.5
                contrast *= 0.5
                saturation *= 0.5
                hue *= 0.5
            
            transform_list.append(transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue
            ))
        
        # Gaussian blur (only for high resolutions)
        blur_config = self.config.get('gaussian_blur', {})
        if blur_config.get('enabled', False) and self.resolution >= 224:
            kernel_size = blur_config.get('kernel_size', 5)
            transform_list.append(transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0)))
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalization
        norm_config = self.config.get('normalization', {})
        if norm_config.get('enabled', True):
            mean = norm_config.get('mean', [0.485, 0.456, 0.406])
            std = norm_config.get('std', [0.229, 0.224, 0.225])
            transform_list.append(transforms.Normalize(mean=mean, std=std))
        
        return transforms.Compose(transform_list)
    
    def get_val_transforms(self) -> transforms.Compose:
        """Get validation/test transforms (no augmentation)."""
        transform_list = [
            transforms.Resize((self.resolution, self.resolution)),
            transforms.ToTensor()
        ]
        
        # Normalization
        norm_config = self.config.get('normalization', {})
        if norm_config.get('enabled', True):
            mean = norm_config.get('mean', [0.485, 0.456, 0.406])
            std = norm_config.get('std', [0.229, 0.224, 0.225])
            transform_list.append(transforms.Normalize(mean=mean, std=std))
        
        return transforms.Compose(transform_list)


def get_data_loaders(
    data_dir: str,
    resolution: int,
    batch_size: int,
    num_workers: int = 4,
    augment: bool = False,
    augment_config: Optional[dict] = None,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create GPU-optimized PyTorch DataLoaders with resolution-aware augmentation.

    Parameters
    ----------
    data_dir : str
        Path to dataset directory with 'train', 'val', 'test' subfolders
    resolution : int
        Image resolution (images resized to resolution x resolution)
    batch_size : int
        Batch size
    num_workers : int, optional
        Number of data loading workers (default: 4)
    augment : bool, optional
        Enable training augmentation (default: False)
    augment_config : dict, optional
        Augmentation configuration from config.yaml
    pin_memory : bool, optional
        Pin memory for faster GPU transfer (default: True)
    prefetch_factor : int, optional
        Number of batches to prefetch per worker (default: 2)
    persistent_workers : bool, optional
        Keep workers alive between epochs (default: False)

    Returns
    -------
    tuple
        (train_loader, val_loader, test_loader)
    """
    
    # Setup resolution-aware augmentation
    if augment and augment_config is not None:
        augmenter = ResolutionAwareAugmentation(resolution, augment_config)
        train_transform = augmenter.get_train_transforms()
        val_transform = augmenter.get_val_transforms()
    else:
        # Simple transforms without augmentation
        base_transforms = [
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor()
        ]
        
        train_list = base_transforms.copy()
        if augment:
            # Minimal default augmentation
            train_list.insert(1, transforms.RandomHorizontalFlip())
            train_list.insert(2, transforms.RandomRotation(10))
        
        train_transform = transforms.Compose(train_list)
        val_transform = transforms.Compose(base_transforms)

    # Helper to load dataset safely
    def load_dataset(subset: str, transform):
        path = os.path.join(data_dir, subset)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected directory not found: {path}")
        return datasets.ImageFolder(path, transform=transform)

    # Load datasets with appropriate transforms
    train_ds = load_dataset("train", train_transform)
    val_ds = load_dataset("val", val_transform)
    test_ds = load_dataset("test", val_transform)
    
    # DataLoader kwargs for GPU optimization
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    
    # Add prefetch_factor only if num_workers > 0
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = prefetch_factor
        loader_kwargs['persistent_workers'] = persistent_workers

    # Create DataLoaders
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader