"""
============================================================
src/models.py
============================================================

VGG Model Definitions for ResoMap Experiments
------------------------------------------------------------

This file defines:

1. `VGG`:
   - Modular VGG-style convolutional network with explicit stages.
   - Stage-wise design enables profiling of accuracy, latency,
     and memory under different input resolutions.
   - Adaptive average pooling allows resolution-agnostic FC layers.
   - Supports VGG11, VGG13, and custom variants from YAML configs.

2. `build_model`:
   - Factory function to construct a model instance from a YAML
     configuration dictionary.
   - Designed to maintain a uniform API across multiple model types.

Key Features:
- Stage-wise modularity with `nn.ModuleDict`.
- Flexible depth and width controlled via YAML.
- Compatible with CPU-only inference and profiling experiments.
- Fully connected classifier with optional dropout.
"""

import torch
import torch.nn as nn


class VGG(nn.Module):
    """
    VGG-style convolutional network with explicit stage boundaries.

    Features:
    - Stage-wise modularity using nn.ModuleDict for profiling.
    - Each stage can contain multiple Conv2d + ReLU blocks.
    - MaxPool2d at the end of each stage.
    - AdaptiveAvgPool2d ensures classifier is resolution-agnostic.
    - Fully connected classifier with optional dropout.

    Intended for:
    - Resolution sensitivity experiments under CPU constraints (ResoMap)
    - Easy extension to VGG11, VGG13, or custom variants
    """

    def __init__(self, stages, fc_layers, dropout=0.5):
        """
        Initialize VGG network.

        Parameters
        ----------
        stages : list of dict
            Stage definitions from YAML. Each dict contains:
                - 'layers': list of [in_channels, out_channels, kernel, stride, padding]
                - 'pool': [pool_kernel, pool_stride]
                - 'stage' (optional): stage name
        fc_layers : list of int
            Fully connected layer sizes.
        dropout : float
            Dropout probability applied after each FC layer.
        """
        super().__init__()

        # Store convolutional stages
        self.stages = nn.ModuleDict()

        for idx, stage_cfg in enumerate(stages):
            blocks = []

            # Build convolutional layers for this stage
            for layer in stage_cfg["layers"]:
                in_c, out_c, k, s, p = layer
                blocks.append(nn.Conv2d(in_c, out_c, k, s, p))
                blocks.append(nn.ReLU(inplace=True))

            # Add max-pooling at the end of the stage
            pool_k, pool_s = stage_cfg["pool"]
            blocks.append(nn.MaxPool2d(kernel_size=pool_k, stride=pool_s))

            # Name stage explicitly if provided, else default
            stage_name = stage_cfg.get("stage", f"stage{idx+1}")
            self.stages[stage_name] = nn.Sequential(*blocks)

        # Adaptive pooling ensures FC input size is consistent across resolutions
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # Build classifier
        fc_seq = []
        in_dim = 512 * 7 * 7  # Matches final conv output channels * spatial size
        for out_dim in fc_layers:
            fc_seq.append(nn.Linear(in_dim, out_dim))
            fc_seq.append(nn.ReLU(inplace=True))
            fc_seq.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.classifier = nn.Sequential(*fc_seq)

    def forward(self, x):
        """
        Forward pass through VGG network.

        Sequentially applies:
        1. Convolutional stages
        2. Adaptive average pooling
        3. Flattening
        4. Fully connected classifier

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, H, W)

        Returns
        -------
        torch.Tensor
            Output logits of shape (N, num_classes)
        """
        for stage in self.stages.values():
            x = stage(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def build_model(model_cfg, resolution=None):
    """
    Construct a model instance from configuration dictionary.

    Parameters
    ----------
    model_cfg : dict
        Dictionary loaded from YAML describing model architecture.
    resolution : int, optional
        Placeholder for API consistency (used by other model variants).

    Returns
    -------
    nn.Module
        Instantiated model.
    """
    model_type = model_cfg["type"].lower()

    if model_type == "vgg":
        return VGG(
            stages=model_cfg["stages"],
            fc_layers=model_cfg["classifier"]["fc_layers"],
            dropout=model_cfg["classifier"].get("dropout", 0.5),
        )

    raise ValueError(f"Unknown model type: {model_cfg['type']}")