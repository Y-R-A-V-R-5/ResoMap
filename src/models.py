"""
============================================================
src/models.py
============================================================

Comprehensive Model Architectures for ResoMap Experiments
------------------------------------------------------------

This module provides implementations of multiple CNN architectures:
1. VGG Family (VGG11, VGG13, VGG16)
2. ResNet Family (ResNet18, ResNet34, ResNet50, ResNet101)
3. MobileNet Family (MobileNetV2, MobileNetV3)
4. Simple CNNs (Custom lightweight models)

All models support:
- Variable input resolutions via adaptive pooling
- Configurable number of classes
- GPU and CPU execution
- Gradient-based explainability (Grad-CAM compatible)
- Resolution sensitivity experiments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import math


# ============================================================
# VGG Family
# ============================================================

class VGG(nn.Module):
    """VGG-style CNN with modular stages."""
    
    def __init__(self, stages: List[Dict], fc_layers: List[int], 
                 num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        
        self.stages = nn.ModuleDict()
        
        # Build convolutional stages
        for stage_cfg in stages:
            stage_name = stage_cfg['stage']
            in_channels, out_channels = stage_cfg['channels']
            num_layers = stage_cfg['num_layers']
            
            layers = []
            for i in range(num_layers):
                layers.append(nn.Conv2d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1
                ))
                layers.append(nn.ReLU(inplace=True))
            
            if stage_cfg.get('pool', True):
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
            self.stages[stage_name] = nn.Sequential(*layers)
        
        # Adaptive pooling for variable resolutions
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Classifier
        classifier_layers = []
        in_features = out_channels * 7 * 7
        
        for out_features in fc_layers:
            classifier_layers.append(nn.Linear(in_features, out_features))
            classifier_layers.append(nn.ReLU(inplace=True))
            classifier_layers.append(nn.Dropout(dropout))
            in_features = out_features
        
        classifier_layers.append(nn.Linear(in_features, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
    
    def forward(self, x):
        for stage in self.stages.values():
            x = stage(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ============================================================
# ResNet Family
# ============================================================

class BasicBlock(nn.Module):
    """ResNet Basic Block."""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class Bottleneck(nn.Module):
    """ResNet Bottleneck Block."""
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet(nn.Module):
    """ResNet architecture."""
    
    def __init__(self, block_type: str, layers: List[int], 
                 channels: List[int], num_classes: int = 10,
                 initial_channels: int = 64):
        super().__init__()
        
        block = BasicBlock if block_type == 'basic' else Bottleneck
        self.in_channels = initial_channels
        
        # Initial layers
        self.conv1 = nn.Conv2d(3, initial_channels, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # ResNet stages
        self.layer1 = self._make_layer(block, channels[0], layers[0])
        self.layer2 = self._make_layer(block, channels[1], layers[1], 2)
        self.layer3 = self._make_layer(block, channels[2], layers[2], 2)
        self.layer4 = self._make_layer(block, channels[3], layers[3], 2)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[3] * block.expansion, num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ============================================================
# MobileNet Family
# ============================================================

class InvertedResidual(nn.Module):
    """MobileNetV2 Inverted Residual Block."""
    
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(in_ch * expand_ratio))
        self.use_res = stride == 1 and in_ch == out_ch
        
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_ch, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


class MobileNetV2(nn.Module):
    """MobileNetV2 architecture."""
    
    def __init__(self, inverted_residual_setting: List[List[int]],
                 num_classes: int = 10, width_mult: float = 1.0,
                 dropout: float = 0.2):
        super().__init__()
        
        input_ch = int(32 * width_mult)
        last_ch = int(1280 * max(1.0, width_mult))
        
        features = [
            nn.Conv2d(3, input_ch, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_ch),
            nn.ReLU6(inplace=True)
        ]
        
        for t, c, n, s in inverted_residual_setting:
            output_ch = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_ch, output_ch, stride, t))
                input_ch = output_ch
        
        features.extend([
            nn.Conv2d(input_ch, last_ch, 1, bias=False),
            nn.BatchNorm2d(last_ch),
            nn.ReLU6(inplace=True)
        ])
        
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(last_ch, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ============================================================
# Simple CNN
# ============================================================

class SimpleCNN(nn.Module):
    """Simple CNN for quick experiments."""
    
    def __init__(self, channels: List[int], fc_layers: List[int],
                 num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        
        conv_layers = []
        in_ch = 3
        
        for out_ch in channels:
            conv_layers.extend([
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ])
            in_ch = out_ch
        
        self.features = nn.Sequential(*conv_layers)
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        classifier_layers = []
        in_feat = channels[-1] * 16
        
        for out_feat in fc_layers:
            classifier_layers.extend([
                nn.Linear(in_feat, out_feat),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_feat = out_feat
        
        classifier_layers.append(nn.Linear(in_feat, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ============================================================
# Model Builder
# ============================================================

def build_model(model_cfg: Dict, num_classes: Optional[int] = None,
                resolution: Optional[int] = None) -> nn.Module:
    """Build model from configuration."""
    model_type = model_cfg['type'].lower()
    num_cls = num_classes or model_cfg.get('num_classes', 10)
    
    if model_type == 'vgg':
        return VGG(
            stages=model_cfg['stages'],
            fc_layers=model_cfg['classifier']['fc_layers'],
            num_classes=num_cls,
            dropout=model_cfg['classifier'].get('dropout', 0.5)
        )
    
    elif model_type == 'resnet':
        return ResNet(
            block_type=model_cfg['block_type'],
            layers=model_cfg['layers'],
            channels=model_cfg['channels'],
            num_classes=num_cls,
            initial_channels=model_cfg.get('initial_channels', 64)
        )
    
    elif model_type == 'mobilenet':
        return MobileNetV2(
            inverted_residual_setting=model_cfg['inverted_residual_setting'],
            num_classes=num_cls,
            width_mult=model_cfg.get('width_mult', 1.0),
            dropout=model_cfg.get('dropout', 0.2)
        )
    
    elif model_type == 'simple':
        return SimpleCNN(
            channels=model_cfg['channels'],
            fc_layers=model_cfg['fc_layers'],
            num_classes=num_cls,
            dropout=model_cfg.get('dropout', 0.3)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_model_from_config(model_name: str, config_path: str = 'configs/models.yaml',
                           num_classes: Optional[int] = None) -> nn.Module:
    """Load model by name from config file."""
    import yaml
    from pathlib import Path
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        all_models = yaml.safe_load(f)
    
    model_name_lower = model_name.lower()
    if model_name_lower not in all_models:
        available = ', '.join(all_models.keys())
        raise ValueError(f"Model '{model_name}' not found. Available: {available}")
    
    model_cfg = all_models[model_name_lower]
    return build_model(model_cfg, num_classes=num_classes)


def get_model_info(model: nn.Module) -> Dict:
    """Get model information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024**2
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'params_millions': total_params / 1e6,
        'model_size_mb': size_mb
    }


def get_target_layer_for_gradcam(model: nn.Module) -> nn.Module:
    """Get target layer for Grad-CAM."""
    if hasattr(model, 'stages') and isinstance(model.stages, nn.ModuleDict):
        stage_keys = list(model.stages.keys())
        return model.stages[stage_keys[-1]]
    elif hasattr(model, 'layer4'):
        return model.layer4
    elif hasattr(model, 'features'):
        return model.features[-1]
    else:
        raise ValueError("Cannot determine target layer")
