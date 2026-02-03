# Models & Methods - Architecture & Implementation Details

## Overview

ResoMap implements multiple CNN architectures from `src/models.py` (443 lines), each optimized for different performance/accuracy tradeoffs. All models support variable input resolutions via adaptive pooling.

**Training Status:**
- ✅ **Trained:** simple_cnn, tiny_cnn (10 experiments completed over 2 days)
- 📋 **Available:** VGG, ResNet, MobileNet families (ready for future experiments)

**View Results:** https://dagshub.com/Y-R-A-V-R-5/ResoMap/experiments

---

## 🏗️ VGG Family

### Architecture Overview
VGG models use **stacked convolutional layers** organized into stages, followed by adaptive global average pooling and fully connected layers.

```
Input (variable resolution)
    ↓
Stages (repeating Conv+ReLU blocks)
    ├─ Stage 1: 3→64 channels
    ├─ Stage 2: 64→128 channels
    ├─ Stage 3: 128→256 channels
    ├─ Stage 4: 256→512 channels
    └─ Stage 5: 512→512 channels
    ↓
AdaptiveAvgPool2d (7×7) ← Handles any resolution!
    ↓
Classifier (FC layers + Dropout)
    ↓
Output (num_classes)
```

### Implementation Details

**Class:** `VGG(nn.Module)` in [src/models.py](src/models.py)

**Key Features:**
- Modular stage-based construction from config
- Adaptive average pooling for any resolution
- Configurable FC layer sizes (default: 4096→4096→num_classes)
- Dropout regularization (default: 0.5)
- ReLU activations throughout

**Variants:**
```python
vgg11:  [1, 1, 2, 2, 2]   # 11 conv layers
vgg13:  [2, 2, 2, 2, 2]   # 13 conv layers
vgg16:  [2, 2, 3, 3, 3]   # 16 conv layers (ImageNet standard)
vgg19:  [2, 2, 3, 4, 4]   # 19 conv layers (deeper)
```

Numbers represent how many Conv layers per stage.

**Parameters:**
- vgg11: ~128M
- vgg13: ~133M
- vgg16: ~138M (most commonly trained)

**Code Reference:**
```python
def forward(self, x):
    # x shape: (batch, 3, H, W) - H and W can be any size
    for stage in self.stages.values():
        x = stage(x)  # Each stage ends with MaxPool
    
    x = self.avgpool(x)  # (batch, 512, 7, 7) for any input
    x = x.view(x.size(0), -1)  # Flatten
    x = self.classifier(x)  # FC layers
    return x
```

**Why Adaptive Pooling Works:**
- After all MaxPool operations, spatial dimensions are reduced
- `AdaptiveAvgPool2d((7,7))` guarantees 7×7 output regardless of input
- This enables fixed FC layer input (512×7×7 = 25,088 features)

**Best For:**
- ✅ Explainability research (simple, interpretable)
- ✅ Understanding layer-wise behavior
- ✅ GPU comparison (baseline architecture)
- ❌ Mobile inference (too many parameters)

---

## 🔗 ResNet Family

### Architecture Overview
ResNet uses **residual connections** (skip connections) to train very deep networks without gradient degradation.

```
Input (variable resolution)
    ↓
Initial Conv (7×7, stride=2)
    ↓
Layer1: Multiple Blocks with residuals
Layer2: Multiple Blocks with residuals
Layer3: Multiple Blocks with residuals
Layer4: Multiple Blocks with residuals
    ↓
AdaptiveAvgPool2d (1×1)
    ↓
FC layer (num_features → num_classes)
    ↓
Output
```

### Block Types

**BasicBlock** (ResNet18, ResNet34):
```
Input
  ↓
Conv 3×3 → BatchNorm → ReLU
  ↓
Conv 3×3 → BatchNorm
  ↓
Add with skip connection
  ↓
ReLU
  ↓
Output
```

**Bottleneck** (ResNet50, ResNet101):
```
Input
  ↓
Conv 1×1 (reduce)
  ↓
Conv 3×3 (main)
  ↓
Conv 1×1 (expand)
  ↓
BatchNorm → Add with skip → ReLU
  ↓
Output
```

### Implementation Details

**Class:** `ResNet(nn.Module)` in [src/models.py](src/models.py)

**Key Features:**
- Block repetition counts configurable per layer
- Bottleneck blocks for depth (ResNet50+)
- Batch normalization throughout
- Identity skip connections (straight path)
- Projected skip connections when spatial dims change
- Stride-2 in first block of layer 2-4 (downsampling)

**Variants:**
```python
resnet18:  [2, 2, 2, 2] blocks per layer + BasicBlock
resnet34:  [3, 4, 6, 3] blocks per layer + BasicBlock
resnet50:  [3, 4, 6, 3] blocks per layer + Bottleneck
resnet101: [3, 4, 23, 3] blocks per layer + Bottleneck
```

**Parameters:**
- resnet18: ~11M (lightweight!)
- resnet34: ~21M
- resnet50: ~25M
- resnet101: ~44M

**Code Reference:**
```python
class Bottleneck(nn.Module):
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)  # 1×1 reduce
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)  # 3×3 main
        out = self.bn2(out)
        out = F.relu(out)
        
        out = self.conv3(out)  # 1×1 expand
        out = self.bn3(out)
        
        out += identity  # Skip connection!
        out = F.relu(out)
        return out
```

**Why Skip Connections Matter:**
- Gradients can flow directly through the skip path
- Allows training very deep networks (101+ layers)
- Each block learns residual (difference) not absolute mapping
- Identity initialization: early training benefits from identity path

**Best For:**
- ✅ Accuracy vs depth analysis
- ✅ Computational efficiency comparison
- ✅ Transfer learning (great pre-trained models available)
- ✅ Mobile deployment (resnet18 is very efficient)
- ✅ Balanced accuracy/speed tradeoff

---

## 📱 MobileNet Family

### Architecture Overview
MobileNet uses **depthwise separable convolutions** to achieve high accuracy with minimal parameters - designed for mobile/edge devices.

```
Input (variable resolution)
    ↓
Conv 3×3 (32 filters)
    ↓
MobileBlock 1: Depthwise + Pointwise (expansion=1)
MobileBlock 2: Depthwise + Pointwise (expansion=6)
... (multiple blocks with different configs)
    ↓
AdaptiveAvgPool (1×1)
    ↓
FC (1000 → num_classes)
    ↓
Output
```

### Depthwise Separable Convolution

**Standard Convolution:**
- Input: (batch, in_channels, H, W)
- Kernel: (out_channels, in_channels, 3, 3)
- Computation: in_channels × H × W × out_channels × 9 operations

**Depthwise Separable:**
1. **Depthwise:** (in_channels, 1, 3, 3) - one filter per channel
2. **Pointwise:** (out_channels, in_channels, 1, 1) - cross-channel mixing

**Benefit:** ~8-9x fewer operations!

### MobileNetV2: Inverted Residual

```
Input (expansion=6 for middle blocks)
  ↓
1×1 Conv (expand by 6x)
  ↓
Depthwise Conv 3×3 (ReLU6)
  ↓
1×1 Conv (project back)
  ↓
Skip connection (only if stride=1)
  ↓
Output
```

Why "inverted"? Traditional ResNet: wide→narrow→wide. MobileNet: narrow→wide→narrow.

### Implementation Details

**Class:** `MobileNetV2(nn.Module)` in [src/models.py](src/models.py)

**Key Features:**
- Configurable expansion factor (default=6)
- Width multiplier (default=1.0, can reduce to 0.75 for smaller models)
- ReLU6 activations
- Batch normalization throughout
- Stride control for spatial downsampling

**MobileNetV3 Additions:**
- Squeeze-and-Excitation (SE) blocks for channel attention
- Hard Swish activation (more efficient)
- More efficient block design

**Parameters:**
- mobilenet_v2 (width=1.0): ~3.5M
- mobilenet_v2_small (width=0.75): ~2.2M
- mobilenet_v3_small: ~2.5M
- mobilenet_v3_large: ~5.4M

**Width Multiplier Effect:**
```
width=1.0:    all_channels × 1.0  → full model
width=0.75:   all_channels × 0.75 → 50% parameters
width=0.5:    all_channels × 0.5  → 25% parameters
```

**Best For:**
- ✅ Mobile/edge device deployment
- ✅ Efficiency vs accuracy analysis
- ✅ Finding smallest model for target accuracy
- ✅ Latency-critical applications
- ✅ Memory-constrained scenarios

---

## 🎯 Custom CNNs

### SimpleCNN

Minimal architecture for quick experimentation and debugging:

```
Input (variable resolution)
  ↓
Conv 3×3 (3→32) + ReLU + MaxPool 2×2
  ↓
Conv 3×3 (32→64) + ReLU + MaxPool 2×2
  ↓
Conv 3×3 (64→128) + ReLU + MaxPool 2×2
  ↓
AdaptiveAvgPool (4×4)
  ↓
FC (128×16 → 128) + ReLU + Dropout
  ↓
FC (128 → num_classes)
  ↓
Output
```

**Parameters:** <1M
**Use Cases:**
- ✅ Quick debugging
- ✅ Testing pipeline functionality
- ✅ Small dataset experiments

### TinyCNN

Even smaller baseline:
```
Input
  ↓
Conv 3×3 (3→16) + ReLU
  ↓
MaxPool 2×2
  ↓
Conv 3×3 (16→32) + ReLU
  ↓
AdaptiveAvgPool (2×2)
  ↓
FC (32×4 → 10)
  ↓
Output
```

**Parameters:** <0.5M

---

## 🔄 Building Models from Config

### Method 1: From config.yaml

```python
from src.models import build_model
from src.utils import load_config

config = load_config('configs/models.yaml')
model_cfg = config['models']['vgg11']

model = build_model(model_cfg, num_classes=7)
# Returns: VGG model for skin lesion classification (7 classes)
```

### Method 2: Using load_model_from_config

```python
from src.models import load_model_from_config

model = load_model_from_config('resnet50', num_classes=10)
# Automatically loads architecture from configs/models.yaml
```

### Method 3: Direct initialization

```python
from src.models import ResNet, Bottleneck

model = ResNet(
    block=Bottleneck,
    block_counts=[3, 4, 6, 3],  # ResNet50 config
    num_classes=7
)
```

---

## 📊 Model Comparison Table

| Model | Params | Size | Latency | Accuracy* | Explainability | Use Case |
|-------|--------|------|---------|-----------|---|---|
| simple_cnn | <1M | <5MB | ⚡⚡⚡⚡⚡ | ⭐⭐ | ⭐⭐⭐⭐⭐ | Debugging |
| tiny_cnn | <0.5M | <2MB | ⚡⚡⚡⚡⚡ | ⭐⭐ | ⭐⭐⭐⭐⭐ | Baseline |
| mobilenet_v2_small | 2.2M | 10MB | ⚡⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐ | Mobile |
| mobilenet_v2 | 3.5M | 14MB | ⚡⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐ | Mobile |
| resnet18 | 11M | 45MB | ⚡⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Balanced |
| resnet50 | 25M | 100MB | ⚡⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Production |
| vgg16 | 138M | 500MB | ⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Research |

*Approximate accuracy on ImageNet (100 class)

---

## 🛠️ Advanced Features

### Adaptive Pooling for Variable Resolutions

All models use `AdaptiveAvgPool2d()` or `AdaptiveMaxPool2d()` instead of fixed pooling:

```python
# Standard approach (fixed size input)
self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
# Works only for 224×224, breaks for other sizes

# Adaptive approach (any size input)
self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
# Works for 64×64, 224×224, 512×512, anything!
```

**How it works:**
- Calculates stride/kernel dynamically: stride = input_size / output_size
- For 224×224 input → stride ≈ 32, kernel ≈ 32
- For 512×512 input → stride ≈ 73, kernel ≈ 73
- Always outputs exactly 7×7 (or specified size)

### Weight Initialization

All models use proper weight initialization:

```python
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)
```

- Kaiming (He) for convolutional layers
- Normal distribution for FC layers
- BatchNorm initialized to identity

---

## 📈 Architecture Selection Guide

**Choose VGG if:**
- Studying layer-wise behavior
- Need maximum interpretability
- Have sufficient GPU memory
- Analyzing Grad-CAM visualizations

**Choose ResNet if:**
- Want best accuracy/parameter tradeoff
- Training on limited GPU memory
- Need transfer learning models
- Production deployment planned

**Choose MobileNet if:**
- Deploying to mobile/edge devices
- Optimizing for inference speed
- Memory is critical constraint
- Need real-time performance

**Choose Custom CNNs if:**
- Debugging the pipeline
- Quick experimentation
- Establishing baseline
- Research into architecture basics

---

**Next:** [TRAINING_EXECUTION.md](TRAINING_EXECUTION.md) - How to train these models  
**Back:** [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Project overview
