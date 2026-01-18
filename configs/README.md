# ResoMap Configuration Guide

## Configuration Structure

ResoMap uses a modular configuration system with separate YAML files for different aspects:

```
configs/
├── config.yaml           # Main config (project metadata)
├── sweep.yaml           # Experiment grid (models, resolutions)
├── training.yaml        # Training hyperparameters
├── system.yaml          # GPU/system settings
├── data.yaml            # Dataset and augmentation
├── explainability.yaml  # Interpretation methods
├── mlflow.yaml          # Experiment tracking
└── models.yaml          # Model architectures
```

## Configuration Files

### 1. `config.yaml` - Main Configuration
Project name and description. Other configs are loaded automatically.

### 2. `sweep.yaml` - Experiment Grid
Defines which models and resolutions to test:
```yaml
models:
  - "vgg11"
  - "resnet18"
  - "mobilenet_v2"
  # ... more models
resolutions: [224, 256, 320, 384, 512]
```

### 3. `training.yaml` - Training Settings
Hyperparameters for training:
- Batch size, epochs, learning rate
- Optimizer, scheduler, weight decay
- Early stopping, checkpointing

### 4. `system.yaml` - System Configuration
Hardware and performance settings:
- GPU device selection
- Mixed precision training
- Multi-GPU support
- Memory and profiling options

### 5. `data.yaml` - Data Configuration
Dataset and augmentation:
- Dataset paths
- DataLoader settings (workers, prefetch)
- Augmentation strategies (flip, rotation, color jitter)
- Normalization parameters

### 6. `explainability.yaml` - Explainability Settings
Model interpretation configuration:
- Explainability methods (Grad-CAM, Integrated Gradients, Saliency)
- Number of samples to analyze
- Visualization settings

### 7. `mlflow.yaml` - MLflow Tracking
Experiment tracking configuration:
- MLflow tracking URI
- Experiment name

### 8. `models.yaml` - Model Architectures
Detailed model definitions organized by family:
- **VGG Family**: vgg11, vgg13, vgg16
- **ResNet Family**: resnet18, resnet34, resnet50, resnet101
- **MobileNet Family**: mobilenet_v2, mobilenet_v2_small, mobilenet_v3_small, mobilenet_v3_large
- **EfficientNet Family**: efficientnet_b0, efficientnet_b1, efficientnet_b2 (not yet implemented)
- **Custom CNNs**: simple_cnn, tiny_cnn

## Usage

### Loading Configuration

Configs are automatically loaded when you use `load_config()`:

```python
from src.utils import load_config
from pathlib import Path

# Load all configs automatically
config = load_config(Path("configs/config.yaml"))

# Access specific configs
models = config['sweep']['models']
batch_size = config['training']['batch_size']
device = config['system']['device']
```

### Modifying Configuration

To change experiment settings, edit the relevant YAML file:

**Example: Change batch size**
```yaml
# configs/training.yaml
batch_size: 128  # Changed from 64
```

**Example: Add more models**
```yaml
# configs/sweep.yaml
models:
  - "vgg11"
  - "resnet18"
  - "mobilenet_v2"
  - "my_new_model"  # Add here
```

**Example: Change resolutions**
```yaml
# configs/sweep.yaml
resolutions: [224, 384, 512]  # Test only 3 resolutions
```

## Backward Compatibility

The system supports both modular and legacy flat config structures. If modular config files are not found, it falls back to sections within `config.yaml`.

## Benefits of Modular Configuration

1. **Organization**: Related settings grouped together
2. **Maintainability**: Easy to find and update specific settings
3. **Reusability**: Share configs across different experiments
4. **Clarity**: Smaller files are easier to understand
5. **Version Control**: Cleaner git diffs when changing settings