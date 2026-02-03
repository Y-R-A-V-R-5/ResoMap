# Setup & Installation Guide

## 🚀 Quick Start (5 minutes)

### 1. Install PyTorch with GPU Support

First, check your CUDA version:
```bash
nvidia-smi
```

Look at the top line for "CUDA Version X.X"

**For CUDA 11.8:**
```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 12.4:**
```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu124
```

**For CPU-only (fallback if no GPU):**
```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
```

### 2. Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```python
import torch
import torchvision

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("⚠️ No CUDA support - will use CPU (slower!)")

print("\n✅ Installation successful!")
```

---

## 🔧 Configuration for Your GPU

Edit `configs/training.yaml` and `configs/system.yaml` based on your GPU:

### GPU with 4-6 GB VRAM
Examples: GTX 1650, RTX 3050, RTX 3060 (mobile), Tesla T4

**configs/training.yaml:**
```yaml
batch_size: 32  # Smaller batches
epochs: 50      # Reduce if memory still tight
```

**configs/sweep.yaml:**
```yaml
resolutions: [224, 256, 320]  # Avoid 384, 512
```

**configs/system.yaml:**
```yaml
device: "cuda"
use_mixed_precision: true  # ESSENTIAL for 4-6GB GPUs!
gradient_accumulation_steps: 2
pin_memory: true
num_workers: 2  # Fewer workers to save memory
```

### GPU with 8-10 GB VRAM
Examples: RTX 3060 (desktop), RTX 3070, RTX 3080, RTX 4060 Ti

**configs/training.yaml:**
```yaml
batch_size: 64
epochs: 50
```

**configs/sweep.yaml:**
```yaml
resolutions: [224, 256, 320, 384]
```

**configs/system.yaml:**
```yaml
device: "cuda"
use_mixed_precision: true
gradient_accumulation_steps: 1
pin_memory: true
num_workers: 4
```

### GPU with 12+ GB VRAM
Examples: RTX 3090, RTX 4090, A100, L40

**configs/training.yaml:**
```yaml
batch_size: 128
epochs: 50
```

**configs/sweep.yaml:**
```yaml
resolutions: [224, 256, 320, 384, 512]  # All resolutions!
```

**configs/system.yaml:**
```yaml
device: "cuda"
use_mixed_precision: true  # Still recommended for speed
gradient_accumulation_steps: 1
pin_memory: true
num_workers: 8
cudnn_benchmark: true
```

### Multi-GPU Setup (2+ GPUs)

**configs/system.yaml:**
```yaml
device: "cuda"
multi_gpu: true
use_mixed_precision: true
gradient_accumulation_steps: 1
pin_memory: true
num_workers: 8
cudnn_benchmark: true
```

Training will automatically use all available GPUs with DataParallel.

### CPU-Only Setup (Fallback)

**configs/system.yaml:**
```yaml
device: "cpu"
use_mixed_precision: false  # Not supported on CPU
num_workers: 4  # Use CPU cores
```

**configs/training.yaml:**
```yaml
batch_size: 16  # Very small
epochs: 50
```

**configs/sweep.yaml:**
```yaml
resolutions: [64, 128, 224]  # Only small resolutions
# Higher resolutions will be very slow on CPU
```

---

## 📋 Configuration File Reference

### System Configuration (configs/system.yaml)

```yaml
device: "cuda"                  # "cuda" or "cpu"
num_workers: 4                  # DataLoader workers

# GPU Optimizations
use_mixed_precision: true       # AMP for faster training
multi_gpu: false                # Use DataParallel if true
gradient_accumulation_steps: 1  # Effective batch = batch_size × this
gradient_clip_norm: 1.0         # Gradient clipping (None to disable)
pin_memory: true                # Faster GPU transfers
cudnn_benchmark: true           # cuDNN optimization

# Scheduler
use_scheduler: true
scheduler_type: "cosine"        # "cosine", "step", "plateau"
scheduler_params:
  t_max: 100                    # For cosine
  eta_min: 1.0e-6
```

### Training Configuration (configs/training.yaml)

```yaml
batch_size: 64                  # Adjust for your GPU memory
epochs: 50                      # Training epochs

learning_rate: 0.001            # Initial LR
weight_decay: 1.0e-4            # L2 regularization
momentum: 0.9                   # For SGD

optimizer: "adam"               # "adam" or "sgd"
loss_function: "cross_entropy"

early_stopping:
  enabled: true
  patience: 10                  # Stop if no improvement for 10 epochs
  min_delta: 0.001              # Minimum improvement threshold

checkpointing:
  enabled: true
  save_dir: "checkpoints"
  save_top_k: 3                 # Keep 3 best models
  monitor: "val_loss"           # Which metric to monitor
```

### Data Configuration (configs/data.yaml)

```yaml
dataset_path: "data"
train_split: 0.7
val_split: 0.15
test_split: 0.15

augmentation:
  enable_augmentation: true
  random_flip: true
  random_rotation: 10
  color_jitter: 0.2
  random_crop: true
  
normalization:
  mean: [0.485, 0.456, 0.406]   # ImageNet mean
  std: [0.229, 0.224, 0.225]    # ImageNet std
```

### Models Configuration (configs/models.yaml)

```yaml
models:
  vgg11:
    stages:
      - {stage: "stage1", channels: [3, 64], num_layers: 1}
      - {stage: "stage2", channels: [64, 128], num_layers: 1}
      # ... more stages
    fc_layers: [4096, 4096]
    dropout: 0.5
    
  resnet50:
    block: "bottleneck"
    block_counts: [3, 4, 6, 3]
    channels: [64, 128, 256, 512]
```

---

## 🔍 Verify Everything Works

### Test 1: Check PyTorch and CUDA
```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')
"
```

### Test 2: Load a Model
```bash
python -c "
from src.models import load_model_from_config
model = load_model_from_config('vgg11', num_classes=7)
print(f'Model loaded: {model.__class__.__name__}')
print(f'Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M')
"
```

### Test 3: Load Data
```bash
python -c "
from src.data import get_data_loaders
loaders = get_data_loaders('data', resolution=224, batch_size=32)
train_loader, val_loader, test_loader = loaders
print(f'Loaded {len(train_loader)} training batches')
"
```

### Test 4: Run a Quick Training Test
```bash
python scripts/experiments.py --models simple_cnn --resolutions 64
```

Expected: Completes in ~2-5 minutes with simple_cnn at 64×64 resolution.

---

## ⚠️ Troubleshooting

### Problem: "CUDA out of memory"

**Solutions:**
1. Reduce batch size: `batch_size: 32` in configs/training.yaml
2. Reduce resolution: Remove 512 from configs/sweep.yaml
3. Enable mixed precision: `use_mixed_precision: true` in configs/system.yaml
4. Reduce num_workers: `num_workers: 2` in configs/system.yaml
5. Clear GPU cache between runs

```bash
python -c "
import torch
torch.cuda.empty_cache()
print('GPU cache cleared')
"
```

### Problem: "ModuleNotFoundError: No module named 'torch'"

**Solution:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# Use correct CUDA version from nvidia-smi
```

### Problem: "No CUDA detected but should be available"

**Check:**
```bash
# 1. Is driver installed?
nvidia-smi

# 2. Is PyTorch CUDA version correct?
python -c "import torch; print(torch.version.cuda)"

# 3. Do they match?
nvidia-smi  # Look at "CUDA Version"
```

If mismatched, reinstall PyTorch with correct version.

### Problem: "Dataset not found"

**Solution:**
```bash
# Check structure
ls data/train/
# Should show: akiec/ bcc/ bkl/ df/ mel/ nv/ vasc/

# If missing, prepare dataset
python scripts/data.py  # If data prep script exists
```

### Problem: "Config file not found"

**Solution:**
```bash
# Check configs exist
ls configs/
# Should have: config.yaml, sweep.yaml, training.yaml, etc.

# If missing, create from defaults
# (See configs/README.md)
```

### Problem: "mlflow.tracking.MlflowClient error"

**This is not critical for local training**

If you don't have DagsHub setup:
```bash
python scripts/experiments.py --skip-dagshub-check
```

To setup MLflow locally:
```bash
pip install mlflow
mlflow ui  # Start local tracking server on localhost:5000
```

---

## 🌐 Optional: DagsHub Setup

### Why DagsHub?
- Track experiments across machines
- Resume completed experiments
- Collaborative experiment management

### Setup (5 minutes)

1. Create free account: https://dagshub.com
2. Create new repository
3. Set environment variables:

```bash
# Windows CMD
set MLFLOW_TRACKING_URI=https://dagshub.com/USERNAME/REPO.mlflow
set MLFLOW_TRACKING_USERNAME=USERNAME
set MLFLOW_TRACKING_PASSWORD=YOUR_TOKEN

# Windows PowerShell
$env:MLFLOW_TRACKING_URI="https://dagshub.com/USERNAME/REPO.mlflow"
$env:MLFLOW_TRACKING_USERNAME="USERNAME"
$env:MLFLOW_TRACKING_PASSWORD="YOUR_TOKEN"

# Linux/Mac
export MLFLOW_TRACKING_URI=https://dagshub.com/USERNAME/REPO.mlflow
export MLFLOW_TRACKING_USERNAME=USERNAME
export MLFLOW_TRACKING_PASSWORD=YOUR_TOKEN
```

4. Run experiments:
```bash
python scripts/experiments.py  # Auto-detects DagsHub
```

Completed experiments will be tracked and auto-skipped on future runs.

---

## 📚 Next Steps

1. **Setup Complete?** → Go to [TRAINING_EXECUTION.md](TRAINING_EXECUTION.md)
2. **Want to understand models?** → See [MODELS_METHODS.md](MODELS_METHODS.md)
3. **Ready to train?** → Run `python scripts/experiments.py`

---

## 🆘 Still Having Issues?

Check the relevant section:
- GPU memory → "CUDA out of memory" solution
- Import errors → "ModuleNotFoundError" solution
- Dataset issues → "Dataset not found" solution
- Configuration → See configs/README.md

**Back:** [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Project overview
