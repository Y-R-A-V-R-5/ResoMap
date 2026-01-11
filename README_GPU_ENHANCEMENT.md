# ResoMap: GPU-Enhanced Resolution & Explainability Study

## Overview

ResoMap has been enhanced to leverage GPU capabilities for comprehensive analysis of how image resolutions affect model performance and decision-making. The project now includes:

- **GPU Acceleration**: Mixed precision training, multi-GPU support, optimized data loading
- **Explainability**: Grad-CAM, Integrated Gradients, Saliency Maps
- **Advanced Profiling**: GPU memory tracking, throughput analysis, FLOPs calculation
- **Resolution Analysis**: Compare model behavior across multiple resolutions

## 🚀 Key Enhancements

### 1. GPU Optimization
- **Mixed Precision Training (AMP)**: 2x faster training with lower memory usage
- **Gradient Accumulation**: Train with larger effective batch sizes
- **Multi-GPU Support**: DataParallel for multiple GPUs
- **Optimized Data Loading**: Pin memory, prefetching, persistent workers
- **cuDNN Benchmarking**: Automatic algorithm selection

### 2. Explainability Features
- **Grad-CAM**: Visualize important image regions for predictions
- **Integrated Gradients**: Attribution-based feature importance
- **Saliency Maps**: Fast gradient-based visualizations
- **Resolution Comparison**: Side-by-side explanation analysis across resolutions
- **Consistency Metrics**: Quantify explanation stability

### 3. Advanced Training
- **Model Checkpointing**: Save top-k best models automatically
- **Learning Rate Scheduling**: Cosine, Step, Plateau schedulers
- **Gradient Clipping**: Prevent exploding gradients
- **Performance Tracking**: Epoch times, GPU memory, throughput

### 4. Comprehensive Profiling
- **GPU Memory Tracking**: Allocated, reserved, peak memory
- **CUDA Event Timing**: Precise GPU timing measurements
- **Throughput Analysis**: Samples per second
- **Model Complexity**: FLOPs and parameter counts

## 📦 Installation

### Step 1: Install PyTorch with CUDA Support

```bash
# For CUDA 11.8 (check your CUDA version first)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Install Other Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify GPU Setup

```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"GPU Device: {torch.cuda.get_device_name(0)}")
```

## 🔧 Configuration

ResoMap uses a **modular configuration system** for better organization:

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

### Key Settings

**GPU Settings** (`configs/system.yaml`):
```yaml
device: "cuda"                    # Use GPU
use_cpu_fallback: true           # Fallback to CPU if needed
multi_gpu: false                 # Enable multi-GPU
use_mixed_precision: true        # Enable AMP
gradient_accumulation_steps: 1   # Gradient accumulation
pin_memory: true                 # Pin memory for GPU
cudnn_benchmark: true            # cuDNN optimization
```

### Resolution Sweep (`configs/sweep.yaml`)
```yaml
models:
  - "vgg11"
  - "vgg13"
  - "resnet18"
  - "mobilenet_v2"
  # ... more models
  
resolutions: [224, 256, 320, 384, 512]
```

### Training Configuration (`configs/training.yaml`)
```yaml
batch_size: 64                   # Larger batch for GPU
epochs: 50

scheduler:
  enabled: true
  type: "cosine"
  
checkpointing:
  enabled: true
  save_top_k: 3
  monitor: "val_accuracy"
```

### Explainability Settings (`configs/explainability.yaml`)
```yaml
enabled: true
  methods:
    - "gradcam"
    - "integrated_gradients"
    - "saliency"
  
  num_samples: 20
  save_visualizations: true
  compare_resolutions: true
```

## 🎯 Usage

### Training with GPU Optimization

```python
import torch
from src.trainer import Trainer
from src.models import build_model
from src.data import get_data_loaders
from src.callbacks import EarlyStopping, ModelCheckpoint
from torch.cuda.amp import GradScaler

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_model(model_config)

# Data loaders with GPU optimization
train_loader, val_loader, test_loader = get_data_loaders(
    data_dir='data/',
    resolution=224,
    batch_size=64,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True
)

# Trainer with GPU features
trainer = Trainer(
    model=model,
    device=device,
    criterion=torch.nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    train_loader=train_loader,
    val_loader=val_loader,
    use_amp=True,                    # Mixed precision
    gradient_accumulation_steps=2,   # Accumulation
    gradient_clip_norm=1.0,          # Gradient clipping
    scheduler=scheduler
)

# Callbacks
early_stopping = EarlyStopping(patience=10, mode='min')
checkpoint = ModelCheckpoint(
    save_dir='checkpoints/',
    monitor='val_accuracy',
    mode='max',
    save_top_k=3
)

# Training loop
for epoch in range(num_epochs):
    train_metrics = trainer.train_epoch()
    val_metrics = trainer.validate()
    
    # Callbacks
    checkpoint.step(epoch, val_metrics, model, trainer.optimizer, resolution=224)
    early_stopping.step(epoch, val_metrics['loss'])
    
    if early_stopping.stop:
        break
```

### Generating Explainability Visualizations

```python
from src.explainability import ModelExplainer, get_target_layer

# Setup explainer
target_layer = get_target_layer(model, "VGG11")
explainer = ModelExplainer(
    model=model,
    device=device,
    target_layer=target_layer,
    class_names=['class1', 'class2', ...]
)

# Generate explanations
explanations = explainer.explain_batch(
    images=test_images,
    labels=test_labels,
    method='gradcam'
)

# Visualize single explanation
explainer.visualize_explanation(
    image=test_images[0],
    explanation=explanations[0],
    pred_class=predictions[0],
    true_class=test_labels[0],
    save_path='explanations/sample_001.png'
)
```

### Resolution Comparison Analysis

```python
# Compare explanations across resolutions
images_dict = {
    224: images_224,
    384: images_384,
    512: images_512
}

explainer.compare_resolutions(
    images_dict=images_dict,
    labels=test_labels,
    method='gradcam',
    num_samples=5,
    save_dir='resolution_comparison/'
)
```

### Comprehensive Analysis Script

```bash
# Run full resolution-explainability analysis
python scripts/resolution_explainability_analysis.py \
    --checkpoint-dir checkpoints/ \
    --output-dir analysis_results/ \
    --models vgg11_224 vgg13_224 \
    --resolutions 224 256 320 384 512
```

This will:
1. Load trained models at each resolution
2. Evaluate performance metrics
3. Profile inference speed and memory
4. Generate explainability visualizations
5. Compare explanations across resolutions
6. Create comprehensive reports and plots

## 📊 GPU Profiling

### Profile Model Performance

```python
from src.profiler import profile_model, profile_gpu_memory, get_model_complexity

# Profile inference
results = profile_model(
    model=model,
    device=device,
    loader=test_loader,
    warmup=5,
    runs=25,
    track_activation_memory=True
)

print(f"Avg Inference Time: {results['avg_time_sec']:.4f}s")
print(f"Throughput: {results['throughput_samples_sec']:.2f} samples/s")
print(f"GPU Memory: {results['gpu_memory_peak_mb']:.2f} MB")

# Detailed GPU memory profiling
gpu_mem = profile_gpu_memory(
    model=model,
    device=device,
    input_shape=(1, 3, 224, 224),
    track_activations=True
)

print(f"Model Parameters: {gpu_mem['model_parameters_mb']:.2f} MB")
print(f"Activation Memory: {gpu_mem['activation_memory_mb']:.2f} MB")

# Model complexity
complexity = get_model_complexity(
    model=model,
    input_shape=(1, 3, 224, 224)
)

print(f"Total Parameters: {complexity['params_millions']:.2f}M")
print(f"GFLOPs: {complexity['gflops']:.2f}")
```

## 🎨 Visualization Examples

The analysis script generates:

1. **Performance Plots**: Accuracy, Precision, Recall, F1 vs Resolution
2. **Profiling Plots**: Inference time, Throughput, Memory vs Resolution
3. **Explainability Visualizations**: 
   - Original images
   - Heatmaps
   - Overlays
   - Side-by-side comparisons across resolutions

## 📁 Project Structure

```
ResoMap/
├── configs/
│   ├── config.yaml          # Main configuration
│   ├── sweep.yaml           # Experiment grid
│   ├── training.yaml        # Training settings
│   ├── system.yaml          # GPU/system configuration
│   ├── data.yaml            # Dataset and augmentation
│   ├── explainability.yaml  # Interpretation methods
│   ├── mlflow.yaml          # Experiment tracking
│   └── models.yaml          # Model architectures
├── scripts/
│   ├── data.py
│   ├── experiments.py
│   └── analysis.py          # Analysis script
├── src/
│   ├── callbacks.py         # Enhanced with ModelCheckpoint
│   ├── data.py             # GPU-optimized data loading
│   ├── explainability.py   # NEW: Explainability module
│   ├── models.py
│   ├── profiler.py         # Enhanced GPU profiling
│   ├── trainer.py          # GPU-optimized training
│   └── utils.py
├── requirments.txt         # Updated with GPU & explainability packages
└── README_GPU_ENHANCEMENT.md  # This file
```

## 🔬 Research Questions You Can Answer

1. **How does resolution affect accuracy?**
   - Train models at different resolutions
   - Compare performance metrics

2. **What's the performance-efficiency tradeoff?**
   - Higher resolution = better accuracy?
   - Cost: inference time, memory usage

3. **Do models focus on different features at different resolutions?**
   - Use Grad-CAM to visualize attention
   - Compare explanation consistency

4. **What's the optimal resolution for your use case?**
   - Balance accuracy, speed, and memory
   - Consider deployment constraints

## 🚨 GPU Memory Management Tips

1. **Use Mixed Precision**: Saves ~40% memory
2. **Gradient Accumulation**: Simulate larger batches
3. **Monitor Memory**: Track with `nvidia-smi` or profiler
4. **Clear Cache**: `torch.cuda.empty_cache()` between runs
5. **Reduce Batch Size**: If OOM errors occur

## 📈 Expected Performance Gains

With GPU acceleration:
- **Training Speed**: 10-50x faster than CPU (depending on GPU)
- **Mixed Precision**: 2x speedup, 40% memory savings
- **Batch Size**: 4-8x larger batches possible
- **Resolution Range**: Can experiment with higher resolutions (512+)

## 🐛 Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size
batch_size = 32  # or 16

# Enable gradient accumulation
gradient_accumulation_steps = 2

# Use mixed precision
use_amp = True
```

### Slow Data Loading
```python
# Increase workers
num_workers = 8

# Enable persistent workers
persistent_workers = True

# Pin memory
pin_memory = True
```

### No GPU Detected
```bash
# Check CUDA installation
nvidia-smi

# Reinstall PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📚 Additional Resources

- [PyTorch Mixed Precision](https://pytorch.org/docs/stable/amp.html)
- [Captum Documentation](https://captum.ai/)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)
- [Integrated Gradients Paper](https://arxiv.org/abs/1703.01365)

## 🎓 Citation

If you use this enhanced ResoMap in your research, please cite:

```bibtex
@software{resomap_gpu,
  title={ResoMap: GPU-Enhanced Resolution and Explainability Analysis},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/ResoMap}
}
```

---

**Happy Experimenting! 🚀**