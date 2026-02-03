# ResoMap: GPU-Enhanced Resolution & Explainability Study

## 🎯 Project Overview

**ResoMap** is a comprehensive framework for analyzing how image resolutions affect CNN model performance and computational efficiency. The project combines GPU acceleration and multiple model architectures to enable large-scale resolution sensitivity analysis.

### Key Features

✅ **GPU Acceleration**
- Mixed precision training (AMP) for 2x faster training
- Multi-GPU support with DataParallel
- Gradient accumulation and clipping
- cuDNN benchmarking

✅ **Flexible Architecture**
- Multiple pre-built model families (VGG, ResNet, MobileNet, Custom CNNs)
- Variable resolution support (224-512px)
- Resolution-aware augmentation strategies
- Adaptive pooling for any input size

✅ **Advanced Training**
- Selective model/resolution execution via CLI
- Automatic checkpoint management
- Resume from checkpoint on failure
- MLflow tracking and DagsHub integration

✅ **Comprehensive Evaluation**
- Automatic JSON result export per experiment
- CSV aggregation of all results
- Summary statistics generation
- Detailed performance reporting

✅ **Performance Profiling**
- GPU memory tracking (allocated, reserved, peak)
- CUDA event timing for accurate measurements
- Throughput analysis (samples/second)
- FLOPs and parameter complexity analysis

---

## 📁 Project Structure

```
ResoMap/
├── configs/                          # Modular YAML configuration
│   ├── config.yaml                  # Main project config
│   ├── sweep.yaml                   # Model sweep grid (8 models, 5 resolutions)
│   ├── training.yaml                # Hyperparameters
│   ├── system.yaml                  # GPU/device settings
│   ├── data.yaml                    # Dataset & augmentation
│   ├── explainability.yaml          # Interpretation methods
│   ├── mlflow.yaml                  # Experiment tracking
│   ├── models.yaml                  # Model architecture configs
│   └── README.md                    # Configuration guide
│
├── src/                             # Core implementation modules
│   ├── models.py                    # VGG, ResNet, MobileNet architectures
│   ├── trainer.py                   # GPU training loop with AMP
│   ├── experiment.py                # Experiment orchestration
│   ├── sweep.py                     # Multi-model/resolution sweep
│   ├── data.py                      # GPU-optimized data loading
│   ├── explainability.py            # Grad-CAM, IG, Saliency
│   ├── profiler.py                  # GPU memory & timing profiling
│   ├── results.py                   # Results aggregation & export
│   ├── callbacks.py                 # Training callbacks
│   └── utils.py                     # Utility functions
│
├── scripts/                         # Executable scripts
│   ├── experiments.py               # Main training entry point (with CLI args)
│   ├── aggregate_results.py         # Results aggregation script
│   ├── analysis.py                  # Analysis utilities
│   └── data.py                      # Data preparation
│
├── checkpoints/                     # Saved model checkpoints
│   └── {family}/{model}/{resolution}/
│       ├── best_model.pt            # Best validation checkpoint
│       └── final_model.pt           # Final training checkpoint
│
├── results/                         # Experiment results
│   ├── test_results/                # Individual JSON results (auto-saved)
│   ├── all_results.csv              # Aggregated CSV (manual generation)
│   ├── results_summary.json         # Summary statistics (manual generation)
│   └── detailed_report.txt          # Human-readable report (manual generation)
│
├── summary/                         # Model architecture summaries
├── data/                            # Dataset (train/val/test splits)
├── analysis/                        # Dataset analysis outputs
│
├── PROJECT_SUMMARY.md               # This file - overview & structure
├── MODELS_METHODS.md                # Model architectures & implementation
├── SETUP_INSTALLATION.md            # Installation & GPU setup guide
├── TRAINING_EXECUTION.md            # How to run experiments & resume
├── RESULTS_EVALUATION.md            # Results export & analysis
├── QUICK_REFERENCE.md               # One-liners & common tasks
├── requirements.txt                 # Python dependencies
└── README.md                        # Project readme (if exists)
```

---

## 🔬 Models Trained

This study trained two baseline CNN models across five resolutions over a continuous 2-day period:

**Completed Models:**
- `simple_cnn` - 3-layer baseline CNN
- `tiny_cnn` - Minimal 2-layer CNN

**Tested Resolutions:** 224, 256, 320, 384, 512 pixels

**Total Experiments Completed:** 2 models × 5 resolutions = 10 experiments

**Training Duration:** 2 days continuous (48 hours)

### Other Models Available (Not Trained)

The framework includes additional models that others can experiment with:
- `vgg11`, `vgg13` - Dense convolutional architectures
- `resnet18`, `resnet34` - Skip connection architectures  
- `mobilenet_v2_small`, `mobilenet_v3_small` - Mobile-optimized

These are fully implemented and configurable in `configs/sweep.yaml` for future research.

**Compare Results:** https://dagshub.com/Y-R-A-V-R-5/ResoMap/experiments

---

## 📊 Resolution Analysis

The framework systematically studies model behavior across resolutions:

**Tested Resolutions:** 224, 256, 320, 384, 512 pixels

**Analysis Dimensions:**
1. **Accuracy vs Resolution** - How input size affects classification performance
2. **Speed vs Accuracy Tradeoff** - Inference time vs performance
3. **Memory Requirements** - GPU/CPU memory scaling with resolution

---

## 🎛️ Configuration System

ResoMap uses a **modular YAML configuration** system for maximum flexibility:

- `config.yaml` - Project metadata and file paths
- `sweep.yaml` - Which models and resolutions to test
- `training.yaml` - Hyperparameters (batch size, epochs, learning rate)
- `system.yaml` - GPU/CPU device settings, AMP, multi-GPU
- `data.yaml` - Dataset paths, augmentation strategies
- `mlflow.yaml` - Experiment tracking configuration
- `models.yaml` - Architecture-specific parameters

All configs are automatically loaded and merged in `src/utils.py`.

---

## 🚀 Workflow

### 1. Setup (5 minutes)
```bash
pip install -r requirements.txt
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 2. Train Models (2 days for baseline models completed)
```bash
# Completed: simple_cnn + tiny_cnn × 5 resolutions = 10 experiments
python scripts/experiments.py --models simple_cnn tiny_cnn

# Or train other available models (VGG, ResNet, MobileNet)
python scripts/experiments.py --models vgg11 resnet18 --resolutions 224 320
```

### 3. Generate Results (minutes)
```bash
python scripts/aggregate_results.py
```

### 4. Analyze Results (varies)
```bash
# View report
cat results/detailed_report.txt

# Analyze in Excel
open results/all_results.csv

# Analyze in Python
import pandas as pd
df = pd.read_csv('results/all_results.csv')
df.groupby('model')['test_accuracy'].mean()
```

---

## 📈 Actual Results

### Completed Experiments (simple_cnn and tiny_cnn)

Results from 2-day continuous training run. See full data in [results/all_results.csv](results/all_results.csv).

**simple_cnn Performance:**
- Best Accuracy: 81.69% @ 320px (test_accuracy: 0.8169)
- Fastest Inference: 40.0ms @ 224px
- Peak GPU Memory: 2192 MB @ 512px

**tiny_cnn Performance:**
- Best Accuracy: 67.81% @ 256px (test_accuracy: 0.6781)
- Fastest Inference: 23.6ms @ 224px
- Peak GPU Memory: 1138 MB @ 512px

### Resolution Impact (Observed)

- **Lower resolutions (224px):** Faster inference (~25-40ms), lower memory (~230-460 MB)
- **Higher resolutions (512px):** Slower inference (~67-148ms), higher memory (~1138-2192 MB)
- **Optimal resolution:** 256-320px for simple_cnn (best accuracy/speed tradeoff)

**Compare All Results:** https://dagshub.com/Y-R-A-V-R-5/ResoMap/experiments

---

## 🔧 Key Capabilities

### Resume from Checkpoint
If training fails (e.g., GPU OOM at resolution 320):
```bash
# Automatically loads checkpoint and continues
python scripts/experiments.py --models vgg11 --resolutions 320
```

### DagsHub Integration
Automatically skip already-completed experiments:
```bash
# Checks MLflow/DagsHub for completed runs and skips them
python scripts/experiments.py
```

### Distributed Training
Run different models on different machines:
```bash
# Machine 1
python scripts/experiments.py --models vgg11 vgg16 --skip-dagshub-check

# Machine 2
python scripts/experiments.py --models resnet18 mobilenet_v2 --skip-dagshub-check

# Consolidate with DagsHub check (auto-skips completed)
python scripts/experiments.py
```

---

## 📚 Documentation Structure

| File | Purpose |
|------|---------|
| **PROJECT_SUMMARY.md** | This file - overview, structure, features, quick reference |
| **MODELS_METHODS.md** | Detailed model architectures & implementation |
| **SETUP_INSTALLATION.md** | PyTorch setup, GPU configuration, troubleshooting |
| **TRAINING_EXECUTION.md** | How to run experiments, CLI args, resume guide |
| **RESULTS_EVALUATION.md** | Results export, analysis, metrics explained |
| **ACTUAL_RESULTS.md** | Completed training results (simple_cnn, tiny_cnn) |
| **configs/README.md** | Configuration file guide |

---

## ✅ Validation Checklist

- ✅ All source files present and functional
- ✅ Configuration system working (YAML files)
- ✅ GPU acceleration features implemented (AMP, multi-GPU)
- ✅ Resume capability functional (checkpoint detection & loading)
- ✅ Results export system working (JSON → CSV aggregation)
- ✅ DagsHub integration active (https://dagshub.com/Y-R-A-V-R-5/ResoMap/experiments)
- ✅ CLI argument parsing complete
- ✅ 10 experiments completed (simple_cnn + tiny_cnn × 5 resolutions)

---

## ⚡ Quick Reference

### Common Commands

| Task | Command |
|------|---------|
| Train all experiments | `python scripts/experiments.py` |
| Train specific model | `python scripts/experiments.py --models vgg11` |
| Train specific resolution | `python scripts/experiments.py --resolutions 224` |
| Resume failed run | `python scripts/experiments.py --models vgg11 --resolutions 320` |
| Aggregate results | `python scripts/aggregate_results.py` |
| View results report | `cat results/detailed_report.txt` |

### File Structure Quick Links

- **Source Code:** `src/` - models.py, trainer.py, experiment.py, data.py, etc.
- **Configuration:** `configs/` - All YAML config files
- **Training Script:** `scripts/experiments.py` - Main entry point
- **Results:** `results/test_results/` - Auto-saved JSON files
- **Checkpoints:** `checkpoints/{family}/{model}/{resolution}/`

### Documentation

| File | Purpose |
|------|---------|
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | This file - overview & quick start |
| [MODELS_METHODS.md](MODELS_METHODS.md) | Model architectures & implementation |
| [SETUP_INSTALLATION.md](SETUP_INSTALLATION.md) | Installation & GPU setup |
| [TRAINING_EXECUTION.md](TRAINING_EXECUTION.md) | How to run experiments |
| [RESULTS_EVALUATION.md](RESULTS_EVALUATION.md) | Results analysis |
| [DAGSHUB_COMPARISON_GUIDE.md](DAGSHUB_COMPARISON_GUIDE.md) | Compare results in DagsHub MLflow |
| [ACTUAL_RESULTS.md](ACTUAL_RESULTS.md) | Completed training results |


**Next Steps:** See [SETUP_INSTALLATION.md](SETUP_INSTALLATION.md) for installation or [TRAINING_EXECUTION.md](TRAINING_EXECUTION.md) to start experiments.
