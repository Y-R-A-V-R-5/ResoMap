# Training Execution & Resume Guide

## 🎯 Quick Start Commands

| Task | Command |
|------|---------|
| Run all experiments | `python scripts/experiments.py` |
| Train one model | `python scripts/experiments.py --models simple_cnn` |
| Train one resolution | `python scripts/experiments.py --resolutions 224` |
| Specific combo | `python scripts/experiments.py --models simple_cnn --resolutions 224 320` |
| Resume failed run | `python scripts/experiments.py --models simple_cnn --resolutions 320` |
| Local testing | `python scripts/experiments.py --skip-dagshub-check` |

---

## 📋 Command-Line Arguments

### `scripts/experiments.py`

```bash
python scripts/experiments.py [OPTIONS]
```

**Options:**

```
--models MODEL [MODEL ...]
  Which models to train. Default: all from config
  Example: --models simple_cnn resnet18 mobilenet_v2
  
--resolutions RES [RES ...]
  Which resolutions to train. Default: all from config
  Example: --resolutions 224 320
  
--skip-dagshub-check
  Skip checking DagsHub/MLflow for completed runs
  Use for: local testing, offline work, manual tracking
  Default: Check DagsHub (skip completed runs)

--help
  Show all arguments and exit
```

---

## 🚀 Execution Modes

### Mode 1: Run All Experiments (Default)

```bash
python scripts/experiments.py
```

**What happens:**
1. Checks DagsHub for already-completed (model, resolution) pairs
2. Trains configured models × resolutions from configs/sweep.yaml
3. Skips any already in DagsHub
4. Saves checkpoints after each experiment
5. Logs to MLflow

**Already Completed (2-day training run):**
```
Found 10 completed run(s) in DagsHub:
  ✓ simple_cnn@224
  ✓ simple_cnn@256
  ✓ simple_cnn@320
  ✓ simple_cnn@384
  ✓ simple_cnn@512
  ✓ tiny_cnn@224
  ✓ tiny_cnn@256
  ✓ tiny_cnn@320
  ✓ tiny_cnn@384
  ✓ tiny_cnn@512
```

**View Results:** https://dagshub.com/Y-R-A-V-R-5/ResoMap/experiments

**Duration:** Baseline models (simple_cnn, tiny_cnn) took 2 days continuous for 10 experiments

### Mode 2: Train Specific Models

```bash
python scripts/experiments.py --models simple_cnn tiny_cnn
```

**What happens:**
1. Trains simple_cnn on ALL resolutions (224, 256, 320, 384, 512)
2. Trains tiny_cnn on ALL resolutions
3. Total: 2 models × 5 resolutions = 10 experiments
4. Still checks DagsHub and skips completed

**Completed Training:**
These 10 experiments were completed over 2 days continuous training.

**View Results:** https://dagshub.com/Y-R-A-V-R-5/ResoMap/experiments

**Use cases:**
- Compare baseline CNN architectures
- Benchmark different model complexities
- Focus on subset of models

**Duration:** ~1 hour total

### Mode 3: Train Specific Resolutions

```bash
python scripts/experiments.py --resolutions 224 320
```

**What happens:**
1. Trains ALL models at resolution 224
2. Trains ALL models at resolution 320
3. Total: 8 models × 2 resolutions = 16 experiments

**Use cases:**
- Analyze resolution sensitivity
- Compare low vs high resolution
- Reduce computation time

**Duration:** ~2 hours total

### Mode 4: Combined Selection

```bash
python scripts/experiments.py --models simple_cnn resnet18 mobilenet_v2 --resolutions 224 320 384
```

**What happens:**
1. Trains 3 models × 3 resolutions = 9 experiments

**Use cases:**
- Fine-grained experiment design
- Custom model/resolution matrix
- Targeted comparison

**Duration:** ~45 minutes total

### Mode 5: Resume from Failure

```bash
python scripts/experiments.py --models simple_cnn --resolutions 320
```

**Scenario:**
- simple_cnn failed at resolution 320 (e.g., GPU OOM)
- Other resolutions completed (224, 256, 384, 512)

**What happens:**
1. Detects existing checkpoint: `checkpoints/vgg/simple_cnn/320/best_model.pt`
2. Loads checkpoint (restores model state)
3. Continues training from where it left off
4. NO need to restart from epoch 1

**Result:** Saves training time by ~50% if failure was mid-training

### Mode 6: Local Testing (No DagsHub)

```bash
python scripts/experiments.py --skip-dagshub-check --models simple_cnn --resolutions 64
```

**What happens:**
1. Skips DagsHub connectivity check
2. Trains SimpleCNN at 64×64 resolution only
3. No experiment tracking to remote server
4. Results saved locally in checkpoints/ and results/

**Use cases:**
- Testing without network
- Development/debugging
- Quick validation
- Offline machines

**Duration:** ~2 minutes (fastest option)

---

## 📊 Training Progress & Output

### Console Output Example

```
============================================================
ResoMap Resolution Training Experiment
Models: ['simple_cnn', 'resnet18']
Resolutions: [224, 320]
Epochs: 100
============================================================

Checking DagsHub for completed runs...
Found 0 completed run(s) in DagsHub

============================================================
Sweep Configuration:
  Models: 2 → ['simple_cnn', 'resnet18']
  Resolutions: 2 → [224, 320]
  Total combinations: 4
  Already completed: 0
============================================================

Models: 50%|███████▌| 1/2 [45:23<45:23, 2723.12s/it]

============================================================
[1/2|1/2] Training: simple_cnn @ 224x224
============================================================
Epoch 1/100: loss=1.234, val_loss=1.123, val_acc=0.456
Epoch 2/100: loss=1.100, val_loss=1.050, val_acc=0.523
...
Epoch 45/100: loss=0.234, val_loss=0.245, val_acc=0.893
Epoch 46/100: loss=0.232, val_loss=0.243, val_acc=0.894  ← Best!
...
Epoch 100/100: loss=0.156, val_loss=0.234, val_acc=0.895
[Success] simple_cnn@224 completed ✓

============================================================
[1/2|2/2] Training: simple_cnn @ 320x320
============================================================
[Info] simple_cnn@320 has local checkpoint, may resume
...
[Success] simple_cnn@320 completed ✓

Models: 100%|█████████| 2/2 [90:45<00:00, 2722.50s/it]

============================================================
[2/2|1/2] Training: resnet18 @ 224x224
============================================================
...
[Success] resnet18@224 completed ✓

[2/2|2/2] Training: resnet18 @ 320x320
============================================================
...
[Success] resnet18@320 completed ✓

============================================================
Sweep Summary:
  Total combinations: 4
  Completed: 4
  Failed: 0
  Remaining: 0
============================================================
```

### Status Messages

| Message | Meaning | Action |
|---------|---------|--------|
| `[Skip] model@res ✓ (completed in DagsHub)` | Already trained & tracked | None - auto-skipped |
| `[Info] model@res has local checkpoint, may resume` | Checkpoint exists locally | Will load and continue |
| `[Success] model@res completed ✓` | Training finished successfully | Proceeds to next |
| `[Error] model@res failed: ERROR_MSG` | Training failed | Resume with same args |
| `Epoch 45/100: loss=0.234, val_loss=0.245` | Training in progress | Keep running |
| `Early stopping triggered` | No validation improvement | Training stopped, saved best model |

---

## ✅ Checkpoint System

### Checkpoint Structure

```
checkpoints/
├── vgg/
│   └── simple_cnn/
│       ├── 224/
│       │   ├── best_model.pt        ← Loaded on resume
│       │   └── final_model.pt
│       ├── 256/
│       │   ├── best_model.pt
│       │   └── final_model.pt
│       ├── 320/
│       │   ├── best_model.pt        ← Resume example
│       │   └── final_model.pt
│       ├── 384/
│       └── 512/
├── resnet/
│   ├── resnet18/
│   │   ├── 224/
│   │   └── ...
│   └── resnet50/
└── mobilenet/
    └── mobilenet_v2/
```

### Checkpoint Contents

```python
checkpoint = torch.load('checkpoints/vgg/simple_cnn/320/best_model.pt')

checkpoint.keys() = [
    'model_state_dict',    # Model weights (what's loaded)
    'optimizer_state_dict', # Optimizer state
    'epoch',               # Which epoch reached
    'val_loss',            # Validation loss at save
    'val_accuracy',        # Validation accuracy at save
    'timestamp'            # When saved
]
```

### Resume Mechanism

**Automatic Resume (No Code Change Needed):**

```bash
# Training failed at epoch 45
python scripts/experiments.py --models simple_cnn --resolutions 320

# System automatically:
# 1. Detects checkpoint exists at checkpoints/vgg/simple_cnn/320/
# 2. Loads best_model.pt (best validation performance)
# 3. Continues training from epoch 46 (or epoch 1 if only 1 epoch passed)
# 4. Uses same hyperparameters
# 5. Saves new results (overwriting checkpoint)
```

---

## 🔄 Common Workflows

### Workflow 1: Quick Baseline (30 minutes)

```bash
# Quick validation that everything works
python scripts/experiments.py --models simple_cnn tiny_cnn --resolutions 64 128

# Check results
python scripts/aggregate_results.py --csv-only
cat results/all_results.csv
```

### Workflow 2: Single Model Analysis (2-3 hours)

```bash
# Train simple_cnn on all resolutions to understand resolution sensitivity
python scripts/experiments.py --models simple_cnn

# Generate results
python scripts/aggregate_results.py

# Analyze
grep "simple_cnn" results/all_results.csv | sort -t, -k3 -n
```

### Workflow 3: Model Comparison (2-3 hours)

```bash
# Train 4 models at standard resolution
python scripts/experiments.py --models simple_cnn resnet18 mobilenet_v2 simple_cnn --resolutions 224

# Compare accuracy
python -c "
import pandas as pd
df = pd.read_csv('results/all_results.csv')
print(df[['model', 'resolution', 'test_accuracy']].sort_values('test_accuracy', ascending=False))
"
```

### Workflow 4: Multi-Machine Training (distributed)

**Machine 1:**
```bash
python scripts/experiments.py --models simple_cnn tiny_cnn resnet18 --skip-dagshub-check
```

**Machine 2 (parallel):**
```bash
python scripts/experiments.py --models resnet50 mobilenet_v2 efficientnet_b0 --skip-dagshub-check
```

**Consolidation (run either machine):**
```bash
python scripts/experiments.py  # Auto-skips all completed
```

### Workflow 5: Incremental Training (days)

**Day 1: Start**
```bash
python scripts/experiments.py &  # Background process
```

**Day 2: Check progress**
```bash
python scripts/aggregate_results.py
cat results/detailed_report.txt  # See what's done
```

**Day 3: Continue** (if interrupted)
```bash
python scripts/experiments.py  # Resumes where it left off
```

---

## 🐛 Troubleshooting Execution

### Problem: "CUDA out of memory"

```
RuntimeError: CUDA out of memory. Tried to allocate 2.00GB
```

**Solution:**
```bash
# Reduce resolution
python scripts/experiments.py --resolutions 224 256

# OR reduce batch size (edit configs/training.yaml)
batch_size: 32  # Instead of 64

# OR enable mixed precision (if not already)
# Edit configs/system.yaml
use_mixed_precision: true
```

### Problem: "Model not found"

```
[Warning] Model 'vgg99' not found, skipping
```

**Solution:**
```bash
# Check available models
cat configs/sweep.yaml

# Use correct name
python scripts/experiments.py --models simple_cnn
```

### Problem: "Experiment failed at specific resolution"

```
[Error] simple_cnn@320 failed: CUDA out of memory
```

**Solution:**
```bash
# Resume training (loads checkpoint)
python scripts/experiments.py --models simple_cnn --resolutions 320

# Or skip this resolution
python scripts/experiments.py --models simple_cnn --resolutions 224 256 384 512
```

### Problem: "DagsHub connection error"

```
mlflow.exceptions.MlflowException: Failed to get experiment
```

**Solution:**
```bash
# Skip DagsHub check (for offline work)
python scripts/experiments.py --skip-dagshub-check

# Or setup MLflow locally
mlflow ui  # Start local tracking server
```

### Problem: "Training is very slow"

**Check:**
```bash
# 1. Is GPU being used?
nvidia-smi  # Watch GPU% during training

# 2. Enable mixed precision
# Edit configs/system.yaml
use_mixed_precision: true

# 3. Increase num_workers for data loading
# Edit configs/system.yaml
num_workers: 8

# 4. Reduce resolution
python scripts/experiments.py --resolutions 224
```

---

**Next:** [RESULTS_EVALUATION.md](RESULTS_EVALUATION.md) - Analyze training results  
**Back:** [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Project overview
