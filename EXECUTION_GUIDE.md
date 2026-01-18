# ResoMap Execution Guide

## Overview

The ResoMap training script has been enhanced with advanced capabilities for managing large-scale experiments across 8 models and 5 resolutions. This guide covers all execution modes and features.

## Features

### 1. **Selective Model Execution** (`--models`)
Run training only for specific models on ALL resolutions.

```bash
# Train only VGG11 on all resolutions (224, 256, 320, 384, 512)
python scripts/experiments.py --models vgg11

# Train multiple specific models
python scripts/experiments.py --models vgg11 resnet18 mobilenet_v2
```

### 2. **Selective Resolution Execution** (`--resolutions`)
Run training for ALL models on specific resolutions only.

```bash
# Train all models only on 224 resolution
python scripts/experiments.py --resolutions 224

# Train all models on specific resolutions
python scripts/experiments.py --resolutions 224 320 384
```

### 3. **Combined Selective Execution**
Combine model and resolution filters to run specific model-resolution combinations.

```bash
# Train VGG11 and ResNet18 only on 224 and 320 resolutions
python scripts/experiments.py --models vgg11 resnet18 --resolutions 224 320
```

### 4. **Run All Experiments** (Default)
Without any arguments, trains all 8 models on all 5 resolutions (40 total combinations).

```bash
# Run all 40 experiments (8 models × 5 resolutions)
python scripts/experiments.py
```

## Advanced Features

### DagsHub Integration & Skip Completed Runs

**Automatic DagsHub Checking** (Default):
The script automatically checks your DagsHub project to identify which model-resolution combinations have already been successfully completed. These are skipped automatically to avoid redundant computation.

```
Checking DagsHub for completed runs...
Found 5 completed run(s) in DagsHub
  ✓ vgg11@224
  ✓ vgg11@256
  ✓ resnet18@224
  ✓ resnet18@256
  ✓ mobilenet_v2@224
```

**Skip DagsHub Check** (for local testing):
```bash
python scripts/experiments.py --skip-dagshub-check
```

### Resume Capability

If a training fails at a specific resolution, the system provides guidance to resume:

```
[Error] vgg11@320 failed: CUDA out of memory
[Resume] To resume this model at this resolution, run:
  python scripts/experiments.py --models vgg11 --resolutions 320
```

**How Resume Works:**
1. The system checks for existing checkpoints in `checkpoints/{model_family}/{model_name}/{resolution}/`
2. If a checkpoint exists, it's automatically loaded
3. Training continues from the checkpoint instead of restarting
4. This prevents losing progress on long training runs

**Example Resume Scenario:**
```bash
# Initial run with all models on all resolutions
python scripts/experiments.py

# VGG11 fails at 5th resolution (320)
# Other models continue and complete
# Resume VGG11 at the failed resolution:
python scripts/experiments.py --models vgg11 --resolutions 320

# VGG11 loads checkpoint and continues training
# No need to retrain resolutions 224, 256, 384, 512
```

## Checkpoint Structure

Checkpoints are organized hierarchically for easy management:

```
checkpoints/
├── vgg/
│   ├── vgg11/
│   │   ├── 224/
│   │   │   ├── best_model.pt      # Best checkpoint during training
│   │   │   └── final_model.pt     # Final checkpoint after training
│   │   ├── 256/
│   │   ├── 320/
│   │   ├── 384/
│   │   └── 512/
│   └── vgg16/
│       └── ...
├── resnet/
│   ├── resnet18/
│   │   └── ...
│   └── resnet50/
│       └── ...
└── mobilenet/
    └── mobilenet_v2/
        └── ...
```

## Usage Examples

### Scenario 1: Initial Large-Scale Run
Start all 8 models on all 5 resolutions:
```bash
python scripts/experiments.py
```

### Scenario 2: Resume After Failure
One model failed at 320 resolution:
```bash
# Resume VGG11 at 320 (automatically loads checkpoint and continues)
python scripts/experiments.py --models vgg11 --resolutions 320
```

### Scenario 3: Focus on Specific Model
You want to thoroughly evaluate only VGG11 on all resolutions:
```bash
python scripts/experiments.py --models vgg11
```

### Scenario 4: Test New Resolution
Evaluate all models on a new high resolution (e.g., 640):
```bash
# First add 640 to config.yaml sweep.resolutions, then:
python scripts/experiments.py --resolutions 640
```

### Scenario 5: Two GPU Machines
Run different models on two machines to parallelize work:

**Machine 1 - Dense models:**
```bash
python scripts/experiments.py --models vgg11 vgg16 --skip-dagshub-check
```

**Machine 2 - Efficient models:**
```bash
python scripts/experiments.py --models resnet18 mobilenet_v2 efficientnet_b0 --skip-dagshub-check
```

Then use DagsHub checking on final machine to consolidate:
```bash
python scripts/experiments.py  # Will skip all completed runs and train any remaining
```

## Output and Logging

### Console Output Example
```
============================================================
ResoMap Resolution Training Experiment
Models: ['vgg11', 'resnet18']
Resolutions: [224, 320]
Epochs: 100
============================================================

Checking DagsHub for completed runs...
Found 0 completed run(s) in DagsHub

============================================================
Sweep Configuration:
  Models: 2 → ['vgg11', 'resnet18']
  Resolutions: 2 → [224, 320]
  Total combinations: 4
  Already completed: 0
============================================================

Models: 50%|███████▌| 1/2 [45:23<45:23, 2723.12s/it]
[1/2|1/2] Training: vgg11 @ 224x224
...
[Success] vgg11@224 completed ✓

[Info] vgg11 has local checkpoint, may resume
[1/2|2/2] Training: vgg11 @ 320x320
...
[Success] vgg11@320 completed ✓

============================================================
Sweep Summary:
  Total combinations: 4
  Completed: 2
  Failed: 0
  Remaining: 2
============================================================
```

### MLflow/DagsHub Tracking
Each run is automatically logged with:
- ✓ Model architecture parameters
- ✓ Training hyperparameters
- ✓ Train/Val/Test metrics per epoch
- ✓ GPU memory usage statistics
- ✓ Model profiling results
- ✓ Best and final model checkpoints

## Command Reference

```bash
# Help
python scripts/experiments.py --help

# Run all
python scripts/experiments.py

# Specific models
python scripts/experiments.py --models vgg11 vgg16

# Specific resolutions
python scripts/experiments.py --resolutions 224 384

# Specific combinations
python scripts/experiments.py --models vgg11 --resolutions 224 320

# Skip DagsHub check
python scripts/experiments.py --skip-dagshub-check

# Resume failed run
python scripts/experiments.py --models vgg11 --resolutions 320
```

## Configuration

Models and resolutions are defined in `configs/config.yaml`:

```yaml
sweep:
  models:
    - vgg11
    - vgg16
    - resnet18
    - resnet50
    - mobilenet_v2
    - efficientnet_b0
    - densenet121
    - squeezenet_1_1
  
  resolutions:
    - 224
    - 256
    - 320
    - 384
    - 512
```

## Troubleshooting

### Q: A model failed at resolution 320, others completed. How do I resume?
**A:** Run `python scripts/experiments.py --models MODELNAME --resolutions 320`

### Q: How do I know which runs completed successfully?
**A:** The script automatically checks DagsHub and shows:
```
Found N completed run(s) in DagsHub
  ✓ model@resolution
```

### Q: Can I run experiments on multiple machines?
**A:** Yes! Use `--skip-dagshub-check` on distributed machines, then run a final consolidation on one machine without the flag to skip already-completed runs.

### Q: What if a run completes locally but doesn't appear in DagsHub?
**A:** DagsHub sync happens automatically during MLflow logging. If delays occur, use `--skip-dagshub-check` for the next run.

### Q: How much disk space do I need?
**A:** Each model checkpoint is ~100-500MB. With 8 models × 5 resolutions × (best + final model) = ~800 checkpoints. Budget 2-4GB for checkpoints.

## Performance Tips

1. **GPU Memory Issues**: If you hit OOM at higher resolutions, use selective execution:
   ```bash
   # Run high resolutions separately with batch size reduction
   python scripts/experiments.py --resolutions 512 384
   ```

2. **Monitor Progress**: Set up a separate terminal monitoring DagsHub or local checkpoint directories

3. **Distributed Training**: Use multiple machines with `--skip-dagshub-check`, then consolidate runs

4. **Checkpointing**: Always run with checkpointing enabled (automatic) so you can resume any failed run

---

For more information, see:
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Technical details
- [QUICK_START.md](QUICK_START.md) - Quick start guide
- [configs/config.yaml](configs/config.yaml) - Configuration reference
