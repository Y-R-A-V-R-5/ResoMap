# ResoMap Resume & Checkpoint Guide

## Understanding Resume Capability

The ResoMap training system automatically saves checkpoints after each epoch. If training fails or is interrupted, you can resume from the last checkpoint without losing progress.

## How Resume Works

### Automatic Checkpoint Saving
During training, the system saves two types of checkpoints:

1. **best_model.pt** - Saved whenever validation loss improves
2. **final_model.pt** - Saved at the end of training

### Resume Flow

```
Initial Run
├─ VGG11 @ 224 → Completed ✓
├─ VGG11 @ 256 → Completed ✓
├─ VGG11 @ 320 → FAILED (e.g., GPU OOM at epoch 45/100)
│   ├─ checkpoint created up to epoch 44
│   └─ best_model.pt exists for 320
├─ VGG11 @ 384 → Not started (because prev failed)
└─ VGG11 @ 512 → Not started

Resume Run
bash: python scripts/experiments.py --models vgg11 --resolutions 320
├─ Detects existing checkpoint for VGG11@320
├─ Loads best_model.pt
├─ Continues training from epoch 45
└─ Completes 320 → Success ✓

Next Full Run
bash: python scripts/experiments.py
├─ Skips VGG11@224 (completed in DagsHub)
├─ Skips VGG11@256 (completed in DagsHub)
├─ Skips VGG11@320 (completed in DagsHub)
├─ Starts VGG11@384 → Success ✓
├─ Starts VGG11@512 → Success ✓
└─ Continues with other models...
```

## Checkpoint Detection

The system detects checkpoints automatically based on the directory structure:

```
checkpoints/
├── vgg/
│   └── vgg11/
│       ├── 224/
│       │   ├── best_model.pt     ← Detected for VGG11@224
│       │   └── final_model.pt
│       └── 320/
│           ├── best_model.pt     ← Detected for VGG11@320 (resume)
│           └── final_model.pt
```

When starting an experiment, the system checks:
- Does `checkpoints/{model_family}/{model_name}/{resolution}/` exist?
- Does `best_model.pt` exist in that directory?
- If YES → Checkpoint exists and will be loaded

## Resume Scenarios

### Scenario 1: GPU Memory Error

**What happened:**
```
Epoch 45/100, Batch 32/50
RuntimeError: CUDA out of memory. Tried to allocate 2.00GB
```

**Checkpoint status:**
- Checkpoints saved up to epoch 44 ✓
- best_model.pt exists ✓
- Training stopped at epoch 45 ✗

**Resume:**
```bash
python scripts/experiments.py --models vgg11 --resolutions 320
# Output: [ResoMap] Loaded checkpoint from checkpoints/vgg/vgg11/320/best_model.pt
# Output: Model restored from checkpoint, will continue training
# Continues from epoch 45/100
```

### Scenario 2: Connection Lost

**What happened:**
```
Training interrupted (network disconnected, process killed, etc.)
```

**Checkpoint status:**
- Last saved checkpoint exists ✓
- Training stopped mid-epoch

**Resume:**
```bash
python scripts/experiments.py --models vgg11 --resolutions 320
# Automatically detects and loads checkpoint
# Continues from where it stopped
```

### Scenario 3: Manual Interruption (Ctrl+C)

**What happened:**
```
Training interrupted by user (Ctrl+C)
```

**Checkpoint status:**
- Last successfully saved checkpoint exists ✓
- Current epoch may be lost

**Resume:**
```bash
python scripts/experiments.py --models vgg11 --resolutions 320
# Loads checkpoint and continues
# May repeat last epoch if it wasn't saved
```

### Scenario 4: Multiple Model Failures

**What happened:**
```
python scripts/experiments.py

# Results:
# vgg11 @ 224: Success ✓
# vgg11 @ 256: Success ✓
# vgg11 @ 320: FAILED (GPU OOM)
# resnet18 @ 224: Skipped (because previous model failed)
# ... other models skipped
```

**Recovery:**
```bash
# Resume the failed one
python scripts/experiments.py --models vgg11 --resolutions 320

# After success, continue with the sweep
python scripts/experiments.py
# Will skip vgg11@224, vgg11@256, vgg11@320 (in DagsHub)
# Will resume and complete all remaining combinations
```

## Manual Checkpoint Management

### Viewing Checkpoint Info

```bash
# List all available checkpoints
dir checkpoints /s /b

# Check specific model
dir checkpoints\vgg\vgg11\320

# Output:
# 2024-12-15  10:45 AM      123,456,789 best_model.pt
# 2024-12-15  12:30 PM      123,456,789 final_model.pt
```

### Loading Checkpoint Content

```python
import torch

# Check what's in a checkpoint
checkpoint = torch.load('checkpoints/vgg/vgg11/320/best_model.pt')
print(checkpoint.keys())
# Output: dict_keys(['model_state_dict', 'model', 'resolution', 'epoch', 'val_loss', 'val_accuracy'])

print(f"Epoch: {checkpoint['epoch']}")
print(f"Val Loss: {checkpoint['val_loss']:.4f}")
print(f"Val Accuracy: {checkpoint['val_accuracy']:.4f}")
```

### Recovering from Corrupted Checkpoint

```bash
# If checkpoint is corrupted and resume fails:
# Option 1: Delete and restart fresh
del checkpoints\vgg\vgg11\320\best_model.pt
python scripts/experiments.py --models vgg11 --resolutions 320

# Option 2: Keep final_model.pt as backup (if available)
# (System will try to load best_model.pt first, then continue)
```

## Advanced Resume Patterns

### Pattern 1: Batch Recovery

Multiple models failed at different resolutions:

```bash
# Run all failed combinations together
python scripts/experiments.py --models vgg11 resnet18 --resolutions 320 384

# System will:
# - Load checkpoint for vgg11@320 and continue
# - Load checkpoint for vgg11@384 and continue
# - Load checkpoint for resnet18@320 and continue
# - Load checkpoint for resnet18@384 and continue
```

### Pattern 2: Model Recovery Only

All resolutions of a model failed:

```bash
# Resume all resolutions of one model
python scripts/experiments.py --models vgg11

# System will:
# - Resume vgg11@224 if checkpoint exists
# - Resume vgg11@256 if checkpoint exists
# - Resume vgg11@320 if checkpoint exists
# - Resume vgg11@384 if checkpoint exists
# - Resume vgg11@512 if checkpoint exists
```

### Pattern 3: Progressive Recovery

Resume training progressively from lower to higher resolutions:

```bash
# Day 1: Resume low resolutions first
python scripts/experiments.py --resolutions 224 256

# Day 2: After verifying those work
python scripts/experiments.py --resolutions 320 384 512
```

## Checkpoint Cleanup (Optional)

### Remove All Checkpoints (Start Fresh)

```bash
# ⚠️ WARNING: This will delete all checkpoints!
rmdir checkpoints /s /q
python scripts/experiments.py  # Starts completely fresh
```

### Remove Specific Model Checkpoints

```bash
# Remove all VGG11 checkpoints
rmdir checkpoints\vgg\vgg11 /s /q

# Remove specific resolution
rmdir checkpoints\vgg\vgg11\320 /s /q
```

### Keep Only Best Models

```bash
# Archive final models, delete intermediate checkpoints
# (Intermediate checkpoints are only for resume, final models are saved to MLflow)
```

## Monitoring Resume Progress

### Check Checkpoint Timestamps

```bash
# See when checkpoints were last saved
dir checkpoints\vgg\vgg11\320 /T:W

# Verify before resuming
# Check if timestamps are recent (within last training attempt)
```

### Check Log Files

```bash
# MLflow logs contain epoch-by-epoch information
# Check logs to see which epoch was reached before failure
```

## Troubleshooting Resume Issues

### Issue: Resume Not Working

**Symptoms:**
```bash
python scripts/experiments.py --models vgg11 --resolutions 320
# Training appears to start from epoch 1 (not resuming)
```

**Causes & Solutions:**
```bash
# 1. Checkpoint doesn't exist
dir checkpoints\vgg\vgg11\320
# If empty: Try running normally, system will create checkpoint

# 2. Checkpoint is corrupted
# Delete and restart
del checkpoints\vgg\vgg11\320\best_model.pt
python scripts/experiments.py --models vgg11 --resolutions 320

# 3. Directory structure mismatch
# Verify directory exists and has correct structure
dir checkpoints\vgg\vgg11\320\best_model.pt
```

### Issue: Checkpoint Load Error

**Symptoms:**
```
[ResoMap] ⚠ Could not load checkpoint: Unable to find class...
[ResoMap] Could not load checkpoint, starting fresh
```

**Causes:**
- Model code changed after checkpoint was saved
- PyTorch version mismatch
- Corrupted checkpoint file

**Solution:**
```bash
# Delete checkpoint and restart
del checkpoints\vgg\vgg11\320\best_model.pt
python scripts/experiments.py --models vgg11 --resolutions 320
```

## Best Practices

1. **Always run with checkpointing enabled** (default behavior)
   ```bash
   python scripts/experiments.py  # Checkpointing automatic
   ```

2. **Monitor for failures** during first few epochs
   ```bash
   # If you see errors early, check GPU memory, batch size, etc.
   ```

3. **Don't manually edit checkpoints** unless necessary
   ```bash
   # Let the system manage checkpoint files
   ```

4. **Backup checkpoints** if disk space allows
   ```bash
   # Checkpoints are small (~200MB each) but valuable
   ```

5. **Use DagsHub tracking** for production runs
   ```bash
   # Default: DagsHub checking enabled
   # Ensures no duplicate work across runs
   ```

6. **Use selective execution** for reliability
   ```bash
   # Instead of: python scripts/experiments.py
   # Consider: python scripts/experiments.py --resolutions 224 256
   #           (Then proceed to higher resolutions)
   ```

## Summary

| Feature | Capability |
|---------|-----------|
| Automatic Checkpointing | ✓ Saves after each epoch |
| Resume Detection | ✓ Automatically detects existing checkpoints |
| Progress Preservation | ✓ Continues from last saved epoch |
| Multi-Model Resume | ✓ Can resume specific model-resolution combos |
| DagsHub Integration | ✓ Skips DagsHub-completed runs |
| Error Recovery | ✓ Provides recovery instructions on failure |

---

**For implementation details:** See [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md)
