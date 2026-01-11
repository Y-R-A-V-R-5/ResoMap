# 🚀 ResoMap GPU Enhancement - Final Setup Checklist

## ✅ Installation Steps

### 1. Install PyTorch with CUDA Support

**Important**: Check your CUDA version first!

```bash
# Check CUDA version (if installed)
nvidia-smi

# For CUDA 11.8
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121

# For CPU-only (fallback)
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
import captum

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

print(f"\nCaptum Available: {True}")
print("✓ All dependencies installed successfully!")
```

---

## 🔧 Configuration

ResoMap uses a modular configuration system with separate files for different aspects:

### Configuration Files

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

### For Different GPU Memory Sizes

#### **GPU with 4-6 GB VRAM** (e.g., GTX 1650, RTX 3050)
Edit `configs/training.yaml` and `configs/sweep.yaml`:
```yaml
# training.yaml
batch_size: 32

# sweep.yaml
resolutions: [224, 256, 320]  # Avoid 384, 512

# system.yaml
use_mixed_precision: true  # Essential!
```

#### **GPU with 8-10 GB VRAM** (e.g., RTX 3060, RTX 3080)
```yaml
# training.yaml
batch_size: 64

# sweep.yaml
resolutions: [224, 256, 320, 384]

# system.yaml
use_mixed_precision: true
```

#### **GPU with 12+ GB VRAM** (e.g., RTX 3090, RTX 4090)
```yaml
# training.yaml
batch_size: 128

# sweep.yaml
resolutions: [224, 256, 320, 384, 512]

# system.yaml
use_mixed_precision: true  # Still recommended
gradient_accumulation_steps: 1
```

### For CPU-Only Training (Fallback)
Edit `configs/system.yaml` and `configs/training.yaml`:
```yaml
# system.yaml
device: "cpu"
use_mixed_precision: false
num_threads: 8  # Adjust based on CPU cores

# training.yaml
batch_size: 16  # Smaller batch

# sweep.yaml
  resolutions: [64, 128, 224]  # Limit resolutions

data:
  num_workers: 4
```

---

## 📊 Recommended Workflow

### Phase 1: Quick Validation (30 minutes)
```bash
# 1. Test GPU setup with small experiment
python example_gpu_training.py \
    --resolution 128 \
    --model VGG11 \
    --epochs 3 \
    --batch-size 32

# Expected: Should complete in ~5-10 minutes on GPU
```

### Phase 2: Single Resolution Training (2-4 hours)
```bash
# 2. Train full model at one resolution
python example_gpu_training.py \
    --resolution 224 \
    --model VGG11 \
    --epochs 50 \
    --batch-size 64

# Expected: ~2-3 hours on modern GPU
```

### Phase 3: Explainability Analysis (30 minutes)
```bash
# 3. Generate explanations for trained model
python example_explainability.py \
    --checkpoint checkpoints/VGG11_224/best_model.pt \
    --model VGG11 \
    --resolution 224 \
    --method gradcam \
    --num-samples 20

# Expected: ~15-20 minutes
```

### Phase 4: Full Resolution Sweep (1-2 days)
```bash
# 4. Train at all resolutions
for resolution in 64 128 224 256 320 384; do
    python example_gpu_training.py \
        --resolution $resolution \
        --model VGG11 \
        --epochs 50 \
        --batch-size 64
done

# Expected: ~3-4 hours per resolution = 18-24 hours total
```

### Phase 5: Comprehensive Analysis (1-2 hours)
```bash
# 5. Compare all resolutions
python scripts/analysis.py \
    --checkpoint-dir checkpoints/ \
    --output-dir analysis_results/ \
    --models vgg11_224 \
    --resolutions 224 256 320 384 512

# Expected: ~1-2 hours
```

---

## 🎯 Research Questions & Experiments

### Experiment 1: Resolution vs Accuracy
**Question**: How does image resolution affect model accuracy?

**Steps**:
1. Train VGG11 at resolutions: [64, 128, 224, 256, 320]
2. Evaluate on test set
3. Plot accuracy vs resolution
4. Analyze: Where does accuracy plateau?

**Expected Results**:
- Low resolution (64): ~60-70% accuracy
- Medium resolution (224): ~80-85% accuracy
- High resolution (320+): ~85-90% accuracy (diminishing returns)

### Experiment 2: Computational Efficiency
**Question**: What's the speed-accuracy tradeoff?

**Steps**:
1. Profile inference at each resolution
2. Measure: time per batch, throughput, memory
3. Calculate: accuracy per second, accuracy per MB

**Expected Results**:
- 64x64: Fastest (100+ samples/s), lower accuracy
- 224x224: Balanced (30-50 samples/s), good accuracy
- 512x512: Slowest (5-10 samples/s), best accuracy

### Experiment 3: Explanation Consistency
**Question**: Do models focus on the same regions at different resolutions?

**Steps**:
1. Generate Grad-CAM for same images at different resolutions
2. Compute correlation between explanation heatmaps
3. Visualize side-by-side comparisons

**Expected Results**:
- High correlation at 224-320 range
- Lower correlation between 64 and 512
- Different features matter at different resolutions

### Experiment 4: Model Capacity
**Question**: Does VGG13 benefit more from high resolution than VGG11?

**Steps**:
1. Train both VGG11 and VGG13 at all resolutions
2. Compare accuracy gains from resolution increase
3. Analyze: Does deeper model extract more from high-res?

**Expected Results**:
- VGG13 shows larger gains at high resolutions
- VGG11 plateaus earlier
- Deeper models need higher resolution to utilize capacity

---

## 🔬 Performance Optimization Tips

### 1. Mixed Precision Training
**Impact**: 2x speedup, 40% memory savings
```python
# Already implemented in Trainer
use_amp = True  # Enable in config or script
```

### 2. Gradient Accumulation
**When**: GPU memory limited, want larger effective batch
```python
gradient_accumulation_steps = 4  # Effective batch = 64 * 4 = 256
```

### 3. Data Loading Optimization
**Impact**: Remove data loading bottleneck
```yaml
data:
  num_workers: 8              # Parallel loading
  prefetch_factor: 2          # Prefetch batches
  persistent_workers: true    # Reuse workers
  pin_memory: true            # Faster GPU transfer
```

### 4. CuDNN Benchmark
**Impact**: 5-10% speedup (automatically enabled)
```python
torch.backends.cudnn.benchmark = True
```

### 5. Monitor GPU Utilization
```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Look for:
# - GPU Utilization: Should be 90-100%
# - Memory Usage: Should be high but not OOM
# - Power Usage: Should be near TDP
```

---

## 🐛 Troubleshooting Guide

### Issue 1: CUDA Out of Memory

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions** (try in order):
1. Reduce batch size
   ```bash
   python example_gpu_training.py --batch-size 32  # Instead of 64
   ```

2. Enable mixed precision (if not already)
   ```bash
   python example_gpu_training.py --batch-size 64  # AMP enabled by default
   ```

3. Use gradient accumulation
   ```python
   gradient_accumulation_steps = 2  # Effective batch = 64 * 2
   ```

4. Reduce resolution
   ```bash
   python example_gpu_training.py --resolution 224  # Instead of 384
   ```

5. Clear GPU cache between runs
   ```python
   torch.cuda.empty_cache()
   ```

### Issue 2: Slow Training

**Check 1**: GPU utilization
```bash
nvidia-smi
# Should show: GPU-Util: 90-100%
```

**If Low GPU Usage**:
- Increase `num_workers` (data loading)
- Enable `pin_memory`
- Enable `persistent_workers`
- Check if CPU bottleneck exists

**Check 2**: Data loading speed
```python
# Time data loading
import time
start = time.time()
for batch in train_loader:
    break
print(f"Batch loading: {time.time() - start:.2f}s")
# Should be < 0.1s
```

### Issue 3: Captum Import Error

**Error**:
```
ImportError: No module named 'captum'
```

**Solution**:
```bash
pip install captum
```

### Issue 4: No GPU Detected

**Check CUDA**:
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

**If False**:
1. Check NVIDIA driver: `nvidia-smi`
2. Reinstall PyTorch with correct CUDA version
3. Check CUDA installation: `nvcc --version`

### Issue 5: Explainability Errors

**Error**: Target layer not found

**Solution**: Specify target layer manually
```python
# For VGG models, use last conv stage
target_layer = model.stages['stage5']

explainer = ModelExplainer(
    model=model,
    device=device,
    target_layer=target_layer
)
```

---

## 📈 Expected Training Times

### On RTX 3080 (10GB VRAM):

| Resolution | Batch Size | Epoch Time | Total (50 epochs) |
|------------|-----------|------------|-------------------|
| 64x64      | 128       | 2 min      | 1.5 hours        |
| 128x128    | 64        | 3 min      | 2.5 hours        |
| 224x224    | 64        | 5 min      | 4 hours          |
| 256x256    | 32        | 6 min      | 5 hours          |
| 320x320    | 32        | 8 min      | 6.5 hours        |
| 384x384    | 16        | 10 min     | 8 hours          |
| 512x512    | 8         | 15 min     | 12 hours         |

### On RTX 3090 (24GB VRAM):
- 30-40% faster than 3080
- Can use larger batch sizes
- Can train 512x512 with batch=16

### On CPU (16-core):
- 20-40x slower than GPU
- 224x224: ~2-3 hours per epoch
- Not recommended for full experiments

---

## 💡 Pro Tips

1. **Start Small**: Always test with 1-2 epochs first
2. **Monitor Everything**: Use `nvidia-smi`, `htop`, TensorBoard
3. **Save Checkpoints**: Never lose training progress
4. **Document Results**: Keep notes on what works
5. **Compare Fairly**: Same hyperparameters across resolutions
6. **Visualize Early**: Generate explanations to verify model behavior
7. **Use Version Control**: Git commit configurations and results

---

## 📚 Useful Commands

### Monitor GPU
```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Log GPU usage
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total --format=csv -l 1 > gpu_log.csv
```

### Jupyter Notebook Setup
```bash
# If you want to use notebooks
pip install jupyter ipykernel
python -m ipykernel install --user --name resomap --display-name "ResoMap"

# Launch Jupyter
jupyter notebook
```

### Quick Test Script
```python
# test_gpu.py - Quick GPU test
import torch
import time

device = torch.device('cuda')
x = torch.randn(1000, 1000).to(device)

start = time.time()
for _ in range(1000):
    y = torch.matmul(x, x)
torch.cuda.synchronize()
print(f"Time: {time.time() - start:.2f}s")
# Should be < 1s on modern GPU
```

---

## 🎓 Learning Resources

1. **PyTorch AMP**: https://pytorch.org/docs/stable/amp.html
2. **Captum Tutorials**: https://captum.ai/tutorials/
3. **Grad-CAM Paper**: https://arxiv.org/abs/1610.02391
4. **Mixed Precision Training**: https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html

---

## ✅ Pre-Experiment Checklist

Before starting your full resolution study:

- [ ] GPU properly detected and working
- [ ] All dependencies installed
- [ ] Dataset downloaded and in correct format
- [ ] Config file updated with your settings
- [ ] Tested with short run (1-2 epochs)
- [ ] Checkpointing directory created
- [ ] Sufficient disk space (20+ GB recommended)
- [ ] Results directory structure planned
- [ ] Experiments documented (research questions, hypotheses)

---

## 🎉 You're Ready to Go!

Your ResoMap project is now fully set up for GPU-accelerated research. Start with the recommended workflow above, and enjoy exploring how resolution affects your models!

**Good luck with your research! 🚀**

---

*Need help? Check:*
- *`README_GPU_ENHANCEMENT.md` for detailed usage*
- *`IMPLEMENTATION_SUMMARY.md` for what was changed*
- *GitHub Issues for common problems*