# Actual Training Results

## 🎯 Training Summary

**Duration:** 2 days continuous training (February 1-3, 2026)  
**Models Trained:** simple_cnn, tiny_cnn  
**Resolutions Tested:** 224, 256, 320, 384, 512 pixels  
**Total Experiments:** 10 (2 models × 5 resolutions)

**View All Results:** https://dagshub.com/Y-R-A-V-R-5/ResoMap/experiments

---

## 📊 Performance Results

### simple_cnn Performance

| Resolution | Accuracy | F1 Score | Inference Time | GPU Memory |
|------------|----------|----------|----------------|------------|
| 224px | 77.25% | 0.6889 | 40.1 ms | 458 MB |
| 256px | 81.19% | 0.7276 | 47.8 ms | 583 MB |
| **320px** | **81.69%** | **0.7342** | 67.4 ms | 885 MB |
| 384px | 69.72% | 0.5509 | 94.5 ms | 1254 MB |
| 512px | 72.65% | 0.5767 | 147.8 ms | 2192 MB |

**Best Configuration:** 320px resolution (highest accuracy: 81.69%)

### tiny_cnn Performance

| Resolution | Accuracy | F1 Score | Inference Time | GPU Memory |
|------------|----------|----------|----------------|------------|
| 224px | 67.18% | 0.4988 | 23.6 ms | 233 MB |
| **256px** | **67.81%** | **0.4981** | 25.4 ms | 298 MB |
| 320px | 66.14% | 0.4571 | 35.7 ms | 457 MB |
| 384px | 65.34% | 0.4788 | 44.1 ms | 649 MB |
| 512px | 64.80% | 0.4582 | 67.9 ms | 1139 MB |

**Best Configuration:** 256px resolution (highest accuracy: 67.81%)

---

## 🔍 Key Findings

### Accuracy vs Resolution

1. **simple_cnn** shows optimal performance at **320px** (81.69% accuracy)
   - Performance drops significantly at 384px (69.72%)
   - Moderate accuracy at baseline 224px (77.25%)

2. **tiny_cnn** peaks at **256px** (67.81% accuracy)
   - Small architecture performs consistently across resolutions
   - Accuracy decreases slightly at higher resolutions

### Speed vs Accuracy Tradeoff

**Fastest Configuration:**
- tiny_cnn @ 224px: 23.6ms per inference (1356 samples/sec)

**Best Accuracy:**
- simple_cnn @ 320px: 81.69% accuracy

**Balanced Option:**
- simple_cnn @ 256px: 81.19% accuracy, 47.8ms inference

### Memory Requirements

**Lowest Memory:**
- tiny_cnn @ 224px: 233 MB GPU memory

**Highest Memory:**
- simple_cnn @ 512px: 2192 MB GPU memory

**Memory Scaling Pattern:**
- ~2-3x increase from 224px → 320px
- ~5x increase from 224px → 512px

---

## 📈 Resolution Impact Analysis

### For simple_cnn:
- **224px → 320px:** +4.4% accuracy gain (77.25% → 81.69%)
- **320px → 384px:** -11.97% accuracy drop (81.69% → 69.72%)
- **Optimal range:** 256-320px

### For tiny_cnn:
- **224px → 256px:** +0.63% accuracy gain (67.18% → 67.81%)
- **256px → 512px:** -3.01% accuracy drop (67.81% → 64.80%)
- **Optimal range:** 224-256px (smaller architecture prefers lower resolutions)

---

## 💾 Data Files

All results are stored in:
- **Individual JSONs:** `results/test_results/*.json` (10 files)
- **Aggregated CSV:** `results/all_results.csv`
- **DagsHub Experiments:** https://dagshub.com/Y-R-A-V-R-5/ResoMap/experiments

### CSV Data (Full Results)

```csv
model,resolution,timestamp,best_val_loss,test_loss,test_accuracy,test_precision,test_recall,test_f1_score,profile_avg_time_sec,profile_throughput_samples_sec,profile_gpu_memory_peak_mb
simple_cnn,224,2026-02-01T16:12:06,inf,0.6127,0.7725,0.8102,0.6393,0.6889,0.0401,798.81,457.76
simple_cnn,256,2026-02-01T17:29:45,inf,0.5449,0.8119,0.8423,0.6896,0.7276,0.0478,669.99,583.48
simple_cnn,320,2026-02-01T19:13:49,inf,0.5179,0.8169,0.8433,0.6914,0.7342,0.0674,474.99,885.17
simple_cnn,384,2026-02-02T00:11:07,inf,0.8316,0.6972,0.7029,0.5157,0.5509,0.0945,338.79,1253.86
simple_cnn,512,2026-02-03T00:31:54,inf,0.7394,0.7265,0.7853,0.5386,0.5767,0.1478,216.50,2192.23
tiny_cnn,224,2026-02-03T05:36:05,inf,0.9044,0.6718,0.6571,0.4557,0.4988,0.0236,1356.25,232.74
tiny_cnn,256,2026-02-03T06:40:00,inf,0.8311,0.6781,0.7285,0.4737,0.4981,0.0254,1260.22,298.42
tiny_cnn,320,2026-02-03T07:51:26,inf,0.8934,0.6614,0.5797,0.4418,0.4571,0.0357,896.74,456.51
tiny_cnn,384,2026-02-03T11:01:52,inf,0.9401,0.6534,0.6984,0.4400,0.4788,0.0441,725.09,648.61
tiny_cnn,512,2026-02-03T13:10:55,inf,0.9340,0.6480,0.5929,0.4396,0.4582,0.0679,471.60,1138.79
```

---

## 🎓 Conclusions

1. **Resolution sweet spot exists:** Both models show non-monotonic accuracy patterns
   - simple_cnn: 320px optimal
   - tiny_cnn: 256px optimal
   - Higher ≠ always better

2. **Model complexity matters:** 
   - Larger simple_cnn benefits from higher resolutions (up to 320px)
   - Smaller tiny_cnn performs best at moderate resolutions (256px)

3. **Memory-performance tradeoff:**
   - 10x memory increase (233MB → 2192MB) for 224px → 512px on simple_cnn
   - Only 4% accuracy gain for 3x memory increase (224px → 320px)

4. **Deployment recommendations:**
   - **Mobile/Edge:** tiny_cnn @ 224-256px (low memory, fast)
   - **Server/Cloud:** simple_cnn @ 320px (best accuracy)
   - **Real-time:** tiny_cnn @ 224px (fastest inference)

---

## 🔬 Future Work

Other researchers can explore:
- **VGG family** (vgg11, vgg13) - Dense architectures
- **ResNet family** (resnet18, resnet34) - Skip connections
- **MobileNet family** (mobilenet_v2_small, mobilenet_v3_small) - Mobile optimized

All models are fully implemented in `src/models.py` and ready for training.

**Framework documentation:**
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Overview
- [TRAINING_EXECUTION.md](TRAINING_EXECUTION.md) - How to train
- [RESULTS_EVALUATION.md](RESULTS_EVALUATION.md) - Analysis tools

---

**Last Updated:** February 3, 2026  
**Experiment Tracking:** https://dagshub.com/Y-R-A-V-R-5/ResoMap/experiments
