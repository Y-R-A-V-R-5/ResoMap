# Results Export & Evaluation Guide

## 📊 Overview

After training completes, ResoMap automatically exports test results and provides tools to aggregate and analyze them.

**Data Flow:**
```
Training completes for each model-resolution pair
    ↓
Auto-save JSON: results/test_results/model_resolution.json
    ↓
Run aggregation script
    ↓
CSV: results/all_results.csv
JSON summary: results/results_summary.json
Text report: results/detailed_report.txt
    ↓
Analyze in Excel, Python, or command-line
```

---

## ✅ Automatic JSON Export

### What Gets Saved Automatically

After each training completes, ResoMap saves:

**File:** `results/test_results/{model}_{resolution}.json`

**Example:** `results/test_results/vgg11_224.json`

**Contents:**
```json
{
  "model": "vgg11",
  "resolution": 224,
  "timestamp": "2026-01-18T14:30:45.123456",
  
  "best_val_loss": 0.5441,
  
  "test_metrics": {
    "loss": 0.5519,
    "accuracy": 0.8934,
    "precision": 0.8901,
    "recall": 0.8834,
    "f1_score": 0.8867
  },
  
  "profiling": {
    "avg_time_sec": 0.0231,
    "throughput_samples_sec": 1234.56,
    "gpu_memory_peak_mb": 4096,
    "cpu_memory_peak_mb": 2048
  }
}
```

### Where Files Are Saved

```
results/
├── test_results/
│   ├── vgg11_224.json         ✅ Auto-saved
│   ├── vgg11_256.json         ✅ Auto-saved
│   ├── vgg11_320.json         ✅ Auto-saved
│   ├── vgg11_384.json         ✅ Auto-saved
│   ├── vgg11_512.json         ✅ Auto-saved
│   ├── resnet18_224.json      ✅ Auto-saved
│   └── ... (up to 40 files for 8×5 models)
│
├── all_results.csv            ❌ Manual (run aggregation script)
├── results_summary.json       ❌ Manual (run aggregation script)
└── detailed_report.txt        ❌ Manual (run aggregation script)
```

### Automatic Saving Process

In [src/experiment.py](src/experiment.py), after training each model:

```python
def run_experiment(self, model_name, resolution, dataset_path):
    # ... training code ...
    
    # Save test results automatically
    self._save_results_to_json(
        model_name=model_name,
        resolution=resolution,
        test_metrics=test_metrics,
        profiling_data=profiling_data
    )
    # → Saves to results/test_results/{model_name}_{resolution}.json
```

**No manual action needed!** Results are saved automatically.

---

## 🔄 Manual Aggregation

### Command: Aggregate Results

After training is complete, combine all JSON files into CSV and reports:

```bash
python scripts/aggregate_results.py
```

**What happens:**
1. Finds all `results/test_results/*.json` files
2. Combines into `results/all_results.csv`
3. Generates `results/results_summary.json`
4. Creates `results/detailed_report.txt`
5. Prints console summary

### Options

**Generate all outputs (default):**
```bash
python scripts/aggregate_results.py
```

**Generate only CSV:**
```bash
python scripts/aggregate_results.py --csv-only
```

**Generate only summary:**
```bash
python scripts/aggregate_results.py --summary-only
```

**Generate only text report:**
```bash
python scripts/aggregate_results.py --report-only
```

**Custom output filenames:**
```bash
python scripts/aggregate_results.py \
  --csv my_results.csv \
  --summary my_summary.json \
  --report my_report.txt
```

**Skip console output:**
```bash
python scripts/aggregate_results.py --no-console
```

---

## 📈 CSV File Analysis

### File Location
```
results/all_results.csv
```

### CSV Columns

| Column | Type | Example | Meaning |
|--------|------|---------|---------|
| model | str | vgg11 | Model architecture name |
| resolution | int | 224 | Input image resolution (pixels) |
| timestamp | str | 2026-01-18T14:30:45 | When experiment ran |
| best_val_loss | float | 0.5441 | Best validation loss during training |
| test_loss | float | 0.5519 | Final test loss |
| test_accuracy | float | 0.8934 | Test accuracy (0-1) |
| test_precision | float | 0.8901 | Precision score (0-1) |
| test_recall | float | 0.8834 | Recall score (0-1) |
| test_f1_score | float | 0.8867 | F1 score (0-1) |
| profile_avg_time_sec | float | 0.0231 | Avg inference time per image (seconds) |
| profile_throughput_samples_sec | float | 1234.56 | Images processed per second |
| profile_gpu_memory_peak_mb | int | 4096 | Peak GPU memory during inference (MB) |
| profile_peak_cpu_memory_mb | int | 2048 | Peak CPU memory during inference (MB) |

### Example CSV Content

```csv
model,resolution,timestamp,best_val_loss,test_loss,test_accuracy,test_precision,test_recall,test_f1_score,profile_avg_time_sec,profile_throughput_samples_sec,profile_gpu_memory_peak_mb,profile_peak_cpu_memory_mb
vgg11,224,2026-01-18T14:30:45,0.5441,0.5519,0.8934,0.8901,0.8834,0.8867,0.0231,1234.56,4096,2048
vgg11,256,2026-01-18T15:15:22,0.5523,0.5612,0.8876,0.8843,0.8776,0.8809,0.0312,956.34,5120,2256
vgg11,320,2026-01-18T16:45:10,0.5634,0.5723,0.8812,0.8779,0.8712,0.8745,0.0534,467.80,7168,2512
resnet18,224,2026-01-18T17:20:33,0.4234,0.4356,0.9123,0.9087,0.9023,0.9055,0.0089,3456.78,2048,1536
resnet18,256,2026-01-18T18:05:44,0.4345,0.4467,0.9067,0.9031,0.8967,0.8999,0.0134,2341.82,2560,1792
```

### Opening in Excel

1. **Open file:** `results/all_results.csv`
2. **Format as table:**
   - Select all data
   - Format → As Table
3. **Create pivot table:**
   - Insert → PivotTable
   - Rows: model
   - Columns: resolution
   - Values: test_accuracy (average)
4. **Create chart:**
   - Select pivot table
   - Insert → Chart
   - X-axis: resolution
   - Y-axis: accuracy
   - Series: different models

### Python Analysis Examples

#### Best Overall Model
```python
import pandas as pd

df = pd.read_csv('results/all_results.csv')

# Top 5 best results
top5 = df.nlargest(5, 'test_accuracy')[['model', 'resolution', 'test_accuracy']]
print(top5)
```

Output:
```
      model  resolution  test_accuracy
0     vgg16         384           0.8956
1     vgg16         320           0.8923
2     vgg11         384           0.8912
3     resnet50      384           0.8901
4     resnet50      320           0.8876
```

#### Accuracy by Model
```python
# Average accuracy per model
model_accuracy = df.groupby('model')['test_accuracy'].agg(['mean', 'min', 'max', 'std'])
print(model_accuracy.sort_values('mean', ascending=False))
```

Output:
```
              mean      min      max       std
model                                        
vgg16       0.8812  0.7234  0.8956  0.0512
vgg13       0.8756  0.7156  0.8923  0.0498
resnet50    0.8654  0.6834  0.8901  0.0587
...
```

#### Accuracy by Resolution
```python
# Average accuracy per resolution
res_accuracy = df.groupby('resolution')['test_accuracy'].agg(['mean', 'min', 'max', 'std'])
print(res_accuracy.sort_values('resolution'))
```

Output:
```
            mean      min      max       std
resolution                                 
224        0.8234  0.5123  0.8956  0.0912
256        0.8456  0.6234  0.8923  0.0834
320        0.8612  0.7123  0.8876  0.0756
384        0.8734  0.7456  0.8945  0.0634
512        0.8756  0.7656  0.8967  0.0567
```

#### Speed vs Accuracy Tradeoff
```python
# Find fastest models with good accuracy
df['accuracy_per_ms'] = df['test_accuracy'] / (df['profile_avg_time_sec'] * 1000)

best_tradeoff = df.nlargest(10, 'accuracy_per_ms')[['model', 'resolution', 'test_accuracy', 'profile_avg_time_sec']]
print(best_tradeoff)
```

#### Memory-Efficient Models
```python
# Find models with lowest GPU memory
efficient = df.nsmallest(10, 'profile_gpu_memory_peak_mb')[['model', 'resolution', 'test_accuracy', 'profile_gpu_memory_peak_mb']]
print(efficient)
```

---

## 📋 Summary JSON

### File Location
```
results/results_summary.json
```

### Structure

```json
{
  "generated_at": "2026-01-18T19:45:30.123456",
  
  "statistics": {
    "total_experiments": 40,
    "completed_experiments": 40,
    "avg_accuracy": 0.8523,
    "max_accuracy": 0.8956,
    "min_accuracy": 0.5234,
    "avg_inference_time_sec": 0.0345,
    "max_inference_time_sec": 0.0890,
    "min_inference_time_sec": 0.0089
  },
  
  "by_model": {
    "vgg11": {
      "experiments": 5,
      "avg_accuracy": 0.8612,
      "best_accuracy": 0.8923,
      "best_resolution": 320,
      "avg_inference_time": 0.0345
    },
    "resnet18": {
      "experiments": 5,
      "avg_accuracy": 0.8734,
      "best_accuracy": 0.8945,
      "best_resolution": 384,
      "avg_inference_time": 0.0134
    },
    ...
  },
  
  "by_resolution": {
    "224": {
      "experiments": 8,
      "avg_accuracy": 0.8234,
      "best_accuracy": 0.8456,
      "best_model": "resnet50"
    },
    "256": {
      "experiments": 8,
      "avg_accuracy": 0.8456,
      "best_accuracy": 0.8723,
      "best_model": "vgg16"
    },
    ...
  },
  
  "top_5_results": [
    {"model": "vgg16", "resolution": 384, "accuracy": 0.8956},
    {"model": "vgg16", "resolution": 320, "accuracy": 0.8923},
    {"model": "vgg11", "resolution": 384, "accuracy": 0.8912},
    {"model": "resnet50", "resolution": 384, "accuracy": 0.8901},
    {"model": "resnet50", "resolution": 320, "accuracy": 0.8876}
  ]
}
```

### Reading Summary in Python

```python
import json

with open('results/results_summary.json', 'r') as f:
    summary = json.load(f)

# Overall statistics
print(f"Total experiments: {summary['statistics']['total_experiments']}")
print(f"Average accuracy: {summary['statistics']['avg_accuracy']:.4f}")

# Top 5 results
print("\nTop 5 Results:")
for i, result in enumerate(summary['top_5_results'], 1):
    print(f"{i}. {result['model']}@{result['resolution']} - Accuracy: {result['accuracy']:.4f}")

# Best model per resolution
print("\nBest Model per Resolution:")
for res, data in summary['by_resolution'].items():
    print(f"  {res}×{res}: {data['best_model']} ({data['best_accuracy']:.4f})")
```

---

## 📄 Detailed Text Report

### File Location
```
results/detailed_report.txt
```

### Structure

The report contains:

1. **Header** - Generation timestamp, total experiments
2. **Overall Statistics** - Min/max/avg accuracy, speed metrics
3. **Top 10 Results** - Best performing model-resolution combos
4. **By Model** - Summary for each trained model
5. **By Resolution** - Summary for each resolution tested
6. **Detailed Table** - Complete results table

### Example Content

```
============================================================
ResoMap Experiment Results Report
Generated: 2026-01-18 19:45:30.123456
============================================================

OVERALL STATISTICS
------------------
Total Experiments:    40
Completed:            40
Failed:               0

Accuracy:
  Average:            0.8523
  Best:               0.8956 (vgg16@384)
  Worst:              0.5234 (simple_cnn@64)
  Std Dev:            0.0845

Inference Speed:
  Average Time:       0.0345 sec/sample
  Throughput (avg):   567.89 samples/sec
  Fastest:            0.0089 sec (resnet18@224)
  Slowest:            0.0890 sec (vgg16@512)

Memory Usage:
  GPU Memory (avg):   4567.8 MB
  CPU Memory (avg):   2134.5 MB
  GPU Peak:           8192 MB
  CPU Peak:           4096 MB

============================================================
TOP 10 RESULTS
============================================================
1.  vgg16           @ 384×384 | Accuracy: 0.8956 | F1: 0.8923
2.  vgg16           @ 320×320 | Accuracy: 0.8923 | F1: 0.8890
3.  vgg11           @ 384×384 | Accuracy: 0.8912 | F1: 0.8879
4.  resnet50        @ 384×384 | Accuracy: 0.8901 | F1: 0.8868
5.  resnet50        @ 320×320 | Accuracy: 0.8876 | F1: 0.8843
...

============================================================
RESULTS BY MODEL
============================================================

VGG11:
  Experiments:        5
  Resolutions:        224, 256, 320, 384, 512
  Best Accuracy:      0.8923 @ 320×320
  Worst Accuracy:     0.7456 @ 224×224
  Avg Accuracy:       0.8612
  Avg Inference Time: 0.0345 sec
  Best Speed:         0.0231 sec @ 224×224

VGG16:
  Experiments:        5
  Resolutions:        224, 256, 320, 384, 512
  Best Accuracy:      0.8956 @ 384×384
  Worst Accuracy:     0.7634 @ 224×224
  Avg Accuracy:       0.8734
  Avg Inference Time: 0.0412 sec
  Best Speed:         0.0281 sec @ 224×224

...

============================================================
RESULTS BY RESOLUTION
============================================================

224×224:
  Experiments:        8
  Best Model:         resnet50 (0.8456)
  Worst Model:        simple_cnn (0.5123)
  Avg Accuracy:       0.8234
  Speediest:          resnet18 (0.0089 sec)

256×256:
  Experiments:        8
  Best Model:         vgg16 (0.8723)
  Worst Model:        tiny_cnn (0.5456)
  Avg Accuracy:       0.8456
  Speediest:          resnet18 (0.0123 sec)

...

============================================================
DETAILED RESULTS TABLE
============================================================
Model              Resolution  Accuracy  Precision  Recall    F1-Score  Time(ms)  GPU Mem(MB)
─────────────────────────────────────────────────────────────────────────────────────────────
vgg11              224         0.8456    0.8423     0.8334    0.8378    23.1      4096
vgg11              256         0.8612    0.8579     0.8490    0.8534    31.2      5120
vgg11              320         0.8923    0.8890     0.8801    0.8845    53.4      7168
...

============================================================
```

### Viewing the Report

```bash
# Print to console
type results/detailed_report.txt

# Or on Mac/Linux
cat results/detailed_report.txt

# Search for specific model
grep "vgg11" results/detailed_report.txt

# Find best accuracy
grep "Best Accuracy" results/detailed_report.txt
```

---

## 🔍 Analysis Workflows

### Workflow 1: Find Best Overall Model

```bash
# View detailed report
cat results/detailed_report.txt | grep -A2 "TOP 10"
```

Or in Python:
```python
import pandas as pd
df = pd.read_csv('results/all_results.csv')
best = df.loc[df['test_accuracy'].idxmax()]
print(f"Best: {best['model']} @ {best['resolution']}px")
print(f"Accuracy: {best['test_accuracy']:.4f}")
```

### Workflow 2: Compare Models at Fixed Resolution

```bash
# All models at 224×224
grep ",224," results/all_results.csv | sort -t, -k6 -rn
```

Or in Python:
```python
df = pd.read_csv('results/all_results.csv')
at_224 = df[df['resolution'] == 224].sort_values('test_accuracy', ascending=False)
print(at_224[['model', 'test_accuracy']])
```

### Workflow 3: Resolution Sensitivity per Model

```bash
# Plot accuracy vs resolution for VGG11
python -c "
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results/all_results.csv')
vgg11 = df[df['model'] == 'vgg11'].sort_values('resolution')

plt.figure(figsize=(8, 5))
plt.plot(vgg11['resolution'], vgg11['test_accuracy'], marker='o')
plt.xlabel('Resolution (pixels)')
plt.ylabel('Accuracy')
plt.title('VGG11: Accuracy vs Resolution')
plt.grid(True)
plt.savefig('accuracy_vs_resolution.png')
"
```

### Workflow 4: Speed vs Accuracy Tradeoff

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results/all_results.csv')

plt.figure(figsize=(10, 6))
for model in df['model'].unique():
    model_data = df[df['model'] == model]
    plt.scatter(model_data['profile_avg_time_sec'], 
               model_data['test_accuracy'],
               label=model, s=100)

plt.xlabel('Inference Time (seconds)')
plt.ylabel('Accuracy')
plt.title('Speed vs Accuracy Tradeoff')
plt.legend()
plt.grid(True)
plt.savefig('speed_accuracy_tradeoff.png')
```

---

## 📊 Metrics Explained

### Classification Metrics

- **Accuracy:** (TP + TN) / (TP + TN + FP + FN)
  - Overall correctness, 0-1 scale
  - Best for balanced datasets

- **Precision:** TP / (TP + FP)
  - Of positive predictions, how many correct?
  - Important when false positives are costly

- **Recall:** TP / (TP + FN)
  - Of actual positives, how many found?
  - Important when false negatives are costly

- **F1-Score:** 2 × (Precision × Recall) / (Precision + Recall)
  - Harmonic mean of precision and recall
  - Best for imbalanced data

**Example (skin lesion classification):**
```
100 test images total
90 correctly classified
5 wrong positives (predicted melanoma, actually benign)
5 wrong negatives (predicted benign, actually melanoma)

Accuracy = 90/100 = 0.90
Precision = 45/50 = 0.90 (assuming 45 true melanomas)
Recall = 45/50 = 0.90
F1 = 0.90
```

### Performance Metrics

- **Inference Time:** How long to classify one image (seconds)
  - Lower is better
  - Includes data transfer to GPU

- **Throughput:** Images classified per second
  - Reciprocal of inference time
  - Higher is better

- **GPU Memory:** Peak memory during inference (MB)
  - Lower is better for deployment
  - Affects batch sizes possible

---

## 🎯 Practical Examples

### Example 1: Production Model Selection

```python
import pandas as pd

df = pd.read_csv('results/all_results.csv')

# Target: >90% accuracy with <0.02 sec inference time
candidates = df[(df['test_accuracy'] > 0.90) & (df['profile_avg_time_sec'] < 0.02)]
print(candidates[['model', 'resolution', 'test_accuracy', 'profile_avg_time_sec']])
```

### Example 2: Mobile Deployment

```python
# Find smallest model with >85% accuracy
candidates = df[df['test_accuracy'] > 0.85].nsmallest(5, 'profile_gpu_memory_peak_mb')
print(candidates[['model', 'resolution', 'test_accuracy', 'profile_gpu_memory_peak_mb']])
```

### Example 3: Research Report

```bash
# Generate for paper
python scripts/aggregate_results.py --report-only
cat results/detailed_report.txt  # Copy to paper
```

---

**Back:** [TRAINING_EXECUTION.md](TRAINING_EXECUTION.md) - How to run experiments  
**Home:** [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Project overview
