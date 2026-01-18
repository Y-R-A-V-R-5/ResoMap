# ResoMap Results Export & Aggregation Guide

## Overview

The ResoMap training system now automatically saves all test results in JSON format and provides tools to aggregate them into CSV files and comprehensive reports.

## Features

### 1. **Automatic JSON Export**
After each training completes, results are automatically saved to:
```
results/test_results/
├── vgg11_224.json
├── vgg11_256.json
├── vgg11_320.json
├── resnet18_224.json
└── ...
```

**Each JSON file contains:**
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
    "peak_cpu_memory_mb": 2048,
    "gpu_memory_peak_mb": 4096
  }
}
```

### 2. **CSV Aggregation**
Combine all individual results into a single CSV for easy analysis:
```
results/all_results.csv
```

**Columns include:**
- model
- resolution
- timestamp
- best_val_loss
- test_loss
- test_accuracy
- test_precision
- test_recall
- test_f1_score
- profile_avg_time_sec
- profile_throughput_samples_sec
- profile_peak_cpu_memory_mb
- profile_gpu_memory_peak_mb

### 3. **Summary Statistics**
Generate summary statistics across all models and resolutions:
```
results/results_summary.json
```

Contains:
- Overall statistics
- Per-model averages and best results
- Per-resolution performance
- Top 5 best performing combinations

### 4. **Detailed Report**
Human-readable text report with all analysis:
```
results/detailed_report.txt
```

## Usage

### Automatic Results Saving

Results are saved automatically after each training completes:

```bash
# Train models and automatically save JSON results
python scripts/experiments.py

# After training completes:
# results/test_results/vgg11_224.json ✓
# results/test_results/vgg11_256.json ✓
# ... etc
```

### Generate Aggregated Results

**Generate all outputs (CSV, summary, report):**
```bash
python scripts/aggregate_results.py
```

**Generate only CSV:**
```bash
python scripts/aggregate_results.py --csv-only
```

**Generate only summary statistics:**
```bash
python scripts/aggregate_results.py --summary-only
```

**Generate only detailed report:**
```bash
python scripts/aggregate_results.py --report-only
```

**Custom output filenames:**
```bash
python scripts/aggregate_results.py --csv my_results.csv --summary my_summary.json
```

**Skip console summary:**
```bash
python scripts/aggregate_results.py --no-console
```

## Output Structure

After running aggregation, you'll have:

```
results/
├── test_results/                    # Individual JSON files
│   ├── vgg11_224.json
│   ├── vgg11_256.json
│   ├── vgg11_320.json
│   ├── resnet18_224.json
│   └── ... (8 models × 5 resolutions = 40 files)
│
├── all_results.csv                  # Aggregated CSV
├── results_summary.json             # Summary statistics
└── detailed_report.txt              # Human-readable report
```

## Workflow Examples

### Example 1: Train and Analyze

```bash
# Step 1: Run all training experiments
python scripts/experiments.py

# Step 2: After training completes, aggregate results
python scripts/aggregate_results.py

# Step 3: View console summary
# Output shows:
#   - Top 5 results
#   - Best by model
#   - Best by resolution
```

### Example 2: Incremental Training and Analysis

```bash
# Day 1: Train specific models
python scripts/experiments.py --models vgg11 resnet18

# Aggregate what we have so far
python scripts/aggregate_results.py

# Day 2: Train more models
python scripts/experiments.py --models mobilenet_v2 efficientnet_b0

# Aggregate again (includes all results)
python scripts/aggregate_results.py
```

### Example 3: Compare Resolutions

```bash
# Train at different resolutions and compare
python scripts/experiments.py --resolutions 224 256

# Generate CSV
python scripts/aggregate_results.py --csv-only

# Open all_results.csv in Excel/Sheets
# Compare columns for test_accuracy across resolutions
```

### Example 4: Model Performance Comparison

```bash
# Train all models on one resolution for comparison
python scripts/experiments.py --resolutions 224

# Generate summary
python scripts/aggregate_results.py --summary-only

# View results_summary.json
# Check "by_model" section for model comparison
```

## Understanding the Outputs

### CSV File (all_results.csv)

Perfect for:
- ✅ Opening in Excel, Google Sheets, or Python/Pandas
- ✅ Creating custom charts and visualizations
- ✅ Sorting by accuracy, speed, memory usage
- ✅ Filtering by model or resolution
- ✅ Statistical analysis

**Example Excel Analysis:**
```
Filter by model: VGG11
Sort by: test_accuracy (descending)
Chart: Accuracy vs Resolution
```

### Summary JSON (results_summary.json)

Perfect for:
- ✅ Quick programmatic access to statistics
- ✅ Tracking best results across experiments
- ✅ Monitoring average performance trends
- ✅ Sharing metrics with others

**Example content:**
```json
{
  "statistics": {
    "by_model": {
      "vgg11": {
        "num_experiments": 5,
        "avg_test_accuracy": 0.8834,
        "best_test_accuracy": 0.8956,
        "worst_test_accuracy": 0.8712,
        "avg_test_f1": 0.8867,
        "avg_inference_time_sec": 0.0234
      }
    }
  }
}
```

### Detailed Report (detailed_report.txt)

Perfect for:
- ✅ Printing and sharing in documents
- ✅ Email summaries to stakeholders
- ✅ Including in research papers
- ✅ Quick reference without tools

**Example sections:**
```
TOP 10 BEST RESULTS
VGG11           @ 320px | Accuracy: 0.8956 | F1: 0.8923
ResNet18        @ 320px | Accuracy: 0.8923 | F1: 0.8889
...

RESULTS BY MODEL
VGG11:
  Experiments: 5
  Avg Accuracy: 0.8834
  Best Accuracy: 0.8956 @ 320px
  Avg Inference Time: 0.0234s
```

## Analysis Tips

### Using CSV in Python

```python
import pandas as pd

# Load results
df = pd.read_csv('results/all_results.csv')

# Best overall
print(df.nlargest(5, 'test_accuracy')[['model', 'resolution', 'test_accuracy']])

# By model
df.groupby('model')['test_accuracy'].agg(['mean', 'max', 'min'])

# By resolution
df.groupby('resolution')['test_accuracy'].agg(['mean', 'max', 'min'])

# Speed analysis
df[['model', 'profile_avg_time_sec']].sort_values('profile_avg_time_sec')

# Memory usage
df[['model', 'resolution', 'profile_gpu_memory_peak_mb']].sort_values('profile_gpu_memory_peak_mb')

# Save to different formats
df.to_excel('results/analysis.xlsx')  # Excel
df.to_json('results/data.json')       # JSON
```

### Using CSV in Excel/Sheets

1. **Open:** `results/all_results.csv`
2. **Create Pivot Tables:**
   - Values: test_accuracy (average)
   - Rows: model
   - Columns: resolution
3. **Create Charts:**
   - X-axis: resolution
   - Y-axis: test_accuracy
   - Series: model

### Using Summary JSON in Python

```python
import json

with open('results/results_summary.json') as f:
    summary = json.load(f)

# Get best model performance
for model, stats in summary['statistics']['by_model'].items():
    print(f"{model}: {stats['best_test_accuracy']:.4f}")

# Get performance by resolution
for res, stats in summary['statistics']['by_resolution'].items():
    print(f"{res}px: {stats['avg_test_accuracy']:.4f}")
```

## Integration with Other Tools

### Matplotlib Visualization

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results/all_results.csv')

# Plot: Accuracy by resolution for each model
for model in df['model'].unique():
    model_data = df[df['model'] == model]
    plt.plot(model_data['resolution'], model_data['test_accuracy'], 
             marker='o', label=model)

plt.xlabel('Resolution')
plt.ylabel('Test Accuracy')
plt.legend()
plt.savefig('results/accuracy_vs_resolution.png')
```

### Weights & Biases Integration

```python
import pandas as pd
import wandb

df = pd.read_csv('results/all_results.csv')

wandb.init(project="resomap-results")

for _, row in df.iterrows():
    wandb.log({
        "model": row['model'],
        "resolution": row['resolution'],
        "accuracy": row['test_accuracy'],
        "f1_score": row['test_f1_score']
    })
```

## Troubleshooting

### Q: Results are not being saved
**A:** Check if `results/test_results/` directory exists and has write permissions
```bash
# Results are saved after each training completes
# Check: dir results/test_results
```

### Q: CSV is empty
**A:** No test results found. Train experiments first:
```bash
python scripts/experiments.py --models vgg11 --resolutions 224
python scripts/aggregate_results.py
```

### Q: How do I update results after training more models?
**A:** Simply run aggregation again:
```bash
python scripts/experiments.py --models new_model
python scripts/aggregate_results.py  # Re-generates CSV and summaries
```

### Q: How many rows should the CSV have?
**A:** One row per model-resolution combination:
- 8 models × 5 resolutions = 40 rows (if all completed)
- Fewer rows if training still in progress

## Performance Metrics Explained

| Metric | Meaning | Good Value |
|--------|---------|-----------|
| test_accuracy | Percentage of correct predictions | > 0.85 |
| test_f1_score | Balance of precision/recall | > 0.85 |
| test_precision | False positive rate | > 0.85 |
| test_recall | False negative rate | > 0.85 |
| profile_avg_time_sec | Inference time per image | < 0.05s |
| profile_throughput_samples_sec | Images processed per second | > 100 |
| profile_gpu_memory_peak_mb | Max GPU memory used | < 8000MB |
| profile_peak_cpu_memory_mb | Max CPU memory used | < 4000MB |

## Command Reference

```bash
# Complete workflow
python scripts/experiments.py                    # Train
python scripts/aggregate_results.py              # Aggregate

# Selective training and analysis
python scripts/experiments.py --models vgg11
python scripts/aggregate_results.py --csv-only

# Generate specific outputs
python scripts/aggregate_results.py --summary-only
python scripts/aggregate_results.py --report-only

# Custom filenames
python scripts/aggregate_results.py --csv my_results.csv --summary my_stats.json

# View files
cat results/detailed_report.txt
```

## Summary

| Feature | Status | Location |
|---------|--------|----------|
| Auto JSON export | ✅ Automatic | `results/test_results/*.json` |
| CSV aggregation | ✅ Manual | `results/all_results.csv` |
| Summary stats | ✅ Manual | `results/results_summary.json` |
| Detailed report | ✅ Manual | `results/detailed_report.txt` |
| Console output | ✅ Automatic | Terminal |

---

**Quick Start:**
```bash
python scripts/experiments.py                    # Train
python scripts/aggregate_results.py              # Generate all outputs
# View results in results/ folder
```
