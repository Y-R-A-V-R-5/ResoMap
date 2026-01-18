# Results Export Quick Reference

## One-Liners

| Task | Command |
|------|---------|
| Train and save results | `python scripts/experiments.py` |
| Generate all outputs | `python scripts/aggregate_results.py` |
| Generate only CSV | `python scripts/aggregate_results.py --csv-only` |
| Generate only summary | `python scripts/aggregate_results.py --summary-only` |
| Generate only report | `python scripts/aggregate_results.py --report-only` |
| Custom CSV filename | `python scripts/aggregate_results.py --csv my_results.csv` |
| View results folder | `dir results` |
| View all results | `cat results/detailed_report.txt` |

## Complete Workflow

```bash
# Step 1: Train experiments
python scripts/experiments.py

# Step 2: Generate reports
python scripts/aggregate_results.py

# Step 3: View results
cat results/detailed_report.txt

# Step 4: Analyze in Excel/Python
# Open: results/all_results.csv
```

## File Locations

| File | Location | Auto? | Use |
|------|----------|-------|-----|
| Individual JSON | `results/test_results/MODEL_RES.json` | ✅ Auto | Detailed per-run data |
| Aggregated CSV | `results/all_results.csv` | ❌ Manual | Excel analysis |
| Summary JSON | `results/results_summary.json` | ❌ Manual | Quick stats |
| Report TXT | `results/detailed_report.txt` | ❌ Manual | Human-readable |

## Data Saved per Training

Each training automatically saves:
```json
{
  "model": "model_name",
  "resolution": 224,
  "best_val_loss": 0.5441,
  "test_metrics": {
    "accuracy": 0.8934,
    "f1_score": 0.8867,
    "precision": 0.8901,
    "recall": 0.8834
  },
  "profiling": {
    "avg_time_sec": 0.0231,
    "throughput_samples_sec": 1234.56,
    "gpu_memory_peak_mb": 4096
  }
}
```

## Common Analysis

### Find Best Model
```bash
# In terminal
grep "Best Accuracy:" results/detailed_report.txt | head -10

# In Python
import pandas as pd
df = pd.read_csv('results/all_results.csv')
print(df.nlargest(5, 'test_accuracy')[['model', 'resolution', 'test_accuracy']])
```

### Compare Models
```bash
# In Python
df = pd.read_csv('results/all_results.csv')
df.groupby('model')['test_accuracy'].agg(['mean', 'max', 'min'])
```

### Compare Resolutions
```bash
# In Python
df = pd.read_csv('results/all_results.csv')
df.groupby('resolution')['test_accuracy'].agg(['mean', 'max', 'min'])
```

### Find Fastest Model
```bash
# In Python
df = pd.read_csv('results/all_results.csv')
df.nsmallest(5, 'profile_avg_time_sec')[['model', 'resolution', 'profile_avg_time_sec']]
```

## Arguments Reference

```
scripts/aggregate_results.py [OPTIONS]

Options:
  --csv-only              Generate only CSV
  --summary-only          Generate only summary JSON
  --report-only           Generate only text report
  --csv FILENAME          CSV output name (default: all_results.csv)
  --summary FILENAME      Summary JSON name (default: results_summary.json)
  --report FILENAME       Report text name (default: detailed_report.txt)
  --no-console            Skip console output
  --help                  Show help
```

## Examples

### Example 1: Standard Workflow
```bash
python scripts/experiments.py                    # Train all
python scripts/aggregate_results.py              # Generate all outputs
# Results ready in results/ folder
```

### Example 2: Specific Model Analysis
```bash
python scripts/experiments.py --models vgg11
python scripts/aggregate_results.py --csv-only
# Open results/all_results.csv in Excel
```

### Example 3: Two-Stage Comparison
```bash
# Stage 1: Dense models
python scripts/experiments.py --models vgg11 vgg16
python scripts/aggregate_results.py

# Stage 2: Efficient models
python scripts/experiments.py --models mobilenet_v2 efficientnet_b0
python scripts/aggregate_results.py  # Re-generates with all results
```

### Example 4: Incremental Progress
```bash
# Day 1: Start training
python scripts/experiments.py &  # Background process

# Day 2: Check progress
python scripts/aggregate_results.py  # See what's done

# Day 3: After all complete
cat results/detailed_report.txt
```

## CSV Columns Explained

| Column | Meaning | Example |
|--------|---------|---------|
| model | Model name | vgg11 |
| resolution | Input resolution | 224 |
| timestamp | When trained | 2026-01-18T14:30:45 |
| best_val_loss | Best validation loss | 0.5441 |
| test_loss | Final test loss | 0.5519 |
| test_accuracy | Test accuracy | 0.8934 |
| test_precision | Precision score | 0.8901 |
| test_recall | Recall score | 0.8834 |
| test_f1_score | F1 score | 0.8867 |
| profile_avg_time_sec | Inference time | 0.0231 |
| profile_throughput_samples_sec | Images/sec | 1234.56 |
| profile_gpu_memory_peak_mb | Max GPU memory | 4096 |
| profile_peak_cpu_memory_mb | Max CPU memory | 2048 |

## Analysis in Excel

1. **Open:** `results/all_results.csv`
2. **Pivot Table:**
   - Rows: model
   - Columns: resolution
   - Values: test_accuracy (average)
3. **Create Chart:**
   - X-axis: resolution
   - Y-axis: accuracy
   - Series: model

## Status Check

```bash
# How many results collected?
dir results/test_results | find /c ".json"

# See all results
python scripts/aggregate_results.py --no-console

# View individual result
type results/test_results/vgg11_224.json
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No results saved | Train first: `python scripts/experiments.py` |
| CSV is empty | Check: `dir results/test_results` |
| Report missing | Run: `python scripts/aggregate_results.py` |
| Wrong data | Re-train and re-aggregate |

---

**Full Guide:** [RESULTS_EXPORT_GUIDE.md](RESULTS_EXPORT_GUIDE.md)
**Implementation:** [RESULTS_IMPLEMENTATION_SUMMARY.md](RESULTS_IMPLEMENTATION_SUMMARY.md)
