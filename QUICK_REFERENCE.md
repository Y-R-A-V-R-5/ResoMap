# ResoMap Quick Reference Card

## One-Liners

| Task | Command |
|------|---------|
| Run all experiments (40 combos) | `python scripts/experiments.py` |
| Train VGG11 on all resolutions | `python scripts/experiments.py --models vgg11` |
| Train all models on 224 resolution | `python scripts/experiments.py --resolutions 224` |
| Train VGG11 & ResNet18 on 224 & 320 | `python scripts/experiments.py --models vgg11 resnet18 --resolutions 224 320` |
| Resume failed run (VGG11@320) | `python scripts/experiments.py --models vgg11 --resolutions 320` |
| Test locally (skip DagsHub check) | `python scripts/experiments.py --skip-dagshub-check` |

## Workflow Examples

### Workflow 1: Large-Scale Experiment
```bash
# Day 1: Start all 40 experiments
python scripts/experiments.py

# If one fails, e.g., VGG11@320:
# Day 2: Resume the failed one
python scripts/experiments.py --models vgg11 --resolutions 320

# Continue with remaining unfinished runs
python scripts/experiments.py  # Skips completed ones automatically
```

### Workflow 2: Iterative Model Comparison
```bash
# Test only VGG11 thoroughly
python scripts/experiments.py --models vgg11

# Test only ResNet18
python scripts/experiments.py --models resnet18

# Test only MobileNet
python scripts/experiments.py --models mobilenet_v2
```

### Workflow 3: Resolution Sensitivity Analysis
```bash
# Compare performance across all models at 224
python scripts/experiments.py --resolutions 224

# Then at 320 (after analyzing 224 results)
python scripts/experiments.py --resolutions 320
```

### Workflow 4: Distributed Training (2 Machines)
```bash
# Machine 1 - Train dense models
python scripts/experiments.py --models vgg11 vgg16 resnet18 --skip-dagshub-check

# Machine 2 - Train lightweight models
python scripts/experiments.py --models mobilenet_v2 efficientnet_b0 squeezenet_1_1 --skip-dagshub-check

# Later - Consolidate on one machine
python scripts/experiments.py  # Auto-skips all completed runs
```

## Key Features

| Feature | How to Use |
|---------|-----------|
| **Selective Models** | `--models vgg11 resnet18` |
| **Selective Resolutions** | `--resolutions 224 320` |
| **Skip DagsHub Check** | `--skip-dagshub-check` |
| **Resume Failed Run** | `--models MODEL --resolutions RES` |
| **Auto-Skip Completed** | Run normally (default behavior) |

## Checkpoint Locations

```
checkpoints/
├── vgg/vgg11/{224,256,320,384,512}/{best_model.pt, final_model.pt}
├── resnet/resnet18/{224,256,320,384,512}/{best_model.pt, final_model.pt}
├── mobilenet/mobilenet_v2/{224,256,320,384,512}/{best_model.pt, final_model.pt}
└── ...8 models × 5 resolutions
```

## Status Messages

| Message | Meaning | Action |
|---------|---------|--------|
| `[Skip] model@res ✓ (completed in DagsHub)` | Already done successfully | None - skipped |
| `[Info] model@res has local checkpoint, may resume` | Exists locally | Will continue training |
| `[Success] model@res completed ✓` | Training finished successfully | None - proceeds to next |
| `[Error] model@res failed: ERROR` | Training failed | Resume with selective args |

## Common Issues & Solutions

| Problem | Solution |
|---------|----------|
| "CUDA out of memory at 512" | `--resolutions 224 256 320 384` (skip 512) |
| "Need to run only one model" | `--models vgg11` |
| "Want to retry one resolution" | `--models model --resolutions 320` |
| "Distributed training" | Use `--skip-dagshub-check` on each machine |
| "Don't know what's left" | Run normally to see skipped runs + summary |

## Arguments Syntax

```
--models MODEL1 MODEL2 ...      # Space-separated model names
--resolutions 224 256 320 ...   # Space-separated resolution numbers
--skip-dagshub-check            # No value needed (boolean flag)
```

## Examples of All Models & Resolutions

**Available Models:**
- vgg11, vgg16
- resnet18, resnet50
- mobilenet_v2
- efficientnet_b0
- densenet121
- squeezenet_1_1

**Available Resolutions:**
- 224, 256, 320, 384, 512

---

**For full documentation:** See `EXECUTION_GUIDE.md`
