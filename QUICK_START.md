# ResoMap Quick Start Guide

## 🚀 Quick Model Selection Guide

### Choose Your Model Based on Needs:

#### 🎯 **Best for Accuracy**
```python
# ResNet50 or ResNet101
model = load_model_from_config('resnet50')
```
**Why:** Deep residual networks, batch normalization, proven performance

#### ⚡ **Best for Speed**
```python
# MobileNetV2 Small or TinyCNN
model = load_model_from_config('mobilenet_v2_small')
```
**Why:** Efficient depthwise separable convolutions, low parameter count

#### 🔍 **Best for Explainability**
```python
# VGG11 or VGG13
model = load_model_from_config('vgg11')
```
**Why:** Simple architecture, excellent Grad-CAM visualization, interpretable

#### ⚖️ **Best Balance**
```python
# ResNet18 or MobileNetV2
model = load_model_from_config('resnet18')
```
**Why:** Good accuracy, reasonable speed, moderate parameters

---

## 💻 Complete Training Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models import load_model_from_config
from src.trainer import Trainer
from src.callbacks import EarlyStopping, ModelCheckpoint, LRSchedulerCallback

# 1. Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resolution = 224

# 2. Data
transform = transforms.Compose([
    transforms.Resize((resolution, resolution)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                 download=True, transform=transform)
val_dataset = datasets.CIFAR10(root='./data', train=False, 
                               download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, 
                          num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,
                        num_workers=4, pin_memory=True)

# 3. Model
model = load_model_from_config('resnet18', num_classes=10)
model = model.to(device)

# 4. Optimizer & Loss
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
criterion = nn.CrossEntropyLoss()

# 5. Callbacks
early_stop = EarlyStopping(
    patience=10,
    warmup_epochs=5,
    mode='min',
    min_delta=0.001
)

checkpoint = ModelCheckpoint(
    save_dir=f'checkpoints/resnet18_res{resolution}',
    monitor='val_accuracy',
    mode='max',
    save_top_k=3
)

lr_scheduler = LRSchedulerCallback(
    optimizer=optimizer,
    scheduler_type='cosine',
    scheduler_params={'T_max': 50, 'eta_min': 1e-6}
)

# 6. Trainer
trainer = Trainer(
    model=model,
    device=device,
    criterion=criterion,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    use_amp=True,                       # Mixed precision
    gradient_accumulation_steps=2,       # Effective batch size = 64*2 = 128
    gradient_clip_norm=1.0
)

# 7. Training Loop
epochs = 50
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    print("-" * 40)
    
    # Train
    train_metrics = trainer.train_epoch()
    print(f"Train Loss: {train_metrics['loss']:.4f} | "
          f"Acc: {train_metrics['accuracy']:.4f}")
    
    # Validate
    val_metrics = trainer.validate()
    print(f"Val Loss: {val_metrics['loss']:.4f} | "
          f"Acc: {val_metrics['accuracy']:.4f}")
    
    # Callbacks
    early_stop.step(epoch, val_metrics['loss'])
    checkpoint.step(epoch, val_metrics, model, optimizer, resolution=resolution)
    lr_scheduler.step(epoch)
    
    # Early stopping check
    if early_stop.stop:
        print("Early stopping triggered!")
        break

print("\n✓ Training complete!")
print(f"Best checkpoint: {checkpoint.get_best_checkpoint_path()}")
```

---

## 🔄 Resolution Sweep Example

```python
from src.models import load_model_from_config
from src.data import get_data_loaders
import pandas as pd

# Test multiple resolutions
resolutions = [224, 256, 320, 384, 512]
model_names = ['vgg11_224', 'resnet18_224', 'mobilenet_v2_224']

results = []

for model_name in model_names:
    for resolution in resolutions:
        print(f"\nTesting {model_name} @ {resolution}x{resolution}")
        
        # Load model
        model = load_model_from_config(model_name, num_classes=10)
        model = model.to(device)
        
        # Get data loaders
        train_loader, val_loader = get_data_loaders(
            dataset='CIFAR10',
            resolution=resolution,
            batch_size=64,
            num_workers=4
        )
        
        # Train (simplified)
        # ... training code ...
        
        # Store results
        results.append({
            'model': model_name,
            'resolution': resolution,
            'accuracy': final_accuracy,
            'loss': final_loss,
            'training_time': training_time,
            'params': model_params
        })

# Analyze results
df = pd.DataFrame(results)
print("\nResolution Sensitivity Analysis:")
print(df.pivot_table(index='resolution', columns='model', values='accuracy'))
```

---

## 🎨 Explainability Example

```python
from src.explainability import ModelExplainer
from src.models import load_model_from_config, get_target_layer_for_gradcam
import matplotlib.pyplot as plt

# Load trained model
model = load_model_from_config('vgg11', num_classes=10)
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Get target layer for Grad-CAM
target_layer = get_target_layer_for_gradcam(model)

# Create explainer
explainer = ModelExplainer(
    model=model,
    device=device,
    target_layer=target_layer
)

# Load test image
from PIL import Image
img = Image.open('test_image.jpg')
img_tensor = transform(img).unsqueeze(0).to(device)

# Generate explanations
explanations = explainer.explain(
    input_tensor=img_tensor,
    target_class=0,
    methods=['gradcam', 'integrated_gradients', 'saliency']
)

# Visualize
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(img)
axes[0].set_title('Original')

axes[1].imshow(explanations['gradcam'], cmap='jet')
axes[1].set_title('Grad-CAM')

axes[2].imshow(explanations['integrated_gradients'], cmap='hot')
axes[2].set_title('Integrated Gradients')

axes[3].imshow(explanations['saliency'], cmap='gray')
axes[3].set_title('Saliency Map')

plt.tight_layout()
plt.savefig('explanations.png')
```

---

## 📊 Resolution Comparison

```python
from src.explainability import compare_resolutions

# Compare same image at different resolutions
resolutions = [64, 128, 224, 320]
compare_resolutions(
    model=model,
    image=test_image,
    resolutions=resolutions,
    method='gradcam',
    save_path='resolution_comparison.png'
)
```

---

## ⚙️ LR Scheduler Configurations

### Cosine Annealing (Recommended for Most Cases)
```python
lr_scheduler = LRSchedulerCallback(
    optimizer=optimizer,
    scheduler_type='cosine',
    scheduler_params={
        'T_max': 50,        # Total epochs
        'eta_min': 1e-6     # Minimum LR
    }
)
```

### Step Decay (Classic Approach)
```python
lr_scheduler = LRSchedulerCallback(
    optimizer=optimizer,
    scheduler_type='step',
    scheduler_params={
        'step_size': 10,    # Decay every 10 epochs
        'gamma': 0.1        # Multiply by 0.1
    }
)
```

### Reduce on Plateau (Adaptive)
```python
lr_scheduler = LRSchedulerCallback(
    optimizer=optimizer,
    scheduler_type='plateau',
    scheduler_params={
        'mode': 'min',      # 'min' for loss, 'max' for accuracy
        'factor': 0.1,      # Reduce by 10x
        'patience': 5,      # Wait 5 epochs
        'monitor': 'val_loss',
        'min_lr': 1e-7
    }
)

# Use with metrics
lr_scheduler.step(metrics={'val_loss': val_loss})
```

### Multi-Step Decay (Schedule-based)
```python
lr_scheduler = LRSchedulerCallback(
    optimizer=optimizer,
    scheduler_type='multistep',
    scheduler_params={
        'milestones': [30, 60, 90],  # Decay at these epochs
        'gamma': 0.1
    }
)
```

---

## 🔬 Profiling & Performance

```python
from src.profiler import profile_model, print_profiling_report

# Profile model
profile_results = profile_model(
    model=model,
    input_size=(3, 224, 224),
    batch_size=64,
    device=device,
    num_iterations=100
)

# Print detailed report
print_profiling_report(profile_results)

# Output includes:
# - Forward pass latency
# - GPU memory usage
# - Throughput (images/sec)
# - FLOPs count
# - Parameter count
```

---

## 🎯 Model Recommendations by Use Case

### Academic Research / Explainability Study
```python
models = ['vgg11_224', 'vgg13_224', 'resnet18_224']
resolutions = [224, 256, 320, 384, 512]
```
**Why:** Simple architectures, interpretable, good for analysis

### Production Deployment
```python
models = ['mobilenet_v2_224', 'mobilenet_v2_small_224']
resolutions = [224, 256]
```
**Why:** Fast inference, low memory, good accuracy

### Maximum Accuracy
```python
models = ['resnet50_224', 'resnet101_224']
resolutions = [224, 320, 384]
```
**Why:** Deep networks, higher resolutions, state-of-the-art

### Quick Prototyping
```python
models = ['simple_cnn_224', 'tiny_cnn_224']
resolutions = [224, 256]
```
**Why:** Fast training, easy debugging

---

## 🐛 Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size
trainer = Trainer(..., gradient_accumulation_steps=4)  # Effective batch = 64*4

# Or disable AMP
trainer = Trainer(..., use_amp=False)

# Or use smaller model
model = load_model_from_config('mobilenet_v2_small_224')
```

### Slow Training
```python
# Enable optimizations
torch.backends.cudnn.benchmark = True

# Use more workers
train_loader = DataLoader(..., num_workers=8, pin_memory=True, 
                          persistent_workers=True)

# Enable AMP
trainer = Trainer(..., use_amp=True)
```

### Poor Accuracy
```python
# Try different optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Add data augmentation
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Use LR scheduler
lr_scheduler = LRSchedulerCallback(optimizer, 'cosine', {'T_max': 50})
```

---

## 📚 Key Functions Reference

### Model Loading
```python
from src.models import load_model_from_config, build_model, get_model_info

# By name from config
model = load_model_from_config('resnet18', num_classes=10)

# From config dict
model = build_model(model_cfg, num_classes=10)

# Get info
info = get_model_info(model)  # params, size, etc.
```

### Checkpointing
```python
from src.callbacks import load_checkpoint

# Save (done automatically by ModelCheckpoint callback)
checkpoint.step(epoch, metrics, model, optimizer)

# Load
checkpoint_data = load_checkpoint('path/to/checkpoint.pt', model, optimizer, device)
```

### Explainability
```python
from src.explainability import ModelExplainer

explainer = ModelExplainer(model, device, target_layer)
explanations = explainer.explain(img_tensor, target_class, methods=['gradcam'])
```

---

## ✅ Pre-flight Checklist

Before running experiments:

---

## 🎉 Ready to Run!

```bash
# Test everything
python test_models.py

# Single model training
python example_gpu_training.py --model resnet18_224 --resolution 224 --epochs 10

# Full resolution sweep
python scripts/experiments.py

# Explainability analysis
python example_explainability.py --checkpoint checkpoints/resnet/resnet18_224/224/best_model.pt
```

For detailed information, see [VALIDATION_SUMMARY.md](VALIDATION_SUMMARY.md)
