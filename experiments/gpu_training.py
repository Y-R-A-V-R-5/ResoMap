"""
============================================================
example_gpu_training.py
============================================================

Quick Start Example: GPU-Accelerated Training with Explainability

This script demonstrates how to use the enhanced ResoMap features:
- GPU-optimized training
- Mixed precision (AMP)
- Model checkpointing
- Early stopping
- Learning rate scheduling
- Performance monitoring

Usage:
    python example_gpu_training.py --resolution 224 --model resnet18 --epochs 10
    
Available models:
    VGG: vgg11, vgg13, vgg16
    ResNet: resnet18, resnet34, resnet50, resnet101
    MobileNet: mobilenet_v2, mobilenet_v2_small, mobilenet_v3_small, mobilenet_v3_large
    EfficientNet: efficientnet_b0, efficientnet_b1, efficientnet_b2
    Simple: simple_cnn, tiny_cnn
"""

import argparse
import sys
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Import ResoMap modules
from src.models import load_model_from_config, get_model_info
from src.data import get_data_loaders
from src.trainer import Trainer
from src.callbacks import EarlyStopping, ModelCheckpoint, LRSchedulerCallback
from src.profiler import profile_model, get_model_complexity, print_profiling_report


def main():
    parser = argparse.ArgumentParser(description="ResoMap GPU Training Example")
    parser.add_argument('--resolution', type=int, default=224, help='Image resolution')
    parser.add_argument('--model', type=str, default='resnet18', help='Model name (e.g., vgg11, resnet18, mobilenet_v2)')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--data-dir', type=str, default='data/', help='Data directory')
    parser.add_argument('--num-classes', type=int, default=7, help='Number of classes')
    parser.add_argument('--scheduler', type=str, default='cosine', 
                       choices=['cosine', 'step', 'plateau', 'multistep', 'exponential'], 
                       help='LR scheduler type')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    
    args = parser.parse_args()
    
    print("="*60)
    print("ResoMap GPU-Accelerated Training Example")
    print("="*60)
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    
    # Setup device
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Enable cuDNN benchmark
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    # Load model from unified config
    print("\nBuilding model...")
    try:
        model = load_model_from_config(
            model_name=args.model,
            config_path='configs/models.yaml',
            num_classes=args.num_classes
        )
        model = model.to(device)
    except Exception as e:
        print(f"Error loading model '{args.model}': {e}")
        print("\nAvailable models: vgg11, vgg13, vgg16, resnet18, resnet34, resnet50, resnet101,")
        print("                  mobilenet_v2, mobilenet_v2_small, mobilenet_v3_small, mobilenet_v3_large,")
        print("                  efficientnet_b0, efficientnet_b1, efficientnet_b2, simple_cnn, tiny_cnn")
        return
    
    # Model info
    model_info = get_model_info(model)
    print(f"✓ Model loaded: {args.model}")
    print(f"  Total Parameters: {model_info['params_millions']:.2f}M")
    print(f"  Model Size: {model_info['model_size_mb']:.2f} MB")
    
    # Model complexity (FLOPs)
    try:
        complexity = get_model_complexity(
            model,
            input_shape=(1, 3, args.resolution, args.resolution),
            device=device
        )
        print(f"  GFLOPs: {complexity['gflops']:.2f}")
    except Exception as e:
        print(f"  Could not compute FLOPs: {e}")
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        resolution=args.resolution,
        batch_size=args.batch_size,
        num_workers=4,
        augment=True,
        pin_memory=(device.type == 'cuda'),
        prefetch_factor=2,
        persistent_workers=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler using callback
    scheduler_params = {
        'cosine': {'T_max': args.epochs, 'eta_min': 1e-6},
        'step': {'step_size': 10, 'gamma': 0.1},
        'plateau': {'mode': 'min', 'factor': 0.1, 'patience': 5, 'monitor': 'val_loss'},
        'multistep': {'milestones': [int(args.epochs*0.5), int(args.epochs*0.75)], 'gamma': 0.1},
        'exponential': {'gamma': 0.95}
    }
    
    lr_scheduler = LRSchedulerCallback(
        optimizer=optimizer,
        scheduler_type=args.scheduler,
        scheduler_params=scheduler_params[args.scheduler],
        verbose=True
    )
    
    # Trainer with GPU optimizations
    use_amp = (device.type == 'cuda') and (not args.no_amp)
    
    trainer = Trainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        use_amp=use_amp,
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0
    )
    
    print(f"\nTraining Configuration:")
    print(f"  Mixed Precision: {'Enabled' if use_amp else 'Disabled'}")
    print(f"  LR Scheduler: {args.scheduler}")
    print(f"  Initial LR: {args.lr}")
    print(f"  Batch Size: {args.batch_size}")
    
    # Setup callbacks
    early_stopping = EarlyStopping(
        patience=5,
        warmup_epochs=3,
        mode='min',
        min_delta=0.001
    )
    
    checkpoint = ModelCheckpoint(
        save_dir=f'checkpoints/{args.model}_{args.resolution}',
        monitor='val_accuracy',
        mode='max',
        save_top_k=3,
        verbose=True
    )
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60 + "\n")
    
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_metrics = trainer.train_epoch()
        
        # Validate
        val_metrics = trainer.validate()
        
        # Scheduler step (use metrics for plateau scheduler)
        if args.scheduler == 'plateau':
            lr_scheduler.step(metrics=val_metrics)
        else:
            lr_scheduler.step(epoch=epoch)
        
        # Get current learning rate
        current_lr = lr_scheduler.get_last_lr()[0]

        # Print metrics (include precision/recall/f1 when available)
        print(
            f"\nTrain Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics.get('accuracy', 0.0):.4f} | "
            f"Prec: {train_metrics.get('precision', 0.0):.4f} | "
            f"Recall: {train_metrics.get('recall', 0.0):.4f} | "
            f"F1: {train_metrics.get('f1', 0.0):.4f}"
        )
        print(
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics.get('accuracy', 0.0):.4f} | "
            f"Val Prec: {val_metrics.get('precision', 0.0):.4f} | "
            f"Val Recall: {val_metrics.get('recall', 0.0):.4f} | "
            f"Val F1: {val_metrics.get('f1', 0.0):.4f}"
        )
        print(f"LR: {current_lr:.6f}")
        
        if 'epoch_time' in train_metrics:
            print(f"Epoch Time: {train_metrics['epoch_time']:.2f}s")
        
        if 'max_gpu_memory_gb' in train_metrics:
            print(f"GPU Memory: {train_metrics['max_gpu_memory_gb']:.2f} GB")
        
        # Prepare metrics prefixed for checkpointing and LR schedulers
        ckpt_metrics = {
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics.get('accuracy', 0.0),
            'val_precision': val_metrics.get('precision', 0.0),
            'val_recall': val_metrics.get('recall', 0.0),
            'val_f1': val_metrics.get('f1', 0.0),
        }

        # Save checkpoint (monitor expects keys like 'val_accuracy' or 'val_loss')
        checkpoint.step(
            epoch=epoch,
            metrics=ckpt_metrics,
            model=model,
            optimizer=optimizer,
            resolution=args.resolution,
            additional_info={'train_metrics': train_metrics}
        )
        
        # Early stopping
        early_stopping.step(epoch, val_metrics['loss'])
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            print(f"✓ New best validation accuracy: {best_val_acc:.4f}")
        
        if early_stopping.stop:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break
    
    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation on Test Set")
    print("="*60 + "\n")
    
    model.eval()
    test_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    test_acc = accuracy_score(all_labels, all_preds)
    test_loss_avg = test_loss / len(test_loader)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    print(f"Test Loss: {test_loss_avg:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision (macro): {precision:.4f} | Test Recall (macro): {recall:.4f} | Test F1 (macro): {f1:.4f}")

    print("\nClassification Report:")
    # Use unique labels from actual predictions to ensure correctmnumber of target names
    unique_labels = sorted(set(all_labels) | set(all_preds))
    target_names = [f"Class {i}" for i in unique_labels]
    print(classification_report(all_labels, all_preds, labels=unique_labels, target_names=target_names, zero_division=0))
    
    # Profile model performance
    print("\n" + "="*60)
    print("Model Profiling")
    print("="*60 + "\n")
    
    try:
        prof_results = profile_model(
            model=model,
            device=device,
            loader=test_loader,
            warmup=5,
            runs=10,
            batch_size=args.batch_size
        )
        
        print_profiling_report(prof_results, title=f"{args.model} @ {args.resolution}x{args.resolution}")
    except Exception as e:
        print(f"Could not profile model: {e}")
    
    # Performance summary
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    
    try:
        perf_stats = trainer.get_performance_stats()
        print(f"Average Epoch Time: {perf_stats.get('avg_epoch_time', 0):.2f}s")
        print(f"Total Training Time: {perf_stats.get('total_training_time', 0):.2f}s")
        
        if 'avg_gpu_memory_gb' in perf_stats:
            print(f"Average GPU Memory: {perf_stats['avg_gpu_memory_gb']:.2f} GB")
            print(f"Peak GPU Memory: {perf_stats['max_gpu_memory_gb']:.2f} GB")
    except Exception as e:
        print(f"Performance stats not available: {e}")
    
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    
    # Save checkpoint metadata
    checkpoint.save_metadata()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Checkpoints saved to: checkpoints/{args.model}_{args.resolution}/")
    print("="*60)


if __name__ == "__main__":
    main()