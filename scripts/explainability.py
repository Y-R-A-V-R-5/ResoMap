"""
============================================================
example_explainability.py
============================================================

Quick Start Example: Model Explainability Visualizations

Demonstrates how to generate and visualize model explanations
using Grad-CAM, Integrated Gradients, and Saliency Maps.

Usage:
    python example_explainability.py --checkpoint checkpoints/model.pt --resolution 224
"""

import argparse
import sys
import torch
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import load_model_from_config, get_target_layer_for_gradcam
from src.data import get_data_loaders
from src.explainability import ModelExplainer, batch_explain_and_save, get_target_layer
from src.callbacks import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="ResoMap Explainability Example")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='vgg11', help='Model name (e.g., vgg11, resnet18, mobilenet_v2)')
    parser.add_argument('--resolution', type=int, default=224, help='Image resolution')
    parser.add_argument('--num-classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--data-dir', type=str, default='data/', help='Data directory')
    parser.add_argument('--method', type=str, default='gradcam', 
                       choices=['gradcam', 'integrated_gradients', 'saliency'],
                       help='Explainability method')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--output-dir', type=str, default='explain/', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    
    args = parser.parse_args()
    
    print("="*60)
    print("ResoMap Model Explainability Example")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Method: {args.method}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Setup device
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    
    # Load model from unified config
    print("\nLoading model...")
    try:
        model = load_model_from_config(
            model_name=args.model,
            config_path='configs/models.yaml',
            num_classes=args.num_classes
        )
    except Exception as e:
        print(f"Error loading model '{args.model}': {e}")
        print("\nAvailable models: vgg11, vgg13, vgg16, resnet18, resnet34, resnet50, resnet101,")
        print("                  mobilenet_v2, mobilenet_v2_small, mobilenet_v3_small, mobilenet_v3_large,")
        print("                  efficientnet_b0, efficientnet_b1, efficientnet_b2, simple_cnn, tiny_cnn")
        return
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        checkpoint = load_checkpoint(str(checkpoint_path), model, device=device)
        print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        print(f"Warning: Checkpoint not found: {checkpoint_path}")
        print("Using randomly initialized model...")
    
    model = model.to(device)
    model.eval()
    
    # Load data
    print("\nLoading data...")
    _, _, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        resolution=args.resolution,
        batch_size=args.batch_size,
        num_workers=2,
        augment=False,
        pin_memory=(device.type == 'cuda')
    )
    
    # Get class names (if available)
    class_names = None
    if hasattr(test_loader.dataset, 'classes'):
        class_names = test_loader.dataset.classes
        print(f"Classes: {class_names}")
    
    # Create explainer
    print(f"\nInitializing {args.method} explainer...")
    
    try:
        target_layer = get_target_layer(model, args.model)
        print(f"Target layer: {target_layer}")
    except Exception as e:
        print(f"Warning: Could not automatically determine target layer: {e}")
        print("Using last convolutional stage...")
        # Fallback: use last stage
        if hasattr(model, 'stages'):
            stage_keys = list(model.stages.keys())
            target_layer = model.stages[stage_keys[-1]]
        else:
            print("Error: Cannot determine target layer")
            return
    
    explainer = ModelExplainer(
        model=model,
        device=device,
        target_layer=target_layer,
        class_names=class_names
    )
    
    # Generate explanations
    print(f"\nGenerating {args.method} explanations for {args.num_samples} samples...")
    
    output_dir = Path(args.output_dir) / f"{args.model}_{args.resolution}_{args.method}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sample_count = 0
    
    for batch_idx, (images, labels) in enumerate(test_loader):
        if sample_count >= args.num_samples:
            break
        
        images = images.to(device)
        labels = labels.to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
        
        # Generate explanations
        explanations = explainer.explain_batch(
            images=images,
            labels=labels,
            method=args.method
        )
        
        # Visualize and save
        for i in range(len(images)):
            if sample_count >= args.num_samples:
                break
            
            save_path = output_dir / f"sample_{sample_count:04d}.png"
            
            explainer.visualize_explanation(
                image=images[i],
                explanation=explanations[i],
                pred_class=predictions[i].item(),
                true_class=labels[i].item(),
                save_path=save_path,
                show=False
            )
            
            sample_count += 1
            print(f"Saved: {save_path.name}")
    
    print(f"\n✓ Generated {sample_count} explanations")
    print(f"✓ Saved to: {output_dir}")
    
    # Compare multiple methods (optional)
    print("\n" + "="*60)
    print("Multi-Method Comparison")
    print("="*60)
    
    methods = ['gradcam', 'integrated_gradients', 'saliency']
    comparison_samples = 3
    
    # Get a few samples
    images, labels = next(iter(test_loader))
    images = images[:comparison_samples].to(device)
    labels = labels[:comparison_samples]
    
    with torch.no_grad():
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
    
    # Generate explanations with all methods
    print(f"\nGenerating comparison for {comparison_samples} samples with all methods...")
    
    all_explanations = {}
    for method in methods:
        try:
            print(f"  - {method}...", end=' ')
            explanations = explainer.explain_batch(images, labels, method=method)
            all_explanations[method] = explanations
            print("✓")
        except Exception as e:
            print(f"✗ (Error: {e})")
    
    # Create comparison plot
    if all_explanations:
        fig, axes = plt.subplots(
            comparison_samples, 
            len(all_explanations) + 1, 
            figsize=(5 * (len(all_explanations) + 1), 5 * comparison_samples)
        )
        
        if comparison_samples == 1:
            axes = axes.reshape(1, -1)
        
        for sample_idx in range(comparison_samples):
            # Original image
            img = images[sample_idx].cpu().numpy().transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())
            
            axes[sample_idx, 0].imshow(img)
            axes[sample_idx, 0].set_title("Original Image")
            axes[sample_idx, 0].axis('off')
            
            # Explanations from each method
            for method_idx, (method_name, explanations) in enumerate(all_explanations.items()):
                explanation = explanations[sample_idx]
                
                # Overlay
                import cv2
                h, w = img.shape[:2]
                exp_resized = cv2.resize(explanation, (w, h))
                heatmap = cv2.applyColorMap(np.uint8(255 * exp_resized), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
                overlay = 0.5 * heatmap + 0.5 * img
                
                axes[sample_idx, method_idx + 1].imshow(overlay)
                axes[sample_idx, method_idx + 1].set_title(method_name.upper())
                axes[sample_idx, method_idx + 1].axis('off')
            
            # Add labels
            pred_label = class_names[predictions[sample_idx]] if class_names else predictions[sample_idx].item()
            true_label = class_names[labels[sample_idx]] if class_names else labels[sample_idx].item()
            
            axes[sample_idx, 0].text(
                0.5, -0.1, 
                f"Pred: {pred_label} | True: {true_label}",
                transform=axes[sample_idx, 0].transAxes,
                ha='center',
                fontsize=10
            )
        
        plt.tight_layout()
        comparison_path = output_dir / "method_comparison.png"
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Saved method comparison to: {comparison_path}")
    
    print("\n" + "="*60)
    print("Explainability Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    # Need numpy for cv2 operations
    import numpy as np
    main()