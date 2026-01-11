"""
Quick test script to verify model loading and architecture.

This script tests:
1. Loading different model families from configs/models.yaml
2. Model architecture correctness
3. Forward pass with different resolutions
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.models import load_model_from_config, get_model_info
    import torch
    
    print("=" * 70)
    print("ResoMap Model Architecture Test")
    print("=" * 70)
    
    # Test models from different families
    test_models = [
        'vgg11',
        'vgg13', 
        'resnet18',
        'resnet50',
        'mobilenet_v2',
        'simple_cnn'
    ]
    
    resolutions = [224, 256, 320, 384, 512]
    
    for model_name in test_models:
        print(f"\n{'='*70}")
        print(f"Testing: {model_name.upper()}")
        print(f"{'='*70}")
        
        try:
            # Load model
            model = load_model_from_config(model_name)
            
            # Get model info
            info = get_model_info(model)
            print(f"✓ Model loaded successfully")
            print(f"  - Total parameters: {info['total_params']:,}")
            print(f"  - Trainable parameters: {info['trainable_params']:,}")
            print(f"  - Parameters (millions): {info['params_millions']:.2f}M")
            print(f"  - Model size: {info['model_size_mb']:.2f} MB")
            
            # Test forward pass with different resolutions
            print(f"\n  Testing forward pass:")
            model.eval()
            
            for res in resolutions:
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, res, res)
                    output = model(dummy_input)
                    print(f"    Resolution {res}x{res}: Output shape {tuple(output.shape)} ✓")
            
        except Exception as e:
            print(f"✗ Error testing {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("Test Complete!")
    print(f"{'='*70}")
    
except ImportError as e:
    print(f"Import Error: {e}")
    print("\nPlease install required packages:")
    print("  pip install torch torchvision pyyaml")
    sys.exit(1)
