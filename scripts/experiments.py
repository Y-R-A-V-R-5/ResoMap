"""
============================================================
scripts/experiments.py
============================================================

Experiment Execution Script for ResoMap Training
------------------------------------------------------------

Executable script that triggers full training experiments across
multiple models and resolutions.

This script imports ExperimentRunner from src/ and executes:
1. Model training with validation
2. Early stopping and checkpointing
3. Test set evaluation
4. Model profiling
5. MLflow tracking and artifact logging
6. Dagshub integration

Usage:
    # Run all experiments (all models, all resolutions)
    python scripts/experiments.py
    
    # The script uses configuration from configs/config.yaml
    # Models and resolutions are defined in the sweep section
"""

import sys
from pathlib import Path
import warnings
import dagshub
from tqdm import tqdm

# Show all warnings
warnings.simplefilter("always")

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import experiment runner from src
from src.experiment import ExperimentRunner
from src.utils import load_config


def main():
    """Main execution function for training experiments."""
    
    # Load experiment configs
    config = load_config(PROJECT_ROOT / "configs/config.yaml")
    models_config = load_config(PROJECT_ROOT / "configs/models.yaml")
    
    # Use fixed dataset path
    dataset_path = (PROJECT_ROOT / config["data"]["raw_path"]).resolve()
    
    # Initialize Dagshub MLflow integration
    dagshub.init(repo_owner="Y-R-A-V-R-5", repo_name="ResoMap", mlflow=True)
    
    # Display experiment configuration
    print(f"\n{'='*60}")
    print(f"ResoMap Resolution Training Experiment")
    print(f"Models: {config['sweep']['models']}")
    print(f"Resolutions: {config['sweep']['resolutions']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"{'='*60}\n")
    
    # Create experiment runner
    runner = ExperimentRunner(config, PROJECT_ROOT)
    
    # Sweep over models and resolutions
    # Note: Experiments run SEQUENTIALLY (one after another)
    # Each model trains at all resolutions before moving to the next model
    # This ensures stable GPU memory usage and proper MLflow logging
    for model_name in tqdm(config["sweep"]["models"], desc="Models"):
        # Validate model exists
        if model_name not in models_config:
            print(f"[Warning] Model '{model_name}' not found in models.yaml, skipping...")
            continue
        
        for res in tqdm(
            config["sweep"]["resolutions"],
            desc=f"{model_name} Resolutions",
            leave=False
        ):
            print(f"\n{'='*60}")
            print(f"Training: {model_name} @ {res}x{res}")
            print(f"{'='*60}")
            
            try:
                runner.run_experiment(
                    model_name=model_name,
                    resolution=res,
                    dataset_path=str(dataset_path),
                )
                print(f"✓ Completed: {model_name} @ {res}x{res}")
            except Exception as e:
                print(f"\n[Error] Failed experiment {model_name}@{res}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n{'='*60}")
    print("All Experiments Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
