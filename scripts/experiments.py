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
7. Resume capability for failed experiments
8. Selective model/resolution execution via CLI

Usage:
    # Run all experiments (all models, all resolutions)
    python scripts/experiments.py
    
    # Run only specific models on all resolutions
    python scripts/experiments.py --models vgg11 resnet18
    
    # Run all models on specific resolutions
    python scripts/experiments.py --resolutions 224 320
    
    # Run specific models on specific resolutions
    python scripts/experiments.py --models vgg11 --resolutions 224 320
    
    Options:
      --models MODEL [MODEL ...]: Specific models to train (default: all from config)
      --resolutions RES [RES ...]: Specific resolutions to train (default: all from config)
      --skip-dagshub-check: Skip checking DagsHub for completed runs (default: check)
"""

import sys
import argparse
from pathlib import Path
import warnings
import dagshub
import mlflow
from tqdm import tqdm

# Show all warnings
warnings.simplefilter("always")

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import experiment runner from src
from src.experiment import ExperimentRunner
from src.sweep import ExperimentSweep
from src.utils import load_config


def get_dagshub_completed_runs():
    """
    Query DagsHub/MLflow to get all completed runs.
    
    Returns
    -------
    set
        Set of tuples (model_name, resolution) that have been completed
    """
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("ResoMap")
        
        if experiment is None:
            print("[Info] No 'ResoMap' experiment found in MLflow yet")
            return set()
        
        completed_runs = set()
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        
        for run in runs:
            # Extract model name and resolution from run name or tags
            run_name = run.data.tags.get("mlflow.runName", "")
            
            if run_name and "_" in run_name:
                parts = run_name.rsplit("_", 1)
                if len(parts) == 2:
                    model_name, res_str = parts
                    try:
                        resolution = int(res_str)
                        # Only consider successful runs
                        if run.info.status == "FINISHED":
                            completed_runs.add((model_name, resolution))
                    except ValueError:
                        pass
        
        return completed_runs
    
    except Exception as e:
        print(f"[Warning] Could not query MLflow runs: {e}")
        return set()


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ResoMap Training Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run all models and resolutions:
    python scripts/experiments.py
  
  Run only VGG11 on all resolutions:
    python scripts/experiments.py --models vgg11
  
  Run all models only on 224 resolution:
    python scripts/experiments.py --resolutions 224
  
  Run specific models on specific resolutions:
    python scripts/experiments.py --models vgg11 resnet18 --resolutions 224 320
  
  Skip DagsHub checking (useful for local testing):
    python scripts/experiments.py --skip-dagshub-check
        """
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=None,
        help="Specific models to train (e.g., vgg11 resnet18). Default: all from config"
    )
    
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=None,
        help="Specific resolutions to train (e.g., 224 320 384). Default: all from config"
    )
    
    parser.add_argument(
        "--skip-dagshub-check",
        action="store_true",
        help="Skip checking DagsHub for completed runs"
    )
    
    return parser.parse_args()


def main():
    """Main execution function for training experiments."""
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Load experiment configs
    config = load_config(PROJECT_ROOT / "configs/config.yaml")
    models_config = load_config(PROJECT_ROOT / "configs/models.yaml")
    
    # Use fixed dataset path
    dataset_path = (PROJECT_ROOT / config["data"]["raw_path"]).resolve()
    
    # Initialize Dagshub MLflow integration
    dagshub.init(repo_owner="Y-R-A-V-R-5", repo_name="ResoMap", mlflow=True)
    
    # Determine models and resolutions to run
    all_models = config["sweep"]["models"]
    all_resolutions = config["sweep"]["resolutions"]
    
    # Apply CLI filters
    models_to_run = args.models if args.models else all_models
    resolutions_to_run = args.resolutions if args.resolutions else all_resolutions
    
    # Validate specified models
    invalid_models = [m for m in models_to_run if m not in models_config]
    if invalid_models:
        print(f"[Error] Invalid models: {invalid_models}")
        print(f"Available models: {list(models_config.keys())}")
        return
    
    # Display experiment configuration
    print(f"\n{'='*60}")
    print(f"ResoMap Resolution Training Experiment")
    print(f"Models: {models_to_run}")
    print(f"Resolutions: {resolutions_to_run}")
    print(f"Epochs: {config['training']['epochs']}")
    
    # Check DagsHub for completed runs
    completed_runs = set()
    if not args.skip_dagshub_check:
        print(f"Checking DagsHub for completed runs...")
        completed_runs = get_dagshub_completed_runs()
        if completed_runs:
            print(f"Found {len(completed_runs)} completed run(s) in DagsHub")
            for model, res in sorted(completed_runs):
                print(f"  ✓ {model}@{res}")
    else:
        print("DagsHub check skipped (--skip-dagshub-check)")
    
    print(f"{'='*60}\n")
    
    # Create experiment runner
    runner = ExperimentRunner(config, PROJECT_ROOT)

    # Create sweep controller
    sweep = ExperimentSweep(
        runner=runner,
        config=config,
        models_config=models_config,
        project_root=PROJECT_ROOT,
    )

    # Run sweep with selective execution
    sweep.run(
        dataset_path=str(dataset_path),
        models=models_to_run,
        resolutions=resolutions_to_run,
        completed_runs=completed_runs,
    )
    
    print(f"\n{'='*60}")
    print("All Experiments Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()