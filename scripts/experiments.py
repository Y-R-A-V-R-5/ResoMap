"""
experiments.py - Full Experiment Runner for ResoMap

Handles:
    - Training and validation loops
    - Performance metrics logging
    - Early stopping
    - Model profiling (latency and memory)
    - MLflow experiment tracking
    - Integration with Dagshub for experiment management

Supports CPU/GPU training and multi-resolution model sweeps.
"""

import sys
from pathlib import Path
import warnings  # <-- Added for warnings
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import dagshub

# Show all warnings
warnings.simplefilter("always")

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import project modules
from src.data import get_data_loaders
from src.models import build_model
from src.trainer import Trainer
from src.profiler import profile_model
from src.utils import load_config
from src.callbacks import EarlyStopping


def run_experiment(resolution: int, model_name: str, model_file: str, config: dict, dataset_path: str):
    """
    Run a full training experiment for a single model and resolution.

    Parameters
    ----------
    resolution : int
        Input resolution for images.
    model_name : str
        Name of the model to train.
    model_file : str
        Model configuration file name.
    config : dict
        Full experiment configuration dictionary.
    dataset_path : str
        Path to dataset directory.
    """
    print(f"\n[ResoMap] Starting experiment → model={model_name}, resolution={resolution}")
    print(f"Dataset path: {dataset_path}")

    # -----------------------------
    # Load configuration sections
    # -----------------------------
    data_cfg = config["data"]
    training_cfg = config["training"]
    augment_cfg = config.get("augmentation", {})
    system_cfg = config["system"]

    # -----------------------------
    # Create DataLoaders
    # -----------------------------
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=dataset_path,
        resolution=resolution,
        batch_size=training_cfg["batch_size"],
        num_workers=data_cfg.get("num_workers", 2),
        augment=augment_cfg.get("horizontal_flip", False)
    )

    # -----------------------------
    # Build model, criterion, optimizer
    # -----------------------------
    model_cfg_path = PROJECT_ROOT / "configs" / model_file
    model_cfg = load_config(model_cfg_path)
    model = build_model(model_cfg, resolution)
    device = torch.device(system_cfg.get("device", "cpu"))
    model.to(device)

    criterion = nn.CrossEntropyLoss()  # Classification loss
    optimizer = optim.Adam(model.parameters(), lr=training_cfg["learning_rate"])

    # Wrap model in Trainer utility for training/validation
    trainer = Trainer(model, device, criterion, optimizer, train_loader, val_loader)

    # -----------------------------
    # Early stopping
    # -----------------------------
    es_cfg = training_cfg.get("early_stopping", {})
    early_stopping = EarlyStopping(
        patience=es_cfg.get("patience", 5),
        warmup_epochs=training_cfg.get("warmup_epochs", 3),
        mode="min", # Monitor validation loss
        min_delta=es_cfg.get("min_delta", 0.0),
    )
    early_stopping_enabled = es_cfg.get("enabled", False)

    # -----------------------------
    # Model checkpoint path
    # -----------------------------
    model_dir = PROJECT_ROOT / "models" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_dir / f"res_{resolution}_best.pt"

    # -----------------------------
    # MLflow
    # -----------------------------
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=f"{model_name}_{resolution}"):

        mlflow.log_params({
            "model": model_name,
            "resolution": resolution,
            "batch_size": training_cfg["batch_size"],
            "learning_rate": training_cfg["learning_rate"],
            "epochs": training_cfg["epochs"],
        })

        # -----------------------------
        # Training loop
        # -----------------------------
        for epoch in range(training_cfg["epochs"]):
            print(f"\nEpoch {epoch + 1}/{training_cfg['epochs']}")

            train_metrics = trainer.train_epoch()
            val_metrics = trainer.validate()

            print("Train →", train_metrics)
            print("Val   →", val_metrics)

            mlflow.log_metrics(
                {f"train_{k}": v for k, v in train_metrics.items()}, step=epoch
            )
            mlflow.log_metrics(
                {f"val_{k}": v for k, v in val_metrics.items()}, step=epoch
            )

            if early_stopping_enabled:
                early_stopping.step(epoch, val_metrics["loss"])

                if early_stopping.is_best:
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "model": model_name,
                            "resolution": resolution,
                            "epoch": epoch,
                            "val_loss": val_metrics["loss"],
                        },
                        best_model_path,
                    )
                    mlflow.log_artifact(str(best_model_path), artifact_path="best_models")
                    print(f"[ResoMap] ✔ Saved best model → {best_model_path}")

                if early_stopping.stop:
                    print(f"[ResoMap] Early stopping at epoch {epoch + 1}")
                    break

        # -----------------------------
        # Model Profiling
        # -----------------------------
        print("[ResoMap] Profiling model...")
        latency, peak_mem = profile_model(
            model,
            device,
            test_loader,
            warmup=system_cfg.get("warmup_runs", 5),
            runs=system_cfg.get("num_profiling_runs", 30),
            track_activation_memory=system_cfg.get("track_activation_memory", False),
        )

        mlflow.log_metrics({
            "cpu_latency_sec": latency,
            "peak_ram_mb": peak_mem,
        })

        print(f"[ResoMap] Latency={latency:.6f}s | Peak RAM={peak_mem:.2f} MB")

    print(f"[ResoMap] Finished → model={model_name}, resolution={resolution}")

# -----------------------------
# CLI entry point for sweeps
# -----------------------------

if __name__ == "__main__":

    # Load experiment configs
    config = load_config(PROJECT_ROOT / "configs/config.yaml")
    models_mapping = load_config(PROJECT_ROOT / "configs/models.yaml")
    
    # Use fixed dataset path
    dataset_path = (PROJECT_ROOT / config["data"]["raw_path"]).resolve()

    # Initialize Dagshub MLflow integration
    dagshub.init(repo_owner="Y-R-A-V-R-5", repo_name="ResoMap", mlflow=True)

    # Sweep over models and resolutions
    for model_name in config["sweep"]["models"]:

        # Skip invalid model names
        key = model_name.lower()
        
        if key not in models_mapping:
            print(f"[Warning] Unknown model: {model_name}")
            continue

        model_file = models_mapping[key]["file"]

        for res in config["sweep"]["resolutions"]:
            run_experiment(
                resolution=res,
                model_name=model_name,
                model_file=model_file,
                config=config,
                dataset_path=str(dataset_path),
            )


