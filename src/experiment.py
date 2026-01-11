"""
============================================================
src/experiment_runner.py
============================================================

Experiment execution module for ResoMap training experiments.

Contains:
1. Test set evaluation utilities
2. ExperimentRunner - Complete training experiment orchestration

This module provides reusable experiment running functionality
with MLflow tracking, checkpointing, and comprehensive metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, Any

from .data import get_data_loaders
from .models import build_model
from .trainer import Trainer
from .profiler import profile_model
from .utils import load_config
from .callbacks import EarlyStopping


def get_model_family(model_name: str) -> str:
    """
    Determine model family from model name.
    
    Parameters
    ----------
    model_name : str
        Model name (e.g., 'vgg11', 'resnet18', 'mobilenet_v2')
        
    Returns
    -------
    str
        Model family name (e.g., 'vgg', 'resnet', 'mobilenet')
    """
    if model_name.startswith('vgg'):
        return 'vgg'
    elif model_name.startswith('resnet'):
        return 'resnet'
    elif model_name.startswith('mobilenet'):
        return 'mobilenet'
    elif model_name.startswith('efficientnet'):
        return 'efficientnet'
    elif 'cnn' in model_name.lower():
        return 'custom_cnn'
    else:
        return 'other'


def evaluate_test_set(model, test_loader, device, criterion):
    """
    Evaluate model on test set and return comprehensive metrics.
    
    Parameters
    ----------
    model : nn.Module
        Model to evaluate
    test_loader : DataLoader
        Test data loader
    device : torch.device
        Device to run evaluation on
    criterion : nn.Module
        Loss function
        
    Returns
    -------
    dict
        Dictionary containing test metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    avg_loss = total_loss / len(test_loader)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


class ExperimentRunner:
    """
    Comprehensive experiment runner for model training and evaluation.
    
    Handles:
    - Model building and initialization
    - Training loop with validation
    - Early stopping
    - Checkpointing (best and final models)
    - Test set evaluation
    - Model profiling
    - MLflow tracking and artifact logging
    """
    
    def __init__(self, config: Dict[str, Any], project_root: Path):
        """
        Initialize experiment runner.
        
        Parameters
        ----------
        config : dict
            Full configuration dictionary
        project_root : Path
            Path to project root directory
        """
        self.config = config
        self.project_root = Path(project_root)
        
        # Extract config sections
        self.data_cfg = config["data"]
        self.training_cfg = config["training"]
        self.augment_cfg = config.get("augmentation", {})
        self.system_cfg = config["system"]
        self.mlflow_cfg = config.get("mlflow", {})
        
        # Load models config
        models_config_path = self.project_root / "configs" / "models.yaml"
        self.models_config = load_config(models_config_path)
    
    def run_experiment(
        self,
        model_name: str,
        resolution: int,
        dataset_path: str
    ) -> Dict[str, Any]:
        """
        Run a full training experiment for a single model and resolution.
        
        Parameters
        ----------
        model_name : str
            Name of the model to train
        resolution : int
            Input resolution for images
        dataset_path : str
            Path to dataset directory
            
        Returns
        -------
        dict
            Dictionary containing experiment results
        """
        print(f"\n[ResoMap] Starting experiment → model={model_name}, resolution={resolution}")
        print(f"Dataset path: {dataset_path}")
        
        # Validate model exists
        if model_name not in self.models_config:
            raise ValueError(f"Model '{model_name}' not found in models.yaml")
        
        # -----------------------------
        # Create DataLoaders
        # -----------------------------
        train_loader, val_loader, test_loader = get_data_loaders(
            data_dir=dataset_path,
            resolution=resolution,
            batch_size=self.training_cfg["batch_size"],
            num_workers=self.data_cfg.get("num_workers", 2),
            augment=self.augment_cfg.get("horizontal_flip", False)
        )
        
        # -----------------------------
        # Build model, criterion, optimizer
        # -----------------------------
        model_cfg = self.models_config[model_name]
        model = build_model(model_cfg, resolution)
        device = torch.device(self.system_cfg.get("device", "cpu"))
        model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.training_cfg["learning_rate"]
        )
        
        # Create trainer
        trainer = Trainer(model, device, criterion, optimizer, train_loader, val_loader)
        
        # -----------------------------
        # Early stopping setup
        # -----------------------------
        es_cfg = self.training_cfg.get("early_stopping", {})
        early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 5),
            warmup_epochs=self.training_cfg.get("warmup_epochs", 3),
            mode="min",
            min_delta=es_cfg.get("min_delta", 0.0),
        )
        early_stopping_enabled = es_cfg.get("enabled", False)
        
        # -----------------------------
        # Checkpoint paths with hierarchical structure
        # Format: checkpoints/{model_family}/{model_name}/{resolution}/
        # Example: checkpoints/vgg/vgg11/224/best_model.pt
        # -----------------------------
        model_family = get_model_family(model_name)
        checkpoint_dir = (
            self.project_root / "checkpoints" / 
            model_family / model_name / str(resolution)
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = checkpoint_dir / "best_model.pt"
        final_model_path = checkpoint_dir / "final_model.pt"
        
        print(f"\nCheckpoint directory: {checkpoint_dir}")
        
        # -----------------------------
        # MLflow tracking
        # -----------------------------
        mlflow.set_experiment(self.mlflow_cfg.get("experiment_name", "ResoMap"))
        
        with mlflow.start_run(run_name=f"{model_name}_{resolution}"):
            
            # Log parameters
            self._log_parameters(
                model, model_name, resolution, device,
                early_stopping_enabled, es_cfg
            )
            
            # Training loop
            best_val_loss = self._training_loop(
                trainer, model, model_name, resolution,
                best_model_path, early_stopping,
                early_stopping_enabled
            )
            
            # Save final model
            self._save_final_model(
                model, model_name, resolution,
                final_model_path, val_metrics=trainer.last_val_metrics
            )
            
            # Log model artifacts
            mlflow.log_artifact(str(best_model_path), artifact_path="models")
            mlflow.log_artifact(str(final_model_path), artifact_path="models")
            
            # Test evaluation
            test_metrics = self._evaluate_test_set(
                model, test_loader, device, criterion
            )
            
            # Model profiling
            prof_results = self._profile_model(
                model, device, test_loader
            )
            
            # Log summary
            self._print_summary(
                model_name, resolution, best_val_loss,
                test_metrics, prof_results
            )
            
            results = {
                'model': model_name,
                'resolution': resolution,
                'best_val_loss': best_val_loss,
                'test_metrics': test_metrics,
                'profiling': prof_results
            }
        
        print(f"[ResoMap] Finished → model={model_name}, resolution={resolution}")
        return results
    
    def _log_parameters(
        self, model, model_name, resolution, device,
        early_stopping_enabled, es_cfg
    ):
        """Log experiment parameters to MLflow."""
        mlflow.log_params({
            "model": model_name,
            "resolution": resolution,
            "batch_size": self.training_cfg["batch_size"],
            "learning_rate": self.training_cfg["learning_rate"],
            "epochs": self.training_cfg["epochs"],
            "optimizer": self.training_cfg.get("optimizer", "adam"),
            "criterion": self.training_cfg.get("criterion", "cross_entropy"),
            "num_workers": self.data_cfg.get("num_workers", 2),
            "device": str(device),
            "early_stopping_enabled": early_stopping_enabled,
            "early_stopping_patience": es_cfg.get("patience", 5),
        })
        
        # Log model architecture info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_params({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        })
    
    def _training_loop(
        self, trainer, model, model_name, resolution,
        best_model_path, early_stopping, early_stopping_enabled
    ):
        """Execute training loop with early stopping and checkpointing."""
        best_val_loss = float('inf')
        
        for epoch in tqdm(
            range(self.training_cfg["epochs"]),
            desc=f"Training {model_name}@{resolution}"
        ):
            print(f"\nEpoch {epoch + 1}/{self.training_cfg['epochs']}")
            
            train_metrics = trainer.train_epoch()
            val_metrics = trainer.validate()
            
            print("Train →", train_metrics)
            print("Val   →", val_metrics)
            
            # Log metrics
            mlflow.log_metrics(
                {f"train_{k}": v for k, v in train_metrics.items()},
                step=epoch
            )
            mlflow.log_metrics(
                {f"val_{k}": v for k, v in val_metrics.items()},
                step=epoch
            )
            
            # Save best model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model": model_name,
                        "resolution": resolution,
                        "epoch": epoch,
                        "val_loss": val_metrics["loss"],
                        "val_accuracy": val_metrics.get("accuracy", 0.0),
                    },
                    best_model_path,
                )
                print(f"[ResoMap] ✔ Saved best model → {best_model_path}")
            
            # Early stopping check
            if early_stopping_enabled:
                early_stopping.step(epoch, val_metrics["loss"])
                if early_stopping.stop:
                    print(f"[ResoMap] Early stopping at epoch {epoch + 1}")
                    break
        
        return best_val_loss
    
    def _save_final_model(
        self, model, model_name, resolution,
        final_model_path, val_metrics
    ):
        """Save final model checkpoint."""
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model": model_name,
                "resolution": resolution,
                "val_loss": val_metrics.get("loss", 0.0),
                "val_accuracy": val_metrics.get("accuracy", 0.0),
            },
            final_model_path,
        )
        print(f"[ResoMap] ✔ Saved final model → {final_model_path}")
    
    def _evaluate_test_set(self, model, test_loader, device, criterion):
        """Evaluate model on test set."""
        print("\n[ResoMap] Evaluating on test set...")
        test_metrics = evaluate_test_set(model, test_loader, device, criterion)
        
        print("Test Results →", test_metrics)
        
        # Log test metrics
        mlflow.log_metrics({
            f"test_{k}": v for k, v in test_metrics.items()
        })
        
        return test_metrics
    
    def _profile_model(self, model, device, test_loader):
        """Profile model performance."""
        print("\n[ResoMap] Profiling model...")
        prof_results = profile_model(
            model,
            device,
            test_loader,
            warmup=self.system_cfg.get("warmup_runs", 5),
            runs=self.system_cfg.get("num_profiling_runs", 30),
            track_activation_memory=self.system_cfg.get("track_activation_memory", False),
        )
        
        # Log profiling metrics
        mlflow.log_metrics({
            "inference_time_sec": prof_results.get("avg_time_sec", 0),
            "throughput_samples_sec": prof_results.get("throughput_samples_sec", 0),
            "peak_memory_mb": prof_results.get("peak_cpu_memory_mb", 0),
        })
        
        if "gpu_memory_peak_mb" in prof_results:
            mlflow.log_metric("gpu_memory_peak_mb", prof_results["gpu_memory_peak_mb"])
        
        print(f"[ResoMap] Inference Time={prof_results.get('avg_time_sec', 0):.6f}s")
        print(f"[ResoMap] Throughput={prof_results.get('throughput_samples_sec', 0):.2f} samples/s")
        
        return prof_results
    
    def _print_summary(
        self, model_name, resolution, best_val_loss,
        test_metrics, prof_results
    ):
        """Print experiment summary."""
        print(f"\n{'='*60}")
        print(f"Experiment Complete: {model_name} @ {resolution}x{resolution}")
        print(f"Best Val Loss: {best_val_loss:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test F1-Score: {test_metrics['f1_score']:.4f}")
        print(f"{'='*60}")
