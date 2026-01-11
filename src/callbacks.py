"""
============================================================
src/callbacks.py
============================================================

Training Callbacks for ResoMap Experiments
------------------------------------------------------------

This file contains utilities that interact with the training
loop to improve efficiency and reliability.

Currently implemented:

1. EarlyStopping:
   - Monitors a specified metric (e.g., validation loss or accuracy).
   - Stops training early when no meaningful improvement is
     observed for a configurable number of epochs (patience).
   - Supports a warmup period to ignore initial unstable epochs.
   - Configurable for "min" (e.g., loss) or "max" (e.g., accuracy)
     metrics and minimum improvement thresholds (min_delta).

2. ModelCheckpoint:
   - Saves model checkpoints during training.
   - Keeps top-k best models based on monitored metric.
   - Supports saving model state, optimizer state, and training metadata.
   - Essential for resolution experiments to save best models at each resolution.
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any
import json

class EarlyStopping:
    """
    Early stopping utility for training loops.

    Monitors a metric and stops training when no meaningful
    improvement is observed for a given patience window.

    Attributes
    ----------
    patience : int
        Number of consecutive non-improving epochs allowed.
    warmup_epochs : int
        Number of initial epochs to ignore for stability.
    mode : str
        "min" if lower metric values are better, "max" if higher is better.
    min_delta : float
        Minimum improvement required to reset patience.
    best_metric : float or None
        Best metric value observed so far.
    num_bad_epochs : int
        Counter for epochs without improvement.
    stop : bool
        Flag indicating whether training should stop.
    """

    def __init__(
        self,
        patience: int = 5,
        warmup_epochs: int = 3,
        mode: str = "min",
        min_delta: float = 0.0,
    ):
        # Validate mode
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")

        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.mode = mode
        self.min_delta = min_delta

        self.best_metric = None
        self.num_bad_epochs = 0
        self.stop = False
        self.is_best = False  # Indicates a new best metric this epoch


    def step(self, epoch: int, current_metric: float):
        """
        Update early stopping state.

        Parameters
        ----------
        epoch : int
            Current epoch index (0-based).
        current_metric : float
            Value of the monitored metric for this epoch.
        """

        # Reset best-flag at the beginning of each epoch
        self.is_best = False
        
        # Skip early stopping checks during warmup period
        if epoch < self.warmup_epochs:
            return

        # Initialize best_metric after warmup
        if self.best_metric is None:
            self.best_metric = current_metric
            self.is_best = True
            return

        # Determine whether the metric has improved
        if self.mode == "min":
            improved = current_metric < (self.best_metric - self.min_delta)
        else:  # mode == "max"
            improved = current_metric > (self.best_metric + self.min_delta)

        if improved:
            # Update best metric and reset counter
            self.best_metric = current_metric
            self.num_bad_epochs = 0
            self.is_best = True
        else:
            # Increment bad epoch counter
            self.num_bad_epochs += 1

            # Trigger early stopping if patience exceeded
            if self.num_bad_epochs >= self.patience:
                self.stop = True


class ModelCheckpoint:
    """
    Model checkpointing callback for saving best models during training.
    
    Saves the top-k best models based on a monitored metric. Useful for
    resolution experiments where we want to preserve the best model at
    each resolution for later explainability analysis.
    
    Attributes
    ----------
    save_dir : Path
        Directory to save checkpoints
    monitor : str
        Metric to monitor (e.g., 'val_accuracy', 'val_loss')
    mode : str
        'min' if lower is better, 'max' if higher is better
    save_top_k : int
        Number of best models to keep
    verbose : bool
        Whether to print save messages
    best_models : list
        List of (metric_value, checkpoint_path) tuples for top-k models
    """
    
    def __init__(
        self,
        save_dir: str,
        monitor: str = "val_accuracy",
        mode: str = "max",
        save_top_k: int = 3,
        filename_template: str = "model_{resolution}_{epoch:03d}_{metric:.4f}.pt",
        verbose: bool = True,
        save_optimizer: bool = True
    ):
        """
        Initialize ModelCheckpoint callback.
        
        Parameters
        ----------
        save_dir : str
            Directory to save checkpoints
        monitor : str
            Metric name to monitor
        mode : str
            'min' or 'max'
        save_top_k : int
            Number of best models to keep (-1 for all)
        filename_template : str
            Template for checkpoint filenames
        verbose : bool
            Print save notifications
        save_optimizer : bool
            Whether to save optimizer state
        """
        if mode not in ('min', 'max'):
            raise ValueError("mode must be 'min' or 'max'")
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.filename_template = filename_template
        self.verbose = verbose
        self.save_optimizer = save_optimizer
        
        self.best_models = []  # List of (metric, path) tuples
        self.best_metric = None
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.mode == 'min':
            return current < best
        else:
            return current > best
    
    def step(
        self,
        epoch: int,
        metrics: Dict[str, float],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        resolution: Optional[int] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if model should be saved and save if necessary.
        
        Parameters
        ----------
        epoch : int
            Current epoch number
        metrics : dict
            Dictionary of metrics for this epoch
        model : torch.nn.Module
            Model to save
        optimizer : torch.optim.Optimizer, optional
            Optimizer to save
        resolution : int, optional
            Image resolution (for filename)
        additional_info : dict, optional
            Additional metadata to save
        
        Returns
        -------
        bool
            True if model was saved, False otherwise
        """
        if self.monitor not in metrics:
            if self.verbose:
                print(f"Warning: Monitored metric '{self.monitor}' not found in metrics")
            return False
        
        current_metric = metrics[self.monitor]
        
        # Check if this is a new best
        should_save = False
        if self.best_metric is None:
            should_save = True
        elif self._is_better(current_metric, self.best_metric):
            should_save = True
        elif self.save_top_k == -1:
            should_save = True  # Save all
        elif len(self.best_models) < self.save_top_k:
            should_save = True  # Haven't reached top-k yet
        else:
            # Check if better than worst in top-k
            worst_metric = min(self.best_models, key=lambda x: x[0] if self.mode == 'max' else -x[0])[0]
            if self._is_better(current_metric, worst_metric):
                should_save = True
        
        if should_save:
            # Create filename
            filename = self.filename_template.format(
                resolution=resolution or 'unknown',
                epoch=epoch,
                metric=current_metric
            )
            checkpoint_path = self.save_dir / filename
            
            # Prepare checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': metrics,
                'monitor_metric': self.monitor,
                'monitor_value': current_metric,
            }
            
            if self.save_optimizer and optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
            if resolution is not None:
                checkpoint['resolution'] = resolution
            
            if additional_info is not None:
                checkpoint['additional_info'] = additional_info
            
            # Save checkpoint
            torch.save(checkpoint, checkpoint_path)
            
            if self.verbose:
                print(f"Saved checkpoint: {checkpoint_path.name} ({self.monitor}={current_metric:.4f})")
            
            # Update best models list
            self.best_models.append((current_metric, checkpoint_path))
            
            # Keep only top-k
            if self.save_top_k > 0 and len(self.best_models) > self.save_top_k:
                # Sort by metric
                if self.mode == 'max':
                    self.best_models.sort(key=lambda x: x[0], reverse=True)
                else:
                    self.best_models.sort(key=lambda x: x[0])
                
                # Remove worst
                _, path_to_remove = self.best_models.pop()
                if path_to_remove.exists():
                    path_to_remove.unlink()
                    if self.verbose:
                        print(f"Removed checkpoint: {path_to_remove.name}")
            
            # Update best metric
            if self.best_metric is None or self._is_better(current_metric, self.best_metric):
                self.best_metric = current_metric
            
            return True
        
        return False
    
    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Get path to the best checkpoint."""
        if not self.best_models:
            return None
        
        if self.mode == 'max':
            return max(self.best_models, key=lambda x: x[0])[1]
        else:
            return min(self.best_models, key=lambda x: x[0])[1]
    
    def get_all_checkpoints(self) -> list:
        """Get all checkpoint paths sorted by metric."""
        if self.mode == 'max':
            return sorted(self.best_models, key=lambda x: x[0], reverse=True)
        else:
            return sorted(self.best_models, key=lambda x: x[0])
    
    def save_metadata(self, filename: str = "checkpoint_metadata.json"):
        """Save metadata about all checkpoints."""
        metadata_path = self.save_dir / filename
        
        metadata = {
            'monitor': self.monitor,
            'mode': self.mode,
            'save_top_k': self.save_top_k,
            'best_metric': self.best_metric,
            'checkpoints': [
                {
                    'metric_value': metric,
                    'path': str(path.name)
                }
                for metric, path in self.get_all_checkpoints()
            ]
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if self.verbose:
            print(f"Saved checkpoint metadata to {metadata_path}")


class LRSchedulerCallback:
    """
    Learning Rate Scheduler Callback.
    
    Manages learning rate scheduling during training with support for
    multiple scheduler types commonly used in deep learning:
    
    - CosineAnnealingLR: Gradually reduces LR following cosine curve
    - StepLR: Reduces LR by a factor every N epochs
    - ReduceLROnPlateau: Reduces LR when metric stops improving
    - ExponentialLR: Exponentially decays LR each epoch
    - MultiStepLR: Reduces LR at specific milestone epochs
    
    Attributes
    ----------
    scheduler : torch.optim.lr_scheduler._LRScheduler
        PyTorch learning rate scheduler
    scheduler_type : str
        Type of scheduler being used
    verbose : bool
        Whether to print LR updates
    metric_based : bool
        True if scheduler needs metric (ReduceLROnPlateau)
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_type: str = 'cosine',
        scheduler_params: Optional[Dict[str, Any]] = None,
        verbose: bool = True
    ):
        """
        Initialize LR Scheduler Callback.
        
        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer whose learning rate will be scheduled
        scheduler_type : str
            Type of scheduler: 'cosine', 'step', 'plateau', 'exponential', 'multistep'
        scheduler_params : dict, optional
            Parameters specific to the scheduler type
        verbose : bool
            Print learning rate updates
        """
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type.lower()
        self.verbose = verbose
        self.scheduler_params = scheduler_params or {}
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Check if scheduler needs metric
        self.metric_based = isinstance(
            self.scheduler,
            torch.optim.lr_scheduler.ReduceLROnPlateau
        )
        
        self.current_lr = None
        self._update_current_lr()
    
    def _create_scheduler(self):
        """Create the appropriate scheduler based on type."""
        if self.scheduler_type == 'cosine':
            T_max = self.scheduler_params.get('T_max', 50)
            eta_min = self.scheduler_params.get('eta_min', 0)
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max, eta_min=eta_min
            )
        
        elif self.scheduler_type == 'step':
            step_size = self.scheduler_params.get('step_size', 10)
            gamma = self.scheduler_params.get('gamma', 0.1)
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        
        elif self.scheduler_type == 'plateau':
            mode = self.scheduler_params.get('mode', 'min')
            factor = self.scheduler_params.get('factor', 0.1)
            patience = self.scheduler_params.get('patience', 10)
            min_lr = self.scheduler_params.get('min_lr', 0)
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode=mode, factor=factor,
                patience=patience, min_lr=min_lr, verbose=self.verbose
            )
        
        elif self.scheduler_type == 'exponential':
            gamma = self.scheduler_params.get('gamma', 0.95)
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=gamma
            )
        
        elif self.scheduler_type == 'multistep':
            milestones = self.scheduler_params.get('milestones', [30, 60, 90])
            gamma = self.scheduler_params.get('gamma', 0.1)
            return torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=milestones, gamma=gamma
            )
        
        else:
            raise ValueError(
                f"Unknown scheduler type: {self.scheduler_type}. "
                f"Supported: 'cosine', 'step', 'plateau', 'exponential', 'multistep'"
            )
    
    def _update_current_lr(self):
        """Update current learning rate."""
        self.current_lr = [
            param_group['lr'] for param_group in self.optimizer.param_groups
        ]
    
    def step(self, epoch: Optional[int] = None, metrics: Optional[Dict[str, float]] = None):
        """
        Perform scheduler step.
        
        Parameters
        ----------
        epoch : int, optional
            Current epoch (not needed for most schedulers)
        metrics : dict, optional
            Metrics dictionary (required for ReduceLROnPlateau)
        """
        old_lr = self.current_lr.copy()
        
        if self.metric_based:
            # ReduceLROnPlateau needs a metric
            if metrics is None:
                raise ValueError("ReduceLROnPlateau requires metrics dict")
            
            monitor_metric = self.scheduler_params.get('monitor', 'val_loss')
            if monitor_metric not in metrics:
                raise ValueError(f"Monitored metric '{monitor_metric}' not in metrics")
            
            self.scheduler.step(metrics[monitor_metric])
        else:
            # Other schedulers just step normally
            self.scheduler.step()
        
        self._update_current_lr()
        
        # Print LR update if changed
        if self.verbose and self.current_lr != old_lr:
            lr_str = ', '.join([f'{lr:.6f}' for lr in self.current_lr])
            print(f"Learning rate updated: {lr_str}")
    
    def get_last_lr(self) -> list:
        """Get current learning rates."""
        return self.current_lr
    
    def state_dict(self) -> dict:
        """Get scheduler state for checkpointing."""
        return {
            'scheduler_state': self.scheduler.state_dict(),
            'scheduler_type': self.scheduler_type,
            'scheduler_params': self.scheduler_params
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load scheduler state from checkpoint."""
        self.scheduler.load_state_dict(state_dict['scheduler_state'])
        self._update_current_lr()


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, 
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Load a model checkpoint.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to checkpoint file
    model : torch.nn.Module
        Model to load state into
    optimizer : torch.optim.Optimizer, optional
        Optimizer to load state into
    device : torch.device, optional
        Device to load checkpoint to
    
    Returns
    -------
    dict
        Checkpoint metadata
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Monitored metric ({checkpoint['monitor_metric']}): {checkpoint['monitor_value']:.4f}")
    
    return checkpoint