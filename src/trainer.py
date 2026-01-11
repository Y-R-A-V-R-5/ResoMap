"""
trainer.py - Training Utilities with Multiple Metrics for ResoMap

This module defines a Trainer class that supports training and validation
of PyTorch models while tracking multiple performance metrics.

Metrics:
- Loss
- Accuracy
- Precision
- Recall
- F1-score

Metrics are computed per epoch and returned as a dictionary.
"""

import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import time

class Trainer:
    """
    Trainer class for supervised learning with PyTorch.

    Handles:
        - Training and validation loops
        - Loss computation and optimization
        - Performance metrics tracking: accuracy, precision, recall, F1-score

    Parameters
    ----------
    model : nn.Module
        PyTorch model.
    device : torch.device
        Device to run the model ('cpu' or 'cuda').
    criterion : nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimizer.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    metrics : list of str, optional
        List of metrics to track: 'accuracy', 'precision', 'recall', 'f1'. Defaults to all.
    """
    def __init__(self, model, device, criterion, optimizer, train_loader, val_loader, 
                 metrics=None, use_amp=False, gradient_accumulation_steps=1, 
                 gradient_clip_norm=None, scheduler=None):
        self.model = model.to(device)  # Move model to device
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.metrics = metrics or ["accuracy", "precision", "recall", "f1"]  # Default metrics
        
        # GPU optimizations
        self.use_amp = use_amp and device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip_norm = gradient_clip_norm
        self.scheduler = scheduler
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
        
        # Performance tracking
        self.epoch_times = []
        self.gpu_memory_usage = []

    def train_epoch(self):
        """Perform one training epoch with GPU optimizations."""
        self.model.train()  # Set model to training mode
        total_loss = 0
        all_preds, all_labels = [], []
        
        start_time = time.time()

        loader = tqdm(self.train_loader, desc="Training", leave=False)
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

            # Mixed precision training context
            with autocast(enabled=self.use_amp):
                out = self.model(x)
                loss = self.criterion(out, y)
                loss = loss / self.gradient_accumulation_steps  # Scale loss for accumulation

            # Backward pass with gradient scaling
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.gradient_clip_norm is not None:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation_steps

            # Predictions: take class with max probability
            with torch.no_grad():
                preds = out.argmax(dim=1).cpu()
                all_preds.extend(preds.numpy())
                all_labels.extend(y.cpu().numpy())

            # Batch accuracy for progress bar
            batch_acc = (preds == y.cpu()).float().mean().item()
            
            # GPU memory usage
            if self.device.type == 'cuda':
                gpu_mem = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
                loader.set_postfix(loss=f"{loss.item():.4f}", accuracy=f"{batch_acc:.4f}", 
                                 gpu_mem=f"{gpu_mem:.2f}GB")
            else:
                loader.set_postfix(loss=f"{loss.item():.4f}", accuracy=f"{batch_acc:.4f}")
        
        # Epoch statistics
        epoch_time = time.time() - start_time
        self.epoch_times.append(epoch_time)
        
        if self.device.type == 'cuda':
            max_gpu_mem = torch.cuda.max_memory_allocated(self.device) / 1024**3
            self.gpu_memory_usage.append(max_gpu_mem)
            torch.cuda.reset_peak_memory_stats(self.device)

        # Average epoch loss
        avg_loss = total_loss / len(self.train_loader)

        # Compute all requested metrics for the epoch
        metrics = {"loss": avg_loss, "epoch_time": epoch_time}
        metrics.update(self._compute_metrics(all_labels, all_preds))
        
        if self.device.type == 'cuda':
            metrics["max_gpu_memory_gb"] = max_gpu_mem

        return metrics

    def validate(self):
        """Validate model over the validation set with GPU optimizations."""
        self.model.eval()  # Set model to evaluation mode
        total_loss = 0
        all_preds, all_labels = [], []

        loader = tqdm(self.val_loader, desc="Validation", leave=False)
        with torch.no_grad():  # Disable gradient computation
            for x, y in loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                
                # Use mixed precision for inference too
                with autocast(enabled=self.use_amp):
                    out = self.model(x)
                    loss = self.criterion(out, y)
                
                total_loss += loss.item()

                preds = out.argmax(dim=1).cpu()
                all_preds.extend(preds.numpy())
                all_labels.extend(y.cpu().numpy())

                # Batch accuracy for progress bar
                batch_acc = (preds == y.cpu()).float().mean().item()
                loader.set_postfix(loss=f"{loss.item():.4f}", accuracy=f"{batch_acc:.4f}")

        avg_loss = total_loss / len(self.val_loader)

        metrics = {"loss": avg_loss}
        metrics.update(self._compute_metrics(all_labels, all_preds))

        return metrics
    
    def step_scheduler(self, val_metrics):
        """Step the learning rate scheduler if provided."""
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['loss'])
            else:
                self.scheduler.step()
    
    def get_learning_rate(self):
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def get_performance_stats(self):
        """Get training performance statistics."""
        stats = {
            'avg_epoch_time': sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0,
            'total_training_time': sum(self.epoch_times)
        }
        
        if self.gpu_memory_usage:
            stats['avg_gpu_memory_gb'] = sum(self.gpu_memory_usage) / len(self.gpu_memory_usage)
            stats['max_gpu_memory_gb'] = max(self.gpu_memory_usage)
        
        return stats

    def _compute_metrics(self, labels, preds):
        """
        Compute performance metrics: accuracy, precision, recall, F1-score.

        Parameters
        ----------
        labels : list or array
            Ground truth labels.
        preds : list or array
            Predicted labels by the model.

        Returns
        -------
        dict
            Dictionary containing requested metric values.
        """
        results = {}
        labels_tensor = torch.tensor(labels)
        preds_tensor = torch.tensor(preds)

        if "accuracy" in self.metrics:
            results["accuracy"] = (preds_tensor == labels_tensor).float().mean().item()
        if "precision" in self.metrics:
            results["precision"] = precision_score(labels, preds, average="macro", zero_division=0)
        if "recall" in self.metrics:
            results["recall"] = recall_score(labels, preds, average="macro", zero_division=0)
        if "f1" in self.metrics:
            results["f1"] = f1_score(labels, preds, average="macro", zero_division=0)

        return results