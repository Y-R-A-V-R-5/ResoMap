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
    def __init__(self, model, device, criterion, optimizer, train_loader, val_loader, metrics=None):
        self.model = model.to(device)  # Move model to device
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.metrics = metrics or ["accuracy", "precision", "recall", "f1"]  # Default metrics

    def train_epoch(self):
        """Perform one training epoch and compute metrics for the epoch."""
        self.model.train()  # Set model to training mode
        total_loss = 0
        all_preds, all_labels = [], []

        loader = tqdm(self.train_loader, desc="Training", leave=False)
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Predictions: take class with max probability
            preds = out.argmax(dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(y.cpu().numpy())

            # Batch accuracy for progress bar
            batch_acc = (preds == y.cpu()).float().mean().item()
            loader.set_postfix(loss=f"{loss.item():.4f}", accuracy=f"{batch_acc:.4f}")

        # Average epoch loss
        avg_loss = total_loss / len(self.train_loader)

        # Compute all requested metrics for the epoch
        metrics = {"loss": avg_loss}
        metrics.update(self._compute_metrics(all_labels, all_preds))

        return metrics

    def validate(self):
        """Validate model over the validation set and compute metrics."""
        self.model.eval()  # Set model to evaluation mode
        total_loss = 0
        all_preds, all_labels = [], []

        loader = tqdm(self.val_loader, desc="Validation", leave=False)
        with torch.no_grad():  # Disable gradient computation
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
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