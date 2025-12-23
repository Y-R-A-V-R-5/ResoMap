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
"""

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