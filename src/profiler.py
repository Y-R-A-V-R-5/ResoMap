"""
profiler.py - Model Profiling Utilities for ResoMap Experiments

This module provides utilities to benchmark PyTorch models in terms of:
1. Inference time per batch
2. Memory usage (RAM)

Main Functionality:
- profile_model:
    Measures average forward-pass time and peak memory usage for a given model 
    and DataLoader. Supports optional warmup iterations to stabilize GPU timing.

Design Notes:
- Uses CPU memory tracking via psutil.
- Includes warmup iterations to avoid initial GPU initialization overhead.
- Can be used for both CPU and GPU devices.
- Returns average time per batch and peak memory usage in MB.
"""

import torch
import time
import psutil

def profile_model(
    model: torch.nn.Module, 
    device: torch.device, 
    loader, 
    warmup: int = 5, 
    runs: int = 10, 
    track_activation_memory: bool = False
):
    """
    Profile model inference time and memory usage.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to profile.
    device : torch.device
        Device for inference ('cpu' or 'cuda').
    loader : DataLoader
        DataLoader providing input batches.
    warmup : int, optional
        Number of warmup iterations to ignore before measuring, default=5.
    runs : int, optional
        Number of measured iterations, default=10.
    track_activation_memory : bool, optional
        Currently not implemented; placeholder to track GPU activation memory.

    Returns
    -------
    tuple
        (avg_time_sec, peak_memory_mb)
        - avg_time_sec: Average forward-pass time per batch in seconds.
        - peak_memory_mb: Peak process memory usage (RSS) in MB.
    """
    # Move model to device (CPU or GPU)
    model.to(device)
    model.eval()  # Set to evaluation mode to disable dropout/batchnorm updates

    # ------------------------
    # Warmup iterations
    # ------------------------
    # Helps stabilize timing (esp. on GPU) by running a few batches first
    with torch.no_grad():
        for _ in range(warmup):
            for x, _ in loader:
                model(x.to(device))

    # ------------------------
    # Profiling iterations
    # ------------------------
    times, mems = [] , []

    with torch.no_grad():
        for _ in range(runs):
            for x, _ in loader:
                # Start timing
                start = time.time()

                # Forward pass
                out = model(x.to(device))

                # End timing
                end = time.time()

                # Record elapsed time
                times.append(end - start)

                # Record memory usage in MB
                mems.append(psutil.Process().memory_info().rss / 1e6)  

    # Compute average latency and peak memory
    avg_time = sum(times) / len(times)
    peak_mem = max(mems)

    return avg_time, peak_mem
