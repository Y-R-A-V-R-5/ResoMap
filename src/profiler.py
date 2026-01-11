"""
profiler.py - Enhanced Model Profiling Utilities for ResoMap Experiments

This module provides comprehensive utilities to benchmark PyTorch models:
1. Inference time per batch (CPU and GPU)
2. Memory usage (RAM and GPU VRAM)
3. Throughput (samples/second)
4. FLOPs and parameter count

Main Functionality:
- profile_model:
    Measures average forward-pass time, memory usage, and throughput
    Supports both CPU and GPU profiling with CUDA synchronization
    
- profile_gpu_memory:
    Detailed GPU memory profiling for activation memory tracking
    
- get_model_complexity:
    Compute FLOPs and parameter count for model complexity analysis

Design Notes:
- Uses CUDA events for accurate GPU timing
- Tracks both allocated and reserved GPU memory
- Includes warmup iterations to stabilize measurements
- Provides detailed profiling reports
"""

import torch
import time
import psutil
from typing import Tuple, Dict, Optional
import numpy as np

def profile_model(
    model: torch.nn.Module, 
    device: torch.device, 
    loader, 
    warmup: int = 5, 
    runs: int = 10, 
    track_activation_memory: bool = False,
    batch_size: Optional[int] = None
) -> Dict[str, float]:
    """
    Profile model inference time, memory usage, and throughput.

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
        Track GPU activation memory (requires CUDA).
    batch_size : int, optional
        Batch size for throughput calculation. If None, inferred from loader.

    Returns
    -------
    dict
        Dictionary containing:
        - avg_time_sec: Average forward-pass time per batch
        - std_time_sec: Standard deviation of time
        - throughput_samples_sec: Samples processed per second
        - peak_memory_mb: Peak CPU memory (RSS)
        - gpu_memory_allocated_mb: GPU memory allocated (if CUDA)
        - gpu_memory_reserved_mb: GPU memory reserved (if CUDA)
        - gpu_memory_peak_mb: Peak GPU memory (if CUDA)
    """
    # Move model to device
    model.to(device)
    model.eval()
    
    is_cuda = device.type == 'cuda'
    
    # Infer batch size if not provided
    if batch_size is None:
        for x, _ in loader:
            batch_size = x.size(0)
            break
    
    # Reset GPU memory stats if tracking
    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    
    # ------------------------
    # Warmup iterations
    # ------------------------
    with torch.no_grad():
        warmup_count = 0
        for x, _ in loader:
            if warmup_count >= warmup:
                break
            x = x.to(device, non_blocking=True)
            if is_cuda:
                torch.cuda.synchronize(device)
            _ = model(x)
            if is_cuda:
                torch.cuda.synchronize(device)
            warmup_count += 1
    
    # ------------------------
    # Profiling iterations
    # ------------------------
    times = []
    cpu_mems = []
    
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        # Use CUDA events for accurate timing
        start_events = []
        end_events = []
    
    with torch.no_grad():
        run_count = 0
        for x, _ in loader:
            if run_count >= runs:
                break
            
            x = x.to(device, non_blocking=True)
            
            if is_cuda:
                # GPU timing with CUDA events
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                torch.cuda.synchronize(device)
                start_event.record()
                
                _ = model(x)
                
                end_event.record()
                torch.cuda.synchronize(device)
                
                elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
                times.append(elapsed_time)
                
            else:
                # CPU timing
                start = time.time()
                _ = model(x)
                end = time.time()
                times.append(end - start)
            
            # CPU memory
            cpu_mems.append(psutil.Process().memory_info().rss / 1e6)  # MB
            run_count += 1
    
    # Compute statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    throughput = batch_size / avg_time if avg_time > 0 else 0
    peak_cpu_mem = max(cpu_mems)
    
    results = {
        'avg_time_sec': avg_time,
        'std_time_sec': std_time,
        'min_time_sec': min(times),
        'max_time_sec': max(times),
        'throughput_samples_sec': throughput,
        'peak_cpu_memory_mb': peak_cpu_mem,
    }
    
    # GPU memory statistics
    if is_cuda:
        results['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated(device) / 1024**2
        results['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved(device) / 1024**2
        results['gpu_memory_peak_mb'] = torch.cuda.max_memory_allocated(device) / 1024**2
    
    return results


def profile_gpu_memory(
    model: torch.nn.Module,
    device: torch.device,
    input_shape: Tuple[int, ...],
    track_activations: bool = True
) -> Dict[str, float]:
    """
    Detailed GPU memory profiling for a model.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to profile
    device : torch.device
        CUDA device
    input_shape : tuple
        Shape of input tensor (B, C, H, W)
    track_activations : bool
        Whether to track activation memory
    
    Returns
    -------
    dict
        Memory statistics in MB
    """
    if device.type != 'cuda':
        raise ValueError("GPU profiling requires CUDA device")
    
    model.to(device)
    model.eval()
    
    # Reset memory stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    # Measure model parameters
    initial_mem = torch.cuda.memory_allocated(device)
    
    # Forward pass
    dummy_input = torch.randn(*input_shape, device=device)
    
    with torch.no_grad():
        _ = model(dummy_input)
    
    torch.cuda.synchronize(device)
    
    # Memory measurements
    mem_after_forward = torch.cuda.memory_allocated(device)
    peak_mem = torch.cuda.max_memory_allocated(device)
    reserved_mem = torch.cuda.memory_reserved(device)
    
    # Activation memory (approximate)
    activation_mem = mem_after_forward - initial_mem
    
    return {
        'model_parameters_mb': initial_mem / 1024**2,
        'activation_memory_mb': activation_mem / 1024**2,
        'total_allocated_mb': mem_after_forward / 1024**2,
        'peak_memory_mb': peak_mem / 1024**2,
        'reserved_memory_mb': reserved_mem / 1024**2
    }


def get_model_complexity(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Compute model complexity: FLOPs and parameter count.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to analyze
    input_shape : tuple
        Input shape (B, C, H, W)
    device : torch.device, optional
        Device for computation
    
    Returns
    -------
    dict
        Model complexity statistics
    """
    if device is None:
        device = torch.device('cpu')
    
    model.to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate FLOPs (simplified - multiply-add operations in Conv and Linear)
    def count_conv2d_flops(module, input, output):
        # FLOPs = 2 * in_channels * out_channels * kernel_h * kernel_w * out_h * out_w
        batch_size, in_channels, in_h, in_w = input[0].shape
        out_channels, out_h, out_w = output.shape[1:]
        kernel_h, kernel_w = module.kernel_size
        flops = 2 * in_channels * out_channels * kernel_h * kernel_w * out_h * out_w
        module.__flops__ += flops
    
    def count_linear_flops(module, input, output):
        # FLOPs = 2 * in_features * out_features
        in_features = input[0].shape[-1]
        out_features = output.shape[-1]
        flops = 2 * in_features * out_features
        module.__flops__ += flops
    
    # Register hooks
    hooks = []
    for module in model.modules():
        module.__flops__ = 0
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(count_conv2d_flops))
        elif isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(count_linear_flops))
    
    # Forward pass
    dummy_input = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        _ = model(dummy_input)
    
    # Sum FLOPs
    total_flops = sum(module.__flops__ for module in model.modules())
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_flops': total_flops,
        'gflops': total_flops / 1e9,  # Giga FLOPs
        'params_millions': total_params / 1e6
    }


def print_profiling_report(results: Dict[str, float], title: str = "Profiling Report"):
    """
    Print a formatted profiling report.
    
    Parameters
    ----------
    results : dict
        Profiling results dictionary
    title : str
        Report title
    """
    print("\n" + "=" * 60)
    print(f"{title:^60}")
    print("=" * 60)
    
    for key, value in results.items():
        key_formatted = key.replace('_', ' ').title()
        if isinstance(value, float):
            print(f"{key_formatted:.<40} {value:.4f}")
        else:
            print(f"{key_formatted:.<40} {value}")
    
    print("=" * 60 + "\n")
