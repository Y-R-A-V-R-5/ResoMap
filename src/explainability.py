"""
============================================================
src/explainability.py
============================================================

Model Explainability Utilities for ResoMap
------------------------------------------------------------

This module provides comprehensive explainability tools to understand
how different image resolutions affect model decision-making.

Implemented Methods:
1. Grad-CAM (Gradient-weighted Class Activation Mapping)
   - Visualizes which regions of an image are important for predictions
   - Shows spatial attention patterns across different resolutions

2. Integrated Gradients
   - Attribution method that assigns importance scores to input features
   - Compares feature importance across resolutions

3. Saliency Maps
   - Basic gradient-based visualization
   - Fast computation for quick insights

4. Resolution Comparison
   - Side-by-side visualization of explanations at different resolutions
   - Quantitative metrics for explanation consistency

Usage:
    explainer = ModelExplainer(model, device, target_layer)
    explanations = explainer.generate_explanations(images, labels, method='gradcam')
    explainer.visualize_comparison(images, resolutions=[224, 384, 512])
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import seaborn as sns

# Captum imports for advanced explainability
try:
    from captum.attr import (
        IntegratedGradients,
        Saliency,
        GuidedGradCam,
        LayerGradCam,
        LayerAttribution
    )
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    print("Warning: Captum not available. Some explainability features will be limited.")


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping implementation.
    
    Visualizes which regions of the input image are important for 
    the model's prediction by using gradients flowing into the final
    convolutional layer.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Initialize Grad-CAM.
        
        Parameters
        ----------
        model : torch.nn.Module
            The model to explain
        target_layer : torch.nn.Module
            The convolutional layer to compute CAM from (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save forward pass activations."""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Class Activation Map.
        
        Parameters
        ----------
        input_tensor : torch.Tensor
            Input image tensor of shape (1, C, H, W)
        target_class : int, optional
            Target class index. If None, uses predicted class.
        
        Returns
        -------
        np.ndarray
            Heatmap of shape (H, W) with values in [0, 1]
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass for target class
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # Compute weighted combination of activation maps
        gradients = self.gradients.cpu().numpy()[0]  # (C, H, W)
        activations = self.activations.cpu().numpy()[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))  # (C,)
        
        # Weighted sum of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU and normalize
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def visualize_cam(
        self,
        input_tensor: torch.Tensor,
        cam: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Overlay CAM heatmap on original image.
        
        Parameters
        ----------
        input_tensor : torch.Tensor
            Original input image (1, C, H, W)
        cam : np.ndarray
            Class activation map
        alpha : float
            Transparency of heatmap overlay
        
        Returns
        -------
        np.ndarray
            RGB image with heatmap overlay
        """
        # Convert tensor to image
        img = input_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        
        # Resize CAM to match image size
        h, w = img.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        # Overlay
        overlay = alpha * heatmap + (1 - alpha) * img
        overlay = np.clip(overlay, 0, 1)
        
        return overlay


class ModelExplainer:
    """
    Comprehensive explainability toolkit for ResoMap experiments.
    
    Supports multiple explanation methods and provides utilities for
    comparing explanations across different resolutions.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        target_layer: Optional[torch.nn.Module] = None,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize ModelExplainer.
        
        Parameters
        ----------
        model : torch.nn.Module
            Model to explain
        device : torch.device
            Computation device
        target_layer : torch.nn.Module, optional
            Target layer for CAM methods
        class_names : list of str, optional
            Human-readable class names
        """
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names
        
        # Initialize Grad-CAM if target layer provided
        self.gradcam = None
        if target_layer is not None:
            self.gradcam = GradCAM(model, target_layer)
        
        # Initialize Captum methods if available
        if CAPTUM_AVAILABLE:
            self.integrated_gradients = IntegratedGradients(model)
            self.saliency = Saliency(model)
    
    def explain_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        method: str = "gradcam",
        target_classes: Optional[torch.Tensor] = None
    ) -> List[np.ndarray]:
        """
        Generate explanations for a batch of images.
        
        Parameters
        ----------
        images : torch.Tensor
            Batch of images (B, C, H, W)
        labels : torch.Tensor
            Ground truth labels (B,)
        method : str
            Explanation method: 'gradcam', 'integrated_gradients', 'saliency'
        target_classes : torch.Tensor, optional
            Classes to explain (if None, uses predicted classes)
        
        Returns
        -------
        list of np.ndarray
            List of explanation heatmaps
        """
        images = images.to(self.device)
        explanations = []
        
        if method == "gradcam":
            if self.gradcam is None:
                raise ValueError("GradCAM not initialized. Provide target_layer.")
            
            for i in range(len(images)):
                img = images[i:i+1]
                target = target_classes[i].item() if target_classes is not None else None
                cam = self.gradcam.generate_cam(img, target)
                explanations.append(cam)
        
        elif method == "integrated_gradients" and CAPTUM_AVAILABLE:
            self.model.eval()
            if target_classes is None:
                with torch.no_grad():
                    outputs = self.model(images)
                    target_classes = outputs.argmax(dim=1)
            
            attributions = self.integrated_gradients.attribute(
                images,
                target=target_classes,
                n_steps=50
            )
            
            # Convert to heatmaps
            for i in range(len(images)):
                attr = attributions[i].cpu().numpy()
                # Sum across channels and normalize
                heatmap = np.abs(attr).sum(axis=0)
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                explanations.append(heatmap)
        
        elif method == "saliency" and CAPTUM_AVAILABLE:
            self.model.eval()
            if target_classes is None:
                with torch.no_grad():
                    outputs = self.model(images)
                    target_classes = outputs.argmax(dim=1)
            
            attributions = self.saliency.attribute(images, target=target_classes)
            
            for i in range(len(images)):
                attr = attributions[i].cpu().numpy()
                heatmap = np.abs(attr).sum(axis=0)
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                explanations.append(heatmap)
        
        else:
            raise ValueError(f"Unknown or unavailable method: {method}")
        
        return explanations
    
    def visualize_explanation(
        self,
        image: torch.Tensor,
        explanation: np.ndarray,
        pred_class: int,
        true_class: int,
        save_path: Optional[Path] = None,
        show: bool = True
    ):
        """
        Visualize a single explanation.
        
        Parameters
        ----------
        image : torch.Tensor
            Original image (C, H, W)
        explanation : np.ndarray
            Explanation heatmap
        pred_class : int
            Predicted class index
        true_class : int
            Ground truth class index
        save_path : Path, optional
            Path to save visualization
        show : bool
            Whether to display the plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        img = image.cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        # Heatmap
        axes[1].imshow(explanation, cmap='jet')
        axes[1].set_title("Explanation Heatmap")
        axes[1].axis("off")
        
        # Overlay
        h, w = img.shape[:2]
        explanation_resized = cv2.resize(explanation, (w, h))
        heatmap = cv2.applyColorMap(np.uint8(255 * explanation_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = 0.5 * heatmap + 0.5 * img
        
        axes[2].imshow(overlay)
        pred_name = self.class_names[pred_class] if self.class_names else str(pred_class)
        true_name = self.class_names[true_class] if self.class_names else str(true_class)
        axes[2].set_title(f"Overlay\nPred: {pred_name} | True: {true_name}")
        axes[2].axis("off")
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def compare_resolutions(
        self,
        images_dict: Dict[int, torch.Tensor],
        labels: torch.Tensor,
        method: str = "gradcam",
        num_samples: int = 5,
        save_dir: Optional[Path] = None
    ):
        """
        Compare explanations across multiple resolutions.
        
        Parameters
        ----------
        images_dict : dict
            Dictionary mapping resolution -> batch of images
        labels : torch.Tensor
            Ground truth labels
        method : str
            Explanation method to use
        num_samples : int
            Number of samples to visualize
        save_dir : Path, optional
            Directory to save visualizations
        """
        resolutions = sorted(images_dict.keys())
        num_samples = min(num_samples, len(labels))
        
        for sample_idx in range(num_samples):
            fig = plt.figure(figsize=(5 * len(resolutions), 10))
            gs = GridSpec(2, len(resolutions), figure=fig)
            
            for res_idx, res in enumerate(resolutions):
                images = images_dict[res]
                img = images[sample_idx:sample_idx+1]
                
                # Generate explanation
                explanations = self.explain_batch(img, labels[sample_idx:sample_idx+1], method)
                explanation = explanations[0]
                
                # Plot original
                ax_img = fig.add_subplot(gs[0, res_idx])
                img_np = img.squeeze().cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
                ax_img.imshow(img_np)
                ax_img.set_title(f"Resolution: {res}x{res}")
                ax_img.axis("off")
                
                # Plot explanation overlay
                ax_exp = fig.add_subplot(gs[1, res_idx])
                h, w = img_np.shape[:2]
                exp_resized = cv2.resize(explanation, (w, h))
                heatmap = cv2.applyColorMap(np.uint8(255 * exp_resized), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
                overlay = 0.5 * heatmap + 0.5 * img_np
                ax_exp.imshow(overlay)
                ax_exp.set_title(f"{method.upper()} Explanation")
                ax_exp.axis("off")
            
            true_label = labels[sample_idx].item()
            label_name = self.class_names[true_label] if self.class_names else str(true_label)
            fig.suptitle(f"Sample {sample_idx + 1} - True Label: {label_name}", 
                        fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            
            if save_dir:
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"resolution_comparison_sample_{sample_idx+1}.png"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
    
    def compute_explanation_metrics(
        self,
        explanations1: List[np.ndarray],
        explanations2: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute similarity metrics between two sets of explanations.
        
        Useful for comparing consistency across resolutions.
        
        Parameters
        ----------
        explanations1 : list of np.ndarray
            First set of explanations
        explanations2 : list of np.ndarray
            Second set of explanations
        
        Returns
        -------
        dict
            Dictionary with similarity metrics
        """
        from scipy.stats import spearmanr
        from sklearn.metrics import mean_squared_error
        
        correlations = []
        mse_values = []
        
        for exp1, exp2 in zip(explanations1, explanations2):
            # Resize to same size
            h, w = min(exp1.shape[0], exp2.shape[0]), min(exp1.shape[1], exp2.shape[1])
            exp1_resized = cv2.resize(exp1, (w, h))
            exp2_resized = cv2.resize(exp2, (w, h))
            
            # Flatten
            exp1_flat = exp1_resized.flatten()
            exp2_flat = exp2_resized.flatten()
            
            # Spearman correlation
            corr, _ = spearmanr(exp1_flat, exp2_flat)
            correlations.append(corr)
            
            # MSE
            mse = mean_squared_error(exp1_flat, exp2_flat)
            mse_values.append(mse)
        
        return {
            "mean_correlation": np.mean(correlations),
            "std_correlation": np.std(correlations),
            "mean_mse": np.mean(mse_values),
            "std_mse": np.std(mse_values)
        }


def get_target_layer(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    """
    Get the appropriate target layer for Grad-CAM based on model architecture.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model
    model_name : str
        Name of the model architecture
    
    Returns
    -------
    torch.nn.Module
        Target layer for Grad-CAM
    """
    if hasattr(model, 'stages'):
        # For VGG-style models
        stage_keys = list(model.stages.keys())
        last_stage = stage_keys[-1]
        return model.stages[last_stage]
    else:
        raise ValueError(f"Cannot automatically determine target layer for {model_name}")


def batch_explain_and_save(
    explainer: ModelExplainer,
    dataloader: torch.utils.data.DataLoader,
    method: str,
    save_dir: Path,
    max_samples: int = 100
):
    """
    Generate and save explanations for a batch of samples.
    
    Parameters
    ----------
    explainer : ModelExplainer
        Explainer instance
    dataloader : DataLoader
        Data loader with samples to explain
    method : str
        Explanation method
    save_dir : Path
        Directory to save visualizations
    max_samples : int
        Maximum number of samples to process
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    explainer.model.eval()
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            if sample_count >= max_samples:
                break
            
            images = images.to(explainer.device)
            outputs = explainer.model(images)
            predictions = outputs.argmax(dim=1)
            
            # Generate explanations
            explanations = explainer.explain_batch(images, labels, method)
            
            # Save visualizations
            for i in range(len(images)):
                if sample_count >= max_samples:
                    break
                
                save_path = save_dir / f"{method}_sample_{sample_count:04d}.png"
                explainer.visualize_explanation(
                    images[i],
                    explanations[i],
                    predictions[i].item(),
                    labels[i].item(),
                    save_path=save_path,
                    show=False
                )
                sample_count += 1
            
            print(f"Processed {sample_count}/{max_samples} samples")
