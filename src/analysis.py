"""
============================================================
src/analysis.py
============================================================

Comprehensive analysis modules for ResoMap project.

Contains:
1. DatasetAnalyzer - Dataset exploration and statistics
2. ModelSummaryAnalyzer - Model architecture analysis (no training required)
3. ResolutionExplainabilityAnalyzer - Model performance and explainability analysis

This module provides reusable analysis functionality for the ResoMap project.
"""

import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import yaml
from typing import Optional, Dict, List, Tuple
from collections import Counter
from PIL import Image
from torchinfo import summary

from .models import build_model
from .data import get_data_loaders
from .explainability import ModelExplainer, get_target_layer, batch_explain_and_save
from .callbacks import load_checkpoint
from .profiler import profile_model, get_model_complexity, print_profiling_report


class DatasetAnalyzer:
    """
    Comprehensive dataset exploration and analysis.
    """
    
    def __init__(self, data_dir: str, output_dir: str = "analysis"):
        """
        Initialize dataset analyzer.
        
        Parameters
        ----------
        data_dir : str
            Root directory containing train/val/test splits
        output_dir : str
            Directory to save analysis results
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect splits
        self.splits = ['train', 'val', 'test']
        self.split_paths = {
            split: self.data_dir / split 
            for split in self.splits 
            if (self.data_dir / split).exists()
        }
        
        print(f"Dataset root: {self.data_dir}")
        print(f"Found splits: {list(self.split_paths.keys())}")
        
        self.stats = {}
    
    def get_class_names(self) -> List[str]:
        """Get list of class names from directory structure."""
        # Use train split to get class names
        train_path = self.split_paths.get('train')
        if train_path and train_path.exists():
            classes = sorted([d.name for d in train_path.iterdir() if d.is_dir()])
            return classes
        return []
    
    def count_images_per_class(self, split: str) -> Dict[str, int]:
        """
        Count number of images per class in a split.
        
        Parameters
        ----------
        split : str
            Split name (train/val/test)
            
        Returns
        -------
        dict
            Class name to image count mapping
        """
        split_path = self.split_paths.get(split)
        if not split_path:
            return {}
        
        class_counts = {}
        
        for class_dir in tqdm(sorted(split_path.iterdir()), 
                             desc=f"Counting {split} images", 
                             leave=False):
            if class_dir.is_dir():
                # Count image files
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
                images = [
                    f for f in class_dir.iterdir() 
                    if f.suffix.lower() in image_extensions
                ]
                class_counts[class_dir.name] = len(images)
        
        return class_counts
    
    def get_image_statistics(self, split: str, sample_size: int = 100) -> Dict:
        """
        Get image resolution and size statistics.
        
        Parameters
        ----------
        split : str
            Split name
        sample_size : int
            Number of images to sample per class
            
        Returns
        -------
        dict
            Statistics about image dimensions
        """
        split_path = self.split_paths.get(split)
        if not split_path:
            return {}
        
        widths = []
        heights = []
        aspects = []
        
        print(f"\nSampling {sample_size} images from {split} split for statistics...")
        
        for class_dir in tqdm(sorted(split_path.iterdir()), 
                             desc=f"Analyzing {split} images",
                             leave=False):
            if not class_dir.is_dir():
                continue
                
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
            images = [
                f for f in class_dir.iterdir() 
                if f.suffix.lower() in image_extensions
            ]
            
            # Sample images
            sample = images[:min(sample_size, len(images))]
            
            for img_path in sample:
                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                        widths.append(w)
                        heights.append(h)
                        aspects.append(w / h if h > 0 else 0)
                except Exception as e:
                    print(f"Warning: Could not read {img_path}: {e}")
        
        return {
            'width': {
                'min': np.min(widths) if widths else 0,
                'max': np.max(widths) if widths else 0,
                'mean': np.mean(widths) if widths else 0,
                'std': np.std(widths) if widths else 0,
            },
            'height': {
                'min': np.min(heights) if heights else 0,
                'max': np.max(heights) if heights else 0,
                'mean': np.mean(heights) if heights else 0,
                'std': np.std(heights) if heights else 0,
            },
            'aspect_ratio': {
                'min': np.min(aspects) if aspects else 0,
                'max': np.max(aspects) if aspects else 0,
                'mean': np.mean(aspects) if aspects else 0,
                'std': np.std(aspects) if aspects else 0,
            },
            'num_samples': len(widths)
        }
    
    def analyze_dataset(self):
        """Perform comprehensive dataset analysis."""
        print("\n" + "="*60)
        print("Dataset Analysis")
        print("="*60)
        
        classes = self.get_class_names()
        print(f"\nNumber of classes: {len(classes)}")
        print(f"Classes: {classes}")
        
        # Analyze each split
        for split in self.splits:
            if split not in self.split_paths:
                continue
            
            print(f"\n{'-'*60}")
            print(f"{split.upper()} Split")
            print(f"{'-'*60}")
            
            # Count images per class
            class_counts = self.count_images_per_class(split)
            
            if not class_counts:
                print(f"No data found for {split} split")
                continue
            
            total_images = sum(class_counts.values())
            print(f"\nTotal images: {total_images:,}")
            print(f"\nImages per class:")
            
            # Create dataframe for better display
            df_data = []
            for cls in sorted(class_counts.keys()):
                count = class_counts[cls]
                percentage = (count / total_images * 100) if total_images > 0 else 0
                df_data.append({
                    'Class': cls,
                    'Count': count,
                    'Percentage': f"{percentage:.2f}%"
                })
            
            df = pd.DataFrame(df_data)
            print(df.to_string(index=False))
            
            # Check class balance
            counts = list(class_counts.values())
            if counts:
                balance_ratio = min(counts) / max(counts) if max(counts) > 0 else 0
                print(f"\nClass balance ratio (min/max): {balance_ratio:.3f}")
                if balance_ratio < 0.5:
                    print("⚠️  Warning: Dataset is imbalanced!")
                else:
                    print("✓ Dataset is relatively balanced")
            
            # Store stats
            self.stats[split] = {
                'class_counts': class_counts,
                'total_images': total_images,
                'num_classes': len(class_counts)
            }
            
            # Get image statistics
            img_stats = self.get_image_statistics(split, sample_size=50)
            if img_stats:
                self.stats[split]['image_stats'] = img_stats
                
                print(f"\nImage dimension statistics (sampled {img_stats['num_samples']} images):")
                print(f"  Width:  {img_stats['width']['mean']:.1f} ± {img_stats['width']['std']:.1f} "
                      f"[{img_stats['width']['min']:.0f} - {img_stats['width']['max']:.0f}]")
                print(f"  Height: {img_stats['height']['mean']:.1f} ± {img_stats['height']['std']:.1f} "
                      f"[{img_stats['height']['min']:.0f} - {img_stats['height']['max']:.0f}]")
                print(f"  Aspect: {img_stats['aspect_ratio']['mean']:.2f} ± {img_stats['aspect_ratio']['std']:.2f}")
        
        # Generate visualizations
        self.visualize_dataset()
        
        # Save report
        self.save_report()
    
    def visualize_dataset(self):
        """Generate visualizations for dataset analysis."""
        print(f"\n{'-'*60}")
        print("Generating visualizations...")
        print(f"{'-'*60}")
        
        # Class distribution across splits
        fig, axes = plt.subplots(1, len(self.stats), figsize=(7*len(self.stats), 6))
        if len(self.stats) == 1:
            axes = [axes]
        
        for idx, (split, stats) in enumerate(self.stats.items()):
            if 'class_counts' not in stats:
                continue
            
            class_counts = stats['class_counts']
            classes = sorted(class_counts.keys())
            counts = [class_counts[c] for c in classes]
            
            ax = axes[idx]
            bars = ax.bar(range(len(classes)), counts, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(classes))))
            ax.set_xlabel('Class', fontsize=12)
            ax.set_ylabel('Number of Images', fontsize=12)
            ax.set_title(f'{split.upper()} Split\n(Total: {stats["total_images"]:,} images)', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(classes)))
            ax.set_xticklabels(classes, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        save_path = self.output_dir / "class_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved class distribution plot: {save_path}")
        
        # Combined split comparison
        if len(self.stats) > 1:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            classes = self.get_class_names()
            x = np.arange(len(classes))
            width = 0.8 / len(self.stats)
            
            for idx, (split, stats) in enumerate(self.stats.items()):
                if 'class_counts' not in stats:
                    continue
                class_counts = stats['class_counts']
                counts = [class_counts.get(c, 0) for c in classes]
                
                offset = width * idx - width * (len(self.stats) - 1) / 2
                ax.bar(x + offset, counts, width, label=split.upper())
            
            ax.set_xlabel('Class', fontsize=12)
            ax.set_ylabel('Number of Images', fontsize=12)
            ax.set_title('Class Distribution Across Splits', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(classes, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            save_path = self.output_dir / "split_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved split comparison plot: {save_path}")
    
    def save_report(self):
        """Save comprehensive analysis report."""
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            return obj
        
        # Save JSON report
        report_path = self.output_dir / "dataset_analysis.json"
        with open(report_path, 'w') as f:
            json.dump(convert_to_native(self.stats), f, indent=2)
        print(f"\n✓ Saved analysis report: {report_path}")
        
        # Save text summary
        summary_path = self.output_dir / "dataset_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("ResoMap Dataset Analysis Summary\n")
            f.write("="*60 + "\n\n")
            
            classes = self.get_class_names()
            f.write(f"Number of classes: {len(classes)}\n")
            f.write(f"Classes: {', '.join(classes)}\n\n")
            
            for split, stats in self.stats.items():
                f.write(f"{split.upper()} Split\n")
                f.write("-"*60 + "\n")
                f.write(f"Total images: {stats.get('total_images', 0):,}\n\n")
                
                if 'class_counts' in stats:
                    df_data = []
                    total = stats['total_images']
                    for cls, count in sorted(stats['class_counts'].items()):
                        pct = (count / total * 100) if total > 0 else 0
                        df_data.append({
                            'Class': cls,
                            'Count': count,
                            'Percentage': f"{pct:.2f}%"
                        })
                    df = pd.DataFrame(df_data)
                    f.write(df.to_string(index=False))
                    f.write("\n\n")
        
        print(f"✓ Saved text summary: {summary_path}")


class ModelSummaryAnalyzer:
    """
    Model architecture analyzer - analyzes models WITHOUT training.
    
    Provides:
    - Parameter counts
    - Computational complexity (GFLOPs)
    - Memory requirements
    - Inference speed profiling
    """
    
    def __init__(self, config: dict, output_dir: str = "summary"):
        """
        Initialize model summary analyzer.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary from config.yaml
        output_dir : str
            Directory to save analysis results
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        device_str = config['system']['device']
        self.device = torch.device(
            device_str if torch.cuda.is_available() else 'cpu'
        )
        print(f"Using device: {self.device}")
        
        # Load models config
        models_config_path = Path("configs/models.yaml")
        with open(models_config_path) as f:
            self.models_config = yaml.safe_load(f)
        
        self.results = []
    
    def analyze_model_architecture(
        self,
        model_name: str,
        resolution: int
    ) -> dict:
        """
        Analyze model architecture using torchinfo.summary().
        
        Parameters
        ----------
        model_name : str
            Model name
        resolution : int
            Input resolution
            
        Returns
        -------
        dict
            Architecture analysis results
        """
        # Build model
        model_cfg = self.models_config[model_name]
        model = build_model(model_cfg, resolution)
        model = model.to(self.device)
        model.eval()
        
        # Get model summary using torchinfo
        print(f"\n{'-'*60}")
        print(f"Model: {model_name} @ {resolution}x{resolution}")
        print(f"{'-'*60}")
        
        # Generate summary with torchinfo
        model_stats = summary(
            model,
            input_size=(1, 3, resolution, resolution),
            device=self.device,
            verbose=0,  # Don't print, we'll extract info
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            row_settings=["var_names"]
        )
        
        # Create hierarchical directory structure: summary/{model}/{resolution}/
        model_dir = self.output_dir / model_name / str(resolution)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed summary to file in model-specific directory
        summary_file = model_dir / "architecture_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(str(summary(
                model,
                input_size=(1, 3, resolution, resolution),
                device=self.device,
                col_names=["input_size", "output_size", "num_params", "mult_adds"],
                row_settings=["var_names"]
            )))
        
        print(f"✓ Detailed summary saved to {summary_file}")
        
        # Extract statistics
        total_params = model_stats.total_params
        trainable_params = model_stats.trainable_params
        total_mult_adds = model_stats.total_mult_adds
        
        # Convert mult-adds to GFLOPs (1 GFLOP = 10^9 FLOPs, mult-add ≈ 2 operations)
        gflops = (total_mult_adds * 2) / 1e9
        
        # Estimate memory (parameters + activations)
        memory_mb = (model_stats.total_input + model_stats.total_output_bytes + 
                    model_stats.total_param_bytes) / (1024 ** 2)
        
        # Print summary
        print(f"\nSummary Statistics:")
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Trainable Parameters: {trainable_params:,}")
        print(f"  GFLOPs: {gflops:.2f}")
        print(f"  Memory (MB): {memory_mb:.2f}")
        
        return {
            'model': model_name,
            'resolution': resolution,
            'total_params': int(total_params),
            'trainable_params': int(trainable_params),
            'gflops': float(gflops),
            'memory_mb': float(memory_mb),
            'mult_adds': int(total_mult_adds)
        }
    
    def analyze_models(self, models: List[str], resolutions: List[int]):
        """
        Analyze multiple models across resolutions.
        
        Parameters
        ----------
        models : list
            List of model names to analyze
        resolutions : list
            List of resolutions to analyze
        """
        print(f"\n{'='*60}")
        print(f"Model Architecture Summary")
        print(f"{'='*60}")
        print(f"Models: {models}")
        print(f"Resolutions: {resolutions}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        self.results = []
        
        for model_name in tqdm(models, desc="Analyzing models"):
            if model_name not in self.models_config:
                print(f"[Warning] Model '{model_name}' not found in models.yaml, skipping...")
                continue
            
            for resolution in tqdm(resolutions, desc=f"{model_name}", leave=False):
                try:
                    result = self.analyze_model_architecture(model_name, resolution)
                    self.results.append(result)
                    
                except Exception as e:
                    print(f"\n[Error] Failed to analyze {model_name}@{resolution}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Generate reports
        self.generate_reports()
    
    def generate_reports(self):
        """Generate comprehensive summary reports and tables."""
        if not self.results:
            print("No results to generate reports from.")
            return
        
        df = pd.DataFrame(self.results)
        
        # Save detailed results
        df.to_csv(self.output_dir / "model_summary_detailed.csv", index=False)
        
        with open(self.output_dir / "model_summary_detailed.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Saved detailed results to {self.output_dir}")
        
        # Create pivot tables for each metric
        metrics = {
            'total_params': 'Total Parameters',
            'gflops': 'GFLOPs',
            'memory_mb': 'Memory (MB)',
            'mult_adds': 'Mult-Adds'
        }
        
        for metric, label in metrics.items():
            if metric not in df.columns:
                continue
            
            pivot = df.pivot(index='model', columns='resolution', values=metric)
            pivot.to_csv(self.output_dir / f"summary_{metric}.csv")
            
            print(f"\n{'='*60}")
            print(f"{label}")
            print(f"{'='*60}")
            print(pivot.to_string())
        
        # Generate text summary
        summary_path = self.output_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write("ResoMap Model Architecture Summary\n")
            f.write("="*60 + "\n\n")
            
            for metric, label in metrics.items():
                if metric not in df.columns:
                    continue
                    
                pivot = df.pivot(index='model', columns='resolution', values=metric)
                f.write(f"\n{label}\n")
                f.write("-"*60 + "\n")
                f.write(pivot.to_string())
                f.write("\n\n")
        
        print(f"\n✓ Saved text summary to {summary_path}")


class ResolutionExplainabilityAnalyzer:
    """
    Comprehensive analyzer for resolution effects on model performance
    and explainability.
    """
    
    def __init__(self, config: dict, output_dir: str = "results"):
        """
        Initialize analyzer.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary from config.yaml
        output_dir : str
            Directory to save analysis results
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        device_str = config['system']['device']
        self.device = torch.device(
            device_str if torch.cuda.is_available() else 'cpu'
        )
        print(f"Using device: {self.device}")
        
        # Setup results storage
        self.results = {
            'performance': {},
            'explainability': {},
            'profiling': {}
        }
    
    def load_model_for_resolution(
        self,
        model_name: str,
        resolution: int,
        checkpoint_dir: Optional[str] = None
    ) -> nn.Module:
        """
        Load model trained at specific resolution.
        
        Parameters
        ----------
        model_name : str
            Model architecture name
        resolution : int
            Training resolution
        checkpoint_dir : str, optional
            Directory containing checkpoints
        
        Returns
        -------
        nn.Module
            Loaded model
        """
        # Load model config from models.yaml
        model_config_path = Path("configs/models.yaml")
        with open(model_config_path) as f:
            all_models = yaml.safe_load(f)
        
        # Extract specific model config
        if model_name not in all_models:
            raise ValueError(f"Model '{model_name}' not found in configs/models.yaml")
        
        model_cfg = all_models[model_name]
        
        # Build model
        model = build_model(model_cfg, resolution=resolution)
        model = model.to(self.device)
        
        # Load checkpoint if available
        if checkpoint_dir is not None:
            checkpoint_path = self._find_best_checkpoint(checkpoint_dir, model_name, resolution)
            if checkpoint_path:
                checkpoint = load_checkpoint(checkpoint_path, model, device=self.device)
                print(f"Loaded checkpoint for {model_name} at resolution {resolution}")
            else:
                print(f"Warning: No checkpoint found for {model_name} at resolution {resolution}")
        
        return model
    
    def _find_best_checkpoint(self, checkpoint_dir: str, model_name: str, resolution: int) -> Optional[Path]:
        """
        Find best checkpoint for given model and resolution.
        
        Searches in hierarchical structure:
        1. checkpoints/{model_family}/{model_name}/{resolution}/best_model.pt
        2. checkpoints/{model_family}/{model_name}/{resolution}/*.pt
        3. Legacy flat structure: checkpoints/{model_name}_{resolution}/best_model.pt
        """
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            return None
        
        # Determine model family
        if model_name.startswith('vgg'):
            model_family = 'vgg'
        elif model_name.startswith('resnet'):
            model_family = 'resnet'
        elif model_name.startswith('mobilenet'):
            model_family = 'mobilenet'
        elif model_name.startswith('efficientnet'):
            model_family = 'efficientnet'
        elif 'cnn' in model_name.lower():
            model_family = 'custom_cnn'
        else:
            model_family = 'other'
        
        # Try hierarchical structure first
        hierarchical_dir = checkpoint_path / model_family / model_name / str(resolution)
        if hierarchical_dir.exists():
            # Look for best_model.pt
            best_model = hierarchical_dir / "best_model.pt"
            if best_model.exists():
                return best_model
            
            # Otherwise look for any .pt file
            checkpoints = list(hierarchical_dir.glob("*.pt"))
            if checkpoints:
                return sorted(checkpoints)[-1]
        
        # Fallback: legacy flat structure
        model_checkpoint_dir = checkpoint_path / f"{model_name}_{resolution}"
        if model_checkpoint_dir.exists():
            # Look for best_model.pt first
            best_model = model_checkpoint_dir / "best_model.pt"
            if best_model.exists():
                return best_model
            
            # Otherwise look for any .pt file
            checkpoints = list(model_checkpoint_dir.glob("*.pt"))
            if checkpoints:
                return sorted(checkpoints)[-1]
        
        # Final fallback: look for checkpoints matching pattern in root
        checkpoints = list(checkpoint_path.glob(f"*{model_name}*{resolution}*.pt"))
        if not checkpoints:
            return None
        
        return sorted(checkpoints)[-1]
    
    def evaluate_model(
        self,
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        resolution: int
    ) -> dict:
        """
        Evaluate model performance.
        
        Parameters
        ----------
        model : nn.Module
            Model to evaluate
        test_loader : DataLoader
            Test data loader
        resolution : int
            Image resolution
        
        Returns
        -------
        dict
            Performance metrics
        """
        model.eval()
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Evaluating @ {resolution}", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        
        return {
            'resolution': resolution,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'avg_loss': total_loss / len(test_loader)
        }
    
    def analyze_resolution_sweep(
        self,
        model_name: str,
        resolutions: list,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Analyze model performance across multiple resolutions.
        
        Parameters
        ----------
        model_name : str
            Model architecture name
        resolutions : list
            List of resolutions to analyze
        checkpoint_dir : str, optional
            Directory with checkpoints
        """
        print(f"\n{'='*60}")
        print(f"Resolution Sweep Analysis: {model_name}")
        print(f"{'='*60}\n")
        
        performance_results = []
        profiling_results = []
        
        for resolution in tqdm(resolutions, desc=f"Analyzing {model_name} resolutions", leave=False):
            print(f"\nAnalyzing resolution: {resolution}x{resolution}")
            
            # Load model
            model = self.load_model_for_resolution(model_name, resolution, checkpoint_dir)
            
            # Load data
            train_loader, val_loader, test_loader = get_data_loaders(
                data_dir=self.config['data']['raw_path'],
                resolution=resolution,
                batch_size=self.config['training']['batch_size'],
                num_workers=self.config['data']['num_workers'],
                augment=False,
                pin_memory=self.config['system']['pin_memory']
            )
            
            # Evaluate performance
            perf_metrics = self.evaluate_model(model, test_loader, resolution)
            performance_results.append(perf_metrics)
            
            # Profile model
            prof_results = profile_model(
                model, 
                self.device, 
                test_loader,
                warmup=self.config['system']['warmup_runs'],
                runs=self.config['system']['num_profiling_runs'],
                track_activation_memory=self.config['system']['track_activation_memory'],
                batch_size=self.config['training']['batch_size']
            )
            prof_results['resolution'] = resolution
            profiling_results.append(prof_results)
            
            # Model complexity
            complexity = get_model_complexity(
                model,
                input_shape=(1, 3, resolution, resolution),
                device=self.device
            )
            
            print(f"\nPerformance @ {resolution}:")
            print(f"  Accuracy: {perf_metrics['accuracy']:.4f}")
            print(f"  F1-Score: {perf_metrics['f1_score']:.4f}")
            print(f"\nProfiling @ {resolution}:")
            print(f"  Inference Time: {prof_results['avg_time_sec']:.4f}s")
            print(f"  Throughput: {prof_results['throughput_samples_sec']:.2f} samples/s")
            if 'gpu_memory_peak_mb' in prof_results:
                print(f"  GPU Memory: {prof_results['gpu_memory_peak_mb']:.2f} MB")
            print(f"\nComplexity @ {resolution}:")
            print(f"  GFLOPs: {complexity['gflops']:.2f}")
        
        # Store results
        self.results['performance'][model_name] = performance_results
        self.results['profiling'][model_name] = profiling_results
        
        # Generate visualizations
        self._plot_resolution_performance(model_name, performance_results)
        self._plot_resolution_profiling(model_name, profiling_results)
    
    def analyze_explainability(
        self,
        model_name: str,
        resolutions: list,
        checkpoint_dir: Optional[str] = None,
        num_samples: int = 20
    ):
        """
        Analyze explainability across resolutions.
        
        Parameters
        ----------
        model_name : str
            Model architecture name
        resolutions : list
            Resolutions to analyze
        checkpoint_dir : str, optional
            Checkpoint directory
        num_samples : int
            Number of samples to analyze
        """
        print(f"\n{'='*60}")
        print(f"Explainability Analysis: {model_name}")
        print(f"{'='*60}\n")
        
        exp_config = self.config.get('explainability', {})
        if not exp_config.get('enabled', False):
            print("Explainability disabled in config")
            return
        
        methods = exp_config.get('methods', ['gradcam'])
        
        # Load test data at first resolution to get samples
        first_res = resolutions[0]
        _, _, test_loader = get_data_loaders(
            data_dir=self.config['data']['raw_path'],
            resolution=first_res,
            batch_size=num_samples,
            num_workers=self.config['data']['num_workers'],
            augment=False,
            pin_memory=self.config['system']['pin_memory']
        )
        
        # Get sample batch
        sample_images, sample_labels = next(iter(test_loader))
        sample_images = sample_images[:num_samples]
        sample_labels = sample_labels[:num_samples]
        
        # Analyze each resolution
        for method in tqdm(methods, desc="Processing explainability methods", leave=False):
            print(f"\nGenerating {method} explanations across resolutions...")
            
            images_dict = {}
            explainers = {}
            
            for resolution in tqdm(resolutions, desc=f"Processing {method} for resolutions", leave=False):
                # Load model
                model = self.load_model_for_resolution(model_name, resolution, checkpoint_dir)
                
                # Resize images to current resolution
                from torchvision.transforms import Resize
                resizer = Resize((resolution, resolution))
                resized_images = torch.stack([resizer(img) for img in sample_images])
                images_dict[resolution] = resized_images
                
                # Create explainer
                target_layer = get_target_layer(model, model_name)
                explainer = ModelExplainer(
                    model,
                    self.device,
                    target_layer=target_layer
                )
                explainers[resolution] = explainer
            
            # Generate comparison visualizations
            first_explainer = explainers[resolutions[0]]
            save_dir = self.output_dir / f"{model_name}_{method}_comparison"
            
            first_explainer.compare_resolutions(
                images_dict,
                sample_labels,
                method=method,
                num_samples=min(5, num_samples),
                save_dir=save_dir
            )
            
            print(f"Saved {method} comparisons to {save_dir}")
    
    def _plot_resolution_performance(self, model_name: str, results: list):
        """Plot performance metrics vs resolution."""
        df = pd.DataFrame(results)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_name} - Performance vs Resolution', fontsize=16, fontweight='bold')
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for ax, metric, title in zip(axes.flat, metrics, titles):
            ax.plot(df['resolution'], df[metric], marker='o', linewidth=2, markersize=8)
            ax.set_xlabel('Resolution', fontsize=12)
            ax.set_ylabel(title, fontsize=12)
            ax.set_title(title, fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(df['resolution'])
        
        plt.tight_layout()
        save_path = self.output_dir / f"{model_name}_performance.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved performance plot to {save_path}")
    
    def _plot_resolution_profiling(self, model_name: str, results: list):
        """Plot profiling metrics vs resolution."""
        df = pd.DataFrame(results)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'{model_name} - Profiling vs Resolution', fontsize=16, fontweight='bold')
        
        # Inference time
        axes[0].plot(df['resolution'], df['avg_time_sec'], marker='o', linewidth=2, markersize=8)
        axes[0].set_xlabel('Resolution', fontsize=12)
        axes[0].set_ylabel('Inference Time (s)', fontsize=12)
        axes[0].set_title('Inference Time per Batch', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(df['resolution'])
        
        # Throughput
        axes[1].plot(df['resolution'], df['throughput_samples_sec'], marker='o', linewidth=2, markersize=8, color='green')
        axes[1].set_xlabel('Resolution', fontsize=12)
        axes[1].set_ylabel('Throughput (samples/s)', fontsize=12)
        axes[1].set_title('Throughput', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks(df['resolution'])
        
        # Memory
        if 'gpu_memory_peak_mb' in df.columns:
            axes[2].plot(df['resolution'], df['gpu_memory_peak_mb'], marker='o', linewidth=2, markersize=8, color='red')
            axes[2].set_ylabel('GPU Memory (MB)', fontsize=12)
            axes[2].set_title('Peak GPU Memory', fontsize=14)
        else:
            axes[2].plot(df['resolution'], df['peak_cpu_memory_mb'], marker='o', linewidth=2, markersize=8, color='orange')
            axes[2].set_ylabel('CPU Memory (MB)', fontsize=12)
            axes[2].set_title('Peak CPU Memory', fontsize=14)
        
        axes[2].set_xlabel('Resolution', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xticks(df['resolution'])
        
        plt.tight_layout()
        save_path = self.output_dir / f"{model_name}_profiling.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved profiling plot to {save_path}")
    
    def generate_report(self):
        """Generate comprehensive analysis report."""
        report_path = self.output_dir / "analysis_report.json"
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nSaved analysis report to {report_path}")
        
        # Generate summary
        summary_path = self.output_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write("ResoMap Resolution-Explainability Analysis Summary\n")
            f.write("=" * 60 + "\n\n")
            
            for model_name, perf_results in self.results['performance'].items():
                f.write(f"Model: {model_name}\n")
                f.write("-" * 60 + "\n")
                
                df = pd.DataFrame(perf_results)
                f.write(df.to_string(index=False))
                f.write("\n\n")
                
                # Best resolution
                best_idx = df['accuracy'].idxmax()
                best_res = df.loc[best_idx, 'resolution']
                best_acc = df.loc[best_idx, 'accuracy']
                f.write(f"Best Resolution: {best_res}x{best_res} (Accuracy: {best_acc:.4f})\n\n")
        
        print(f"Saved summary to {summary_path}")
