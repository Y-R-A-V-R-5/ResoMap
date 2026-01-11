"""
============================================================
scripts/analysis.py
============================================================

Dataset & Model Analysis Script for ResoMap
------------------------------------------------------------

Executable script that performs analysis without requiring training:
1. Dataset exploration and statistics
2. Model architecture summaries (parameters, GFLOPs, memory, speed)

Note: Resolution performance analysis is done during experiments.

Usage:
    # Full analysis (dataset + model summaries)
    python scripts/analysis.py
    
    # Dataset analysis only
    python scripts/analysis.py --data-analysis-only
    
    # Model summaries only
    python scripts/analysis.py --model-summary-only
"""

import argparse
import yaml
import sys
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Import analyzer classes from src
from src.analysis import DatasetAnalyzer, ModelSummaryAnalyzer


def main():
    """Main execution function for dataset and model analysis."""
    parser = argparse.ArgumentParser(description="ResoMap Dataset & Model Analysis")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='Models to analyze (default: from config)')
    parser.add_argument('--resolutions', type=int, nargs='+', default=None,
                       help='Resolutions to analyze (default: from config)')
    parser.add_argument('--data-analysis-only', action='store_true',
                       help='Only perform dataset analysis (skip model summaries)')
    parser.add_argument('--model-summary-only', action='store_true',
                       help='Only perform model architecture analysis (skip dataset analysis)')
    parser.add_argument('--data-output-dir', type=str, default='analysis',
                       help='Directory to save data analysis results')
    parser.add_argument('--model-summary-dir', type=str, default='summary',
                       help='Directory to save model summary results')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Perform dataset analysis if requested
    if args.data_analysis_only:
        print("\n" + "="*60)
        print("Running Dataset Analysis Only")
        print("="*60)
        
        data_dir = config['data']['raw_path']
        dataset_analyzer = DatasetAnalyzer(data_dir, args.data_output_dir)
        dataset_analyzer.analyze_dataset()
        
        print("\n" + "="*60)
        print("Dataset Analysis Complete!")
        print(f"Results saved to: {args.data_output_dir}")
        print("="*60)
        return
    
    # Perform model summary analysis if requested
    if args.model_summary_only:
        print("\n" + "="*60)
        print("Running Model Architecture Summary")
        print("="*60)
        
        models = args.models or config['sweep']['models']
        resolutions = args.resolutions or config['sweep']['resolutions']
        
        model_analyzer = ModelSummaryAnalyzer(config, args.model_summary_dir)
        model_analyzer.analyze_models(models, resolutions)
        
        print("\n" + "="*60)
        print("Model Summary Complete!")
        print(f"Results saved to: {args.model_summary_dir}")
        print("="*60)
        return
    
    # Full analysis: Dataset + Model Summaries
    print("\n" + "="*60)
    print("Step 1: Dataset Analysis")
    print("="*60)
    data_dir = config['data']['raw_path']
    dataset_analyzer = DatasetAnalyzer(data_dir, args.data_output_dir)
    dataset_analyzer.analyze_dataset()
    
    print("\n" + "="*60)
    print("Step 2: Model Architecture Summary")
    print("="*60)
    
    models = args.models or config['sweep']['models']
    resolutions = args.resolutions or config['sweep']['resolutions']
    
    print(f"ResoMap Analysis")
    print(f"Models: {models}")
    print(f"Resolutions: {resolutions}")
    
    model_analyzer = ModelSummaryAnalyzer(config, args.model_summary_dir)
    model_analyzer.analyze_models(models, resolutions)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"\nResults saved to:")
    print(f"  - Dataset Analysis: {args.data_output_dir}")
    print(f"  - Model Summaries: {args.model_summary_dir}")


if __name__ == "__main__":
    main()
