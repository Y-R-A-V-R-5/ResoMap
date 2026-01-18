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
    parser = argparse.ArgumentParser(
        description="ResoMap Dataset & Model Analysis"
    )

    # ------------------------------------------------------------
    # Configuration files
    # ------------------------------------------------------------
    parser.add_argument(
        '--data-config',
        type=str,
        default='configs/data.yaml',
        help='Path to data configuration file (defines raw_path)'
    )

    parser.add_argument(
        '--sweep-config',
        type=str,
        default='configs/sweep.yaml',
        help='Path to sweep configuration file (models, resolutions)'
    )

    parser.add_argument(
        '--system-config',
        type=str,
        default='configs/system.yaml',
        help='Path to system configuration file'
    )

    # ------------------------------------------------------------
    # Optional CLI overrides
    # ------------------------------------------------------------
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='Override models to analyze (default: from sweep config)'
    )

    parser.add_argument(
        '--resolutions',
        type=int,
        nargs='+',
        default=None,
        help='Override resolutions to analyze (default: from sweep config)'
    )

    # ------------------------------------------------------------
    # Execution modes
    # ------------------------------------------------------------
    parser.add_argument(
        '--data-analysis-only',
        action='store_true',
        help='Only perform dataset analysis'
    )

    parser.add_argument(
        '--model-summary-only',
        action='store_true',
        help='Only perform model architecture analysis'
    )

    # ------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------
    parser.add_argument(
        '--data-output-dir',
        type=str,
        default='analysis',
        help='Directory to save dataset analysis results'
    )

    parser.add_argument(
        '--model-summary-dir',
        type=str,
        default='summary',
        help='Directory to save model summary results'
    )
    
    args = parser.parse_args()
    
    # ------------------------------------------------------------
    # Load Data Configuration
    # ------------------------------------------------------------
    with open(args.data_config) as f:
        data_config = yaml.safe_load(f)

    if 'raw_path' not in data_config:
        raise KeyError(
            "Missing 'raw_path' in Data Configuration.\n"
            f"Loaded file: {args.config}\n"
            f"Available keys: {list(data_config.keys())}"
        )

    data_dir = data_config['raw_path']

    # ------------------------------------------------------------
    # Load Sweep Configuration
    # ------------------------------------------------------------
    with open(args.sweep_config) as f:
        sweep_config = yaml.safe_load(f)

    required_sweep_keys = ['models', 'resolutions']
    missing = [k for k in required_sweep_keys if k not in sweep_config]

    if missing:
        raise KeyError(
            "Invalid Sweep Configuration.\n"
            f"Missing keys: {missing}\n"
            f"Loaded file: {args.sweep_config}\n"
            f"Available keys: {list(sweep_config.keys())}"
        )
    
    # ------------------------------------------------------------
    # Load System Configuration
    # ------------------------------------------------------------
    with open(args.system_config) as f:
        system_raw = yaml.safe_load(f)

    # Wrap into the structure expected by ModelSummaryAnalyzer
    system_config = {'system': system_raw}

    if 'device' not in system_config['system']:
        system_config['system']['device'] = 'cpu'
        print("Warning: 'device' not found in system.yaml. Falling back to CPU.")
    
    # Perform dataset analysis if requested
    if args.data_analysis_only:
        print("\n" + "="*60)
        print("Running Dataset Analysis Only")
        print("="*60)
        
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
        
        models = args.models or sweep_config['models']
        resolutions = args.resolutions or sweep_config['resolutions']

        model_analyzer = ModelSummaryAnalyzer(
            models_config=models_config,
            output_dir=args.model_summary_dir
        )
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
    dataset_analyzer = DatasetAnalyzer(data_dir, args.data_output_dir)
    dataset_analyzer.analyze_dataset()
    
    print("\n" + "="*60)
    print("Step 2: Model Architecture Summary")
    print("="*60)

    models = args.models or sweep_config['models']
    resolutions = args.resolutions or sweep_config['resolutions']
        
    print(f"ResoMap Analysis")
    print(f"Models: {models}")
    print(f"Resolutions: {resolutions}")
    
    model_analyzer = ModelSummaryAnalyzer(
        system_config,
        args.model_summary_dir
    )
    model_analyzer.analyze_models(models, resolutions)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"\nResults saved to:")
    print(f"  - Dataset Analysis: {args.data_output_dir}")
    print(f"  - Model Summaries: {args.model_summary_dir}")


if __name__ == "__main__":
    main()