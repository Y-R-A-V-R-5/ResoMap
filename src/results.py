"""
============================================================
src/results.py
============================================================

Results aggregation and export utilities for ResoMap experiments.

Provides functions to:
1. Aggregate individual test results into a combined CSV file
2. Generate summary statistics across models and resolutions
3. Create comparison reports and visualizations
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class ResultsAggregator:
    """Aggregate and export experiment results."""
    
    def __init__(self, project_root: Path):
        """
        Initialize results aggregator.
        
        Parameters
        ----------
        project_root : Path
            Path to project root directory
        """
        self.project_root = Path(project_root)
        self.results_dir = self.project_root / "results"
        self.test_results_dir = self.results_dir / "test_results"
    
    def aggregate_to_csv(self, output_filename: str = "all_results.csv") -> Path:
        """
        Aggregate all individual JSON results into a single CSV file.
        
        Parameters
        ----------
        output_filename : str
            Name of output CSV file
            
        Returns
        -------
        Path
            Path to generated CSV file
        """
        if not self.test_results_dir.exists():
            print("[Info] No test results directory found yet")
            return None
        
        # Collect all JSON results
        all_results = []
        json_files = list(self.test_results_dir.glob("*.json"))
        
        if not json_files:
            print("[Info] No test results found")
            return None
        
        print(f"[Results] Aggregating {len(json_files)} result files...")
        
        for json_file in sorted(json_files):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Flatten the nested structure
                row = {
                    'model': data.get('model'),
                    'resolution': data.get('resolution'),
                    'timestamp': data.get('timestamp'),
                    'best_val_loss': data.get('best_val_loss'),
                }
                
                # Add test metrics
                test_metrics = data.get('test_metrics', {})
                for key, value in test_metrics.items():
                    row[f'test_{key}'] = value
                
                # Add profiling metrics
                profiling = data.get('profiling', {})
                for key, value in profiling.items():
                    row[f'profile_{key}'] = value
                
                all_results.append(row)
            
            except Exception as e:
                print(f"[Warning] Could not load {json_file}: {e}")
        
        if not all_results:
            print("[Warning] No valid results to aggregate")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(all_results)
        
        # Sort by model and resolution
        df = df.sort_values(['model', 'resolution']).reset_index(drop=True)
        
        # Save to CSV
        csv_path = self.results_dir / output_filename
        df.to_csv(csv_path, index=False)
        print(f"[Results] ✓ Saved aggregated results to {csv_path}")
        
        return csv_path
    
    def generate_summary(self, output_filename: str = "results_summary.json") -> Path:
        """
        Generate summary statistics across all results.
        
        Parameters
        ----------
        output_filename : str
            Name of output summary file
            
        Returns
        -------
        Path
            Path to generated summary file
        """
        csv_path = self.results_dir / "all_results.csv"
        
        if not csv_path.exists():
            print("[Info] Running aggregation first...")
            self.aggregate_to_csv()
        
        if not csv_path.exists():
            print("[Warning] No results to summarize")
            return None
        
        # Load aggregated results
        df = pd.read_csv(csv_path)
        
        # Generate statistics
        summary = {
            "generated_at": datetime.now().isoformat(),
            "total_experiments": len(df),
            "models": sorted(df['model'].unique().tolist()),
            "resolutions": sorted(df['resolution'].unique().tolist()),
            "statistics": {}
        }
        
        # Per-model statistics
        summary["statistics"]["by_model"] = {}
        for model in sorted(df['model'].unique()):
            model_data = df[df['model'] == model]
            
            summary["statistics"]["by_model"][model] = {
                "num_experiments": len(model_data),
                "avg_test_accuracy": float(model_data['test_accuracy'].mean()),
                "best_test_accuracy": float(model_data['test_accuracy'].max()),
                "worst_test_accuracy": float(model_data['test_accuracy'].min()),
                "avg_test_f1": float(model_data['test_f1_score'].mean()),
                "avg_inference_time_sec": float(model_data['profile_avg_time_sec'].mean()),
                "avg_throughput_samples_sec": float(model_data['profile_throughput_samples_sec'].mean()),
            }
        
        # Per-resolution statistics
        summary["statistics"]["by_resolution"] = {}
        for res in sorted(df['resolution'].unique()):
            res_data = df[df['resolution'] == res]
            
            summary["statistics"]["by_resolution"][str(res)] = {
                "num_experiments": len(res_data),
                "avg_test_accuracy": float(res_data['test_accuracy'].mean()),
                "best_test_accuracy": float(res_data['test_accuracy'].max()),
                "worst_test_accuracy": float(res_data['test_accuracy'].min()),
                "avg_test_f1": float(res_data['test_f1_score'].mean()),
                "avg_inference_time_sec": float(res_data['profile_avg_time_sec'].mean()),
                "avg_throughput_samples_sec": float(res_data['profile_throughput_samples_sec'].mean()),
            }
        
        # Best performing model-resolution combinations
        df_sorted = df.sort_values('test_accuracy', ascending=False)
        summary["top_5_results"] = df_sorted[['model', 'resolution', 'test_accuracy', 'test_f1_score']].head(5).to_dict('records')
        
        # Save summary
        summary_path = self.results_dir / output_filename
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[Results] ✓ Saved summary to {summary_path}")
        return summary_path
    
    def generate_detailed_report(self, output_filename: str = "detailed_report.txt") -> Path:
        """
        Generate a detailed text report of all results.
        
        Parameters
        ----------
        output_filename : str
            Name of output report file
            
        Returns
        -------
        Path
            Path to generated report file
        """
        csv_path = self.results_dir / "all_results.csv"
        
        if not csv_path.exists():
            print("[Info] Running aggregation first...")
            self.aggregate_to_csv()
        
        if not csv_path.exists():
            print("[Warning] No results to report")
            return None
        
        # Load results
        df = pd.read_csv(csv_path)
        
        # Generate report
        report_path = self.results_dir / output_filename
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("ResoMap Experiment Results Report\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Experiments: {len(df)}\n")
            f.write(f"Models: {', '.join(sorted(df['model'].unique()))}\n")
            f.write(f"Resolutions: {', '.join(map(str, sorted(df['resolution'].unique())))}\n\n")
            
            # Best overall results
            f.write("-"*70 + "\n")
            f.write("TOP 10 BEST RESULTS\n")
            f.write("-"*70 + "\n\n")
            
            top_10 = df.nlargest(10, 'test_accuracy')[['model', 'resolution', 'test_accuracy', 'test_f1_score', 'best_val_loss']]
            for idx, row in top_10.iterrows():
                f.write(f"  {row['model']:20s} @ {row['resolution']:3d}px")
                f.write(f" | Accuracy: {row['test_accuracy']:.4f}")
                f.write(f" | F1: {row['test_f1_score']:.4f}")
                f.write(f" | Val Loss: {row['best_val_loss']:.4f}\n")
            
            f.write("\n")
            
            # Per-model summary
            f.write("-"*70 + "\n")
            f.write("RESULTS BY MODEL\n")
            f.write("-"*70 + "\n\n")
            
            for model in sorted(df['model'].unique()):
                model_data = df[df['model'] == model]
                f.write(f"{model}:\n")
                f.write(f"  Experiments: {len(model_data)}\n")
                f.write(f"  Avg Accuracy: {model_data['test_accuracy'].mean():.4f}\n")
                f.write(f"  Best Accuracy: {model_data['test_accuracy'].max():.4f} @ {model_data.loc[model_data['test_accuracy'].idxmax(), 'resolution']:.0f}px\n")
                f.write(f"  Avg F1-Score: {model_data['test_f1_score'].mean():.4f}\n")
                f.write(f"  Avg Inference Time: {model_data['profile_avg_time_sec'].mean():.6f}s\n")
                f.write("\n")
            
            # Per-resolution summary
            f.write("-"*70 + "\n")
            f.write("RESULTS BY RESOLUTION\n")
            f.write("-"*70 + "\n\n")
            
            for res in sorted(df['resolution'].unique()):
                res_data = df[df['resolution'] == res]
                f.write(f"{res}x{res}:\n")
                f.write(f"  Experiments: {len(res_data)}\n")
                f.write(f"  Avg Accuracy: {res_data['test_accuracy'].mean():.4f}\n")
                f.write(f"  Best Accuracy: {res_data['test_accuracy'].max():.4f} ({res_data.loc[res_data['test_accuracy'].idxmax(), 'model']})\n")
                f.write(f"  Avg F1-Score: {res_data['test_f1_score'].mean():.4f}\n")
                f.write(f"  Avg Inference Time: {res_data['profile_avg_time_sec'].mean():.6f}s\n")
                f.write("\n")
            
            # Detailed table
            f.write("-"*70 + "\n")
            f.write("DETAILED RESULTS TABLE\n")
            f.write("-"*70 + "\n\n")
            
            # Select key columns
            cols_to_show = ['model', 'resolution', 'test_accuracy', 'test_precision', 
                           'test_recall', 'test_f1_score', 'profile_avg_time_sec', 'best_val_loss']
            available_cols = [c for c in cols_to_show if c in df.columns]
            
            f.write(df[available_cols].to_string(index=False))
            f.write("\n\n")
            
            f.write("="*70 + "\n")
            f.write("End of Report\n")
            f.write("="*70 + "\n")
        
        print(f"[Results] ✓ Saved detailed report to {report_path}")
        return report_path
    
    def print_summary_to_console(self):
        """Print summary statistics to console."""
        csv_path = self.results_dir / "all_results.csv"
        
        if not csv_path.exists():
            print("[Info] No aggregated results found")
            return
        
        df = pd.read_csv(csv_path)
        
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        print(f"\nTotal Experiments: {len(df)}")
        print(f"Models: {len(df['model'].unique())}")
        print(f"Resolutions: {len(df['resolution'].unique())}")
        
        print("\n" + "-"*70)
        print("BEST RESULTS")
        print("-"*70)
        
        top_5 = df.nlargest(5, 'test_accuracy')[['model', 'resolution', 'test_accuracy', 'test_f1_score']]
        for idx, row in top_5.iterrows():
            print(f"  {row['model']:20s} @ {row['resolution']:3.0f}px | Accuracy: {row['test_accuracy']:.4f} | F1: {row['test_f1_score']:.4f}")
        
        print("\n" + "-"*70)
        print("BY MODEL")
        print("-"*70)
        
        for model in sorted(df['model'].unique()):
            model_data = df[df['model'] == model]
            print(f"  {model:20s} | Avg Acc: {model_data['test_accuracy'].mean():.4f} | Best: {model_data['test_accuracy'].max():.4f}")
        
        print("\n" + "-"*70)
        print("BY RESOLUTION")
        print("-"*70)
        
        for res in sorted(df['resolution'].unique()):
            res_data = df[df['resolution'] == res]
            print(f"  {res:3.0f}x{res:3.0f} | Avg Acc: {res_data['test_accuracy'].mean():.4f} | Best: {res_data['test_accuracy'].max():.4f}")
        
        print("\n" + "="*70 + "\n")
