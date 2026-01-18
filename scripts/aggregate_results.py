"""
============================================================
scripts/aggregate_results.py
============================================================

Script to aggregate and export experiment results.

Generates:
1. CSV file with all test results (all_results.csv)
2. Summary statistics (results_summary.json)
3. Detailed report (detailed_report.txt)
4. Console summary

Usage:
    # Aggregate all results
    python scripts/aggregate_results.py
    
    # Generate specific outputs
    python scripts/aggregate_results.py --csv-only
    python scripts/aggregate_results.py --summary-only
    python scripts/aggregate_results.py --report-only
"""

import sys
import argparse
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.results import ResultsAggregator


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ResoMap Results Aggregation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate all outputs:
    python scripts/aggregate_results.py
  
  Generate only CSV:
    python scripts/aggregate_results.py --csv-only
  
  Generate only summary:
    python scripts/aggregate_results.py --summary-only
  
  Generate only detailed report:
    python scripts/aggregate_results.py --report-only
  
  Generate all with custom filenames:
    python scripts/aggregate_results.py --csv results.csv --summary summary.json
        """
    )
    
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Generate only CSV aggregation"
    )
    
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Generate only summary statistics"
    )
    
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate only detailed report"
    )
    
    parser.add_argument(
        "--csv",
        type=str,
        default="all_results.csv",
        help="Output CSV filename (default: all_results.csv)"
    )
    
    parser.add_argument(
        "--summary",
        type=str,
        default="results_summary.json",
        help="Output summary filename (default: results_summary.json)"
    )
    
    parser.add_argument(
        "--report",
        type=str,
        default="detailed_report.txt",
        help="Output report filename (default: detailed_report.txt)"
    )
    
    parser.add_argument(
        "--no-console",
        action="store_true",
        help="Skip console summary output"
    )
    
    return parser.parse_args()


def main():
    """Main aggregation function."""
    args = parse_arguments()
    
    print(f"\n{'='*70}")
    print("ResoMap Results Aggregation")
    print(f"{'='*70}\n")
    
    # Initialize aggregator
    aggregator = ResultsAggregator(PROJECT_ROOT)
    
    # Determine what to generate
    generate_all = not (args.csv_only or args.summary_only or args.report_only)
    
    # Generate outputs
    if generate_all or args.csv_only:
        print("[*] Aggregating results to CSV...")
        aggregator.aggregate_to_csv(args.csv)
    
    if generate_all or args.summary_only:
        print("[*] Generating summary statistics...")
        aggregator.generate_summary(args.summary)
    
    if generate_all or args.report_only:
        print("[*] Generating detailed report...")
        aggregator.generate_detailed_report(args.report)
    
    # Print console summary
    if not args.no_console:
        print("\n[*] Printing console summary...")
        aggregator.print_summary_to_console()
    
    print(f"{'='*70}")
    print("Aggregation Complete!")
    print(f"Results saved to: {aggregator.results_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
