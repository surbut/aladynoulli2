#!/usr/bin/env python3
"""
Load Delphi and Cox model comparison results.

This script loads pre-computed comparison results between Aladynoulli and:
- Delphi model (for various diseases)
- Cox proportional hazards models (if available)

The results should be in CSV format with columns for disease, AUC, etc.

Usage:
    python load_delphi_cox_comparisons.py
"""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path

def load_delphi_comparison(delphi_file_path):
    """
    Load Delphi comparison results.
    
    Expected format: CSV with columns for disease, AUC, CI_lower, CI_upper, etc.
    """
    if not Path(delphi_file_path).exists():
        print(f"Warning: Delphi comparison file not found: {delphi_file_path}")
        return None
    
    print(f"Loading Delphi comparison from: {delphi_file_path}")
    df = pd.read_csv(delphi_file_path)
    print(f"  Loaded {len(df)} disease comparisons")
    return df

def load_cox_comparison(cox_file_path):
    """
    Load Cox model comparison results.
    
    Expected format: CSV with columns for disease, AUC, CI_lower, CI_upper, etc.
    """
    if not Path(cox_file_path).exists():
        print(f"Warning: Cox comparison file not found: {cox_file_path}")
        return None
    
    print(f"Loading Cox comparison from: {cox_file_path}")
    df = pd.read_csv(cox_file_path)
    print(f"  Loaded {len(df)} disease comparisons")
    return df

def main():
    parser = argparse.ArgumentParser(description="Load Delphi and Cox comparison results.")
    parser.add_argument('--delphi_file', type=str,
                        default='/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/results/comparisons/delphi_comparison.csv',
                        help='Path to Delphi comparison CSV file')
    parser.add_argument('--cox_file', type=str,
                        default='/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/results/comparisons/cox_comparison.csv',
                        help='Path to Cox comparison CSV file')
    parser.add_argument('--output_dir', type=str,
                        default='/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/results/comparisons/',
                        help='Output directory for loaded results')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("LOADING DELPHI AND COX COMPARISONS")
    print("="*80)
    
    # Load Delphi comparison
    delphi_df = load_delphi_comparison(args.delphi_file)
    if delphi_df is not None:
        delphi_output = output_dir / 'delphi_comparison_loaded.csv'
        delphi_df.to_csv(delphi_output, index=False)
        print(f"✓ Saved Delphi comparison to: {delphi_output}")
    
    # Load Cox comparison
    cox_df = load_cox_comparison(args.cox_file)
    if cox_df is not None:
        cox_output = output_dir / 'cox_comparison_loaded.csv'
        cox_df.to_csv(cox_output, index=False)
        print(f"✓ Saved Cox comparison to: {cox_output}")
    
    print("\n" + "="*80)
    print("LOADING COMPLETE")
    print("="*80)
    
    # Return dictionaries for notebook use
    results = {
        'delphi': delphi_df,
        'cox': cox_df
    }
    
    return results

if __name__ == '__main__':
    results = main()
    print("\nResults loaded. Use 'delphi' and 'cox' keys to access DataFrames.")


