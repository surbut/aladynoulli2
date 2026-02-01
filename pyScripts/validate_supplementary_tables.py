#!/usr/bin/env python3
"""
Validate that Supplementary_Tables.xlsx matches the source CSV files in results/.
"""

import pandas as pd
from pathlib import Path
import numpy as np

# Paths
EXCEL_PATH = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/Apps/Overleaf/Aladynoulli_Nature/SuppDataFiles/Supplementary_Tables.xlsx')
RESULTS_DIR = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results')

def compare_dataframes(df1, df2, name1="Excel", name2="CSV", key_cols=None):
    """Compare two dataframes and report differences."""
    print(f"\n{'='*60}")
    print(f"Comparing {name1} vs {name2}")
    print(f"{'='*60}")
    
    print(f"{name1} shape: {df1.shape}")
    print(f"{name2} shape: {df2.shape}")
    
    print(f"\n{name1} columns: {list(df1.columns)[:10]}...")
    print(f"{name2} columns: {list(df2.columns)[:10]}...")
    
    # Check if shapes match
    if df1.shape != df2.shape:
        print(f"⚠️  Shapes differ!")
    
    # Try to find common columns
    common_cols = set(df1.columns) & set(df2.columns)
    if common_cols:
        print(f"\nCommon columns: {len(common_cols)}")
        
        # Compare numeric columns
        for col in common_cols:
            if df1[col].dtype in ['float64', 'int64'] and df2[col].dtype in ['float64', 'int64']:
                if len(df1) == len(df2):
                    diff = np.abs(df1[col].values - df2[col].values)
                    max_diff = np.nanmax(diff)
                    if max_diff > 0.001:
                        print(f"  ⚠️  {col}: max diff = {max_diff:.6f}")
                    else:
                        print(f"  ✓ {col}: matches (max diff = {max_diff:.2e})")

def main():
    print("Loading Excel file...")
    
    # Get all sheet names
    xl = pd.ExcelFile(EXCEL_PATH)
    print(f"\nExcel sheets: {xl.sheet_names}")
    
    # Load each sheet and show summary
    for sheet in xl.sheet_names:
        df = pd.read_excel(EXCEL_PATH, sheet_name=sheet)
        print(f"\n--- Sheet: {sheet} ---")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)[:8]}{'...' if len(df.columns) > 8 else ''}")
        print(f"First few rows:")
        print(df.head(3).to_string())
    
    print("\n" + "="*80)
    print("COMPARING WITH SOURCE CSV FILES")
    print("="*80)
    
    # Compare gamma associations if exists
    gamma_csv = RESULTS_DIR / 'paper_figs/fig4/gamma_associations.csv'
    if gamma_csv.exists():
        gamma_df = pd.read_csv(gamma_csv)
        print(f"\n--- gamma_associations.csv ---")
        print(f"Shape: {gamma_df.shape}")
        print(f"Columns: {list(gamma_df.columns)}")
        
        # Check if there's a matching sheet
        for sheet in xl.sheet_names:
            if 'gamma' in sheet.lower() or 'prs' in sheet.lower():
                excel_df = pd.read_excel(EXCEL_PATH, sheet_name=sheet)
                compare_dataframes(excel_df, gamma_df, f"Excel:{sheet}", "CSV:gamma_associations")
    
    # Compare performance summary
    perf_csv = RESULTS_DIR / 'paper_figs/fig5/performance_summary_table.csv'
    if perf_csv.exists():
        perf_df = pd.read_csv(perf_csv)
        print(f"\n--- performance_summary_table.csv ---")
        print(f"Shape: {perf_df.shape}")
        print(f"Columns: {list(perf_df.columns)}")
        print(perf_df.head())
        
        for sheet in xl.sheet_names:
            if 'performance' in sheet.lower() or 'auc' in sheet.lower():
                excel_df = pd.read_excel(EXCEL_PATH, sheet_name=sheet)
                compare_dataframes(excel_df, perf_df, f"Excel:{sheet}", "CSV:performance_summary")
    
    # Compare top diseases per signature
    top_diseases_csv = RESULTS_DIR / 'top_diseases_per_signature.csv'
    if top_diseases_csv.exists():
        top_df = pd.read_csv(top_diseases_csv)
        print(f"\n--- top_diseases_per_signature.csv ---")
        print(f"Shape: {top_df.shape}")
        print(f"Columns: {list(top_df.columns)}")
        
        for sheet in xl.sheet_names:
            if 'disease' in sheet.lower() or 'signature' in sheet.lower():
                excel_df = pd.read_excel(EXCEL_PATH, sheet_name=sheet)
                print(f"\nPotential match: {sheet}")
                print(f"Excel shape: {excel_df.shape}, CSV shape: {top_df.shape}")

if __name__ == '__main__':
    main()
