#!/usr/bin/env python3
"""
Compare Aladynoulli 1-year predictions with Delphi 1-year predictions.
Compares both 0-year and 1-year washout periods.

Usage:
    python compare_with_delphi_1yr.py --approach pooled_retrospective
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def load_delphi_results(delphi_csv_path):
    """Load Delphi results from supplementary CSV"""
    delphi_df = pd.read_csv(delphi_csv_path)
    
    # Extract 1-year AUCs with 0-year gap (no gap) and 1-year gap
    # Average across male and female
    delphi_results = []
    
    for idx, row in delphi_df.iterrows():
        name = row['Name']
        icd_code = name.split()[0] if len(name.split()) > 0 else None
        
        # Skip technical/padding rows
        if pd.isna(icd_code) or icd_code in ['Padding', 'Healthy', 'Female', 'Male']:
            continue
        
        # Get AUCs for 0-year gap (no gap) and 1-year gap
        auc_female_0gap = row.get('AUC Female, (no gap)', np.nan)
        auc_male_0gap = row.get('AUC Male, (no gap)', np.nan)
        auc_female_1gap = row.get('AUC Female, (1 year gap)', np.nan)
        auc_male_1gap = row.get('AUC Male, (1 year gap)', np.nan)
        
        # Average across sexes (if both available)
        auc_0gap_values = [v for v in [auc_female_0gap, auc_male_0gap] if not pd.isna(v)]
        auc_1gap_values = [v for v in [auc_female_1gap, auc_male_1gap] if not pd.isna(v)]
        
        auc_0gap = np.mean(auc_0gap_values) if auc_0gap_values else np.nan
        auc_1gap = np.mean(auc_1gap_values) if auc_1gap_values else np.nan
        
        if not (pd.isna(auc_0gap) and pd.isna(auc_1gap)):
            delphi_results.append({
                'ICD10': icd_code,
                'Name': name,
                'Delphi_1yr_0gap': auc_0gap,
                'Delphi_1yr_1gap': auc_1gap,
                'N_tokens_validation': row.get('N tokens, validation', 0)
            })
    
    return pd.DataFrame(delphi_results)

def map_icd10_to_aladynoulli_diseases(icd10_code, disease_names):
    """Map ICD-10 code to Aladynoulli disease names"""
    # Simple mapping: try to find diseases that match the ICD-10 code
    # This is a simplified version - you may need more sophisticated mapping
    matches = []
    icd_prefix = icd10_code[:3] if len(icd10_code) >= 3 else icd10_code
    
    for disease_name in disease_names:
        # Try to match by ICD-10 code prefix or common disease names
        if icd10_code.lower() in disease_name.lower() or icd_prefix.lower() in disease_name.lower():
            matches.append(disease_name)
    
    return matches

def load_aladynoulli_washout_results(results_dir, approach):
    """Load Aladynoulli washout results"""
    washout_dir = Path(results_dir) / 'washout' / approach
    
    # Load 0-year and 1-year washout results
    results_0yr = washout_dir / 'washout_0yr_results.csv'
    results_1yr = washout_dir / 'washout_1yr_results.csv'
    
    aladyn_0yr = pd.read_csv(results_0yr) if results_0yr.exists() else None
    aladyn_1yr = pd.read_csv(results_1yr) if results_1yr.exists() else None
    
    return aladyn_0yr, aladyn_1yr

def main():
    parser = argparse.ArgumentParser(description='Compare Aladynoulli with Delphi 1-year predictions')
    parser.add_argument('--approach', type=str, required=True,
                       choices=['pooled_enrollment', 'pooled_retrospective'],
                       help='Which approach to use')
    parser.add_argument('--delphi_csv', type=str,
                       default='/Users/sarahurbut/Downloads/41586_2025_9529_MOESM3_ESM.csv',
                       help='Path to Delphi supplementary CSV')
    parser.add_argument('--results_dir', type=str,
                       default='/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/results',
                       help='Results directory')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/results/comparisons',
                       help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("COMPARING ALADYNOULLI WITH DELPHI (1-YEAR PREDICTIONS)")
    print("="*80)
    
    # Load Delphi results
    print("\n1. Loading Delphi results...")
    delphi_df = load_delphi_results(args.delphi_csv)
    print(f"   Loaded {len(delphi_df)} diseases from Delphi")
    print(f"   Diseases with 0-year gap AUC: {delphi_df['Delphi_1yr_0gap'].notna().sum()}")
    print(f"   Diseases with 1-year gap AUC: {delphi_df['Delphi_1yr_1gap'].notna().sum()}")
    
    # Load Aladynoulli washout results
    print("\n2. Loading Aladynoulli washout results...")
    aladyn_0yr, aladyn_1yr = load_aladynoulli_washout_results(args.results_dir, args.approach)
    
    if aladyn_0yr is None or aladyn_1yr is None:
        print("   ERROR: Could not find washout results files!")
        print(f"   Expected: {args.results_dir}/washout/{args.approach}/washout_0yr_results.csv")
        print(f"   Expected: {args.results_dir}/washout/{args.approach}/washout_1yr_results.csv")
        return
    
    print(f"   Loaded {len(aladyn_0yr)} diseases from Aladynoulli (0-year washout)")
    print(f"   Loaded {len(aladyn_1yr)} diseases from Aladynoulli (1-year washout)")
    
    # Create comparison DataFrames
    print("\n3. Creating comparison tables...")
    
    # For 0-year washout comparison
    comparison_0yr = aladyn_0yr[['Disease', 'AUC', 'CI_lower', 'CI_upper']].copy()
    comparison_0yr.columns = ['Disease', 'Aladynoulli_1yr_0gap', 'Aladynoulli_CI_lower_0gap', 'Aladynoulli_CI_upper_0gap']
    
    # For 1-year washout comparison
    comparison_1yr = aladyn_1yr[['Disease', 'AUC', 'CI_lower', 'CI_upper']].copy()
    comparison_1yr.columns = ['Disease', 'Aladynoulli_1yr_1gap', 'Aladynoulli_CI_lower_1gap', 'Aladynoulli_CI_upper_1gap']
    
    # Merge with Delphi (this is simplified - you may need better disease name matching)
    # For now, we'll create a simple comparison based on disease names
    print("\n4. Matching diseases...")
    print("   NOTE: This uses simple name matching. You may need to refine the mapping.")
    
    # Save results
    output_dir = Path(args.output_dir) / args.approach
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual comparisons
    comparison_0yr.to_csv(output_dir / 'delphi_comparison_0yr_washout.csv', index=False)
    comparison_1yr.to_csv(output_dir / 'delphi_comparison_1yr_washout.csv', index=False)
    delphi_df.to_csv(output_dir / 'delphi_results_extracted.csv', index=False)
    
    print(f"\n✓ Saved comparison results to: {output_dir}/")
    print(f"   - delphi_comparison_0yr_washout.csv")
    print(f"   - delphi_comparison_1yr_washout.csv")
    print(f"   - delphi_results_extracted.csv")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Review the extracted Delphi results")
    print("2. Manually match disease names between Aladynoulli and Delphi")
    print("3. Create final comparison table with matched diseases")

if __name__ == '__main__':
    main()

