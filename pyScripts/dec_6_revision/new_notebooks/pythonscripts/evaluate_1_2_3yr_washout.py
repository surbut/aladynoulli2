#!/usr/bin/env python3
"""
Evaluate 1-year and 10-year predictions for 1, 2, 3 year washout periods.

This script:
- Loads pi tensors from washout_1yr_local, washout_2yr_local, washout_3yr_local
- Loads baseline from washout_comparison_10k/no_washout
- Calculates 1-year and 10-year AUCs for all versions
- Creates comparison table

Usage:
    %run /Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/pythonscripts/evaluate_1_2_3yr_washout.py
"""

import sys
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Add path for imports
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/')
from evaluatetdccode import evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year_different_start_end_numeric_sex
from fig5utils import evaluate_major_diseases_wsex_with_bootstrap_from_pi

# Load essentials (disease names, etc.)
def load_essentials():
    """Load model essentials including disease names"""
    essentials_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/model_essentials.pt'
    essentials = torch.load(essentials_path, weights_only=False)
    return essentials

print("="*80)
print("EVALUATING 1, 2, 3 YEAR WASHOUT PREDICTIONS")
print("="*80)

# Configuration
washout_base_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal')
baseline_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/washout_comparison_10k')
output_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/washout_evaluation')
output_dir.mkdir(parents=True, exist_ok=True)

n_bootstraps = 100
subset_size = 10000
start_index = 0
end_index = 10000

print(f"Washout base directory: {washout_base_dir}")
print(f"Baseline directory: {baseline_dir}")
print(f"Output directory: {output_dir}")
print(f"Subset size: {subset_size}")
print("="*80)

# Load common data
print("\nLoading common data...")
Y_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt', weights_only=False)
E_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/E_enrollment_full.pt', weights_only=False)
pce_df_full = pd.read_csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pce_prevent_full.csv')
essentials = load_essentials()
disease_names = essentials['disease_names']

# Subset to first 10K patients
print(f"\nSubsetting to first {subset_size} patients...")
Y_subset = Y_full[:subset_size]
E_subset = E_full[:subset_size]
pce_df_subset = pce_df_full.iloc[:subset_size].reset_index(drop=True)

print(f"After subsetting: Y: {Y_subset.shape[0]}, E: {E_subset.shape[0]}, pce_df: {len(pce_df_subset)}")

# Convert Sex column to numeric if needed
if 'Sex' in pce_df_subset.columns and pce_df_subset['Sex'].dtype == 'object':
    pce_df_subset['sex'] = pce_df_subset['Sex'].map({'Female': 0, 'Male': 1}).astype(int)
elif 'sex' not in pce_df_subset.columns:
    raise ValueError("Need 'Sex' or 'sex' column in pce_df")

# Define washout versions and their paths
washout_configs = {
    'no_washout': {
        'pi_path': baseline_dir / 'no_washout' / f'pi_washout_no_washout_{start_index}_{end_index}.pt',
        'label': 'No Washout (Baseline)'
    },
    '1yr': {
        'pi_path': washout_base_dir / 'washout_1yr_local' / f'pi_washout_1yr_fixedphi_sex_{start_index}_{end_index}_withpcs.pt',
        'label': '1-Year Washout'
    },
    '2yr': {
        'pi_path': washout_base_dir / 'washout_2yr_local' / f'pi_washout_2yr_fixedphi_sex_{start_index}_{end_index}_withpcs.pt',
        'label': '2-Year Washout'
    },
    '3yr': {
        'pi_path': washout_base_dir / 'washout_3yr_local' / f'pi_washout_3yr_fixedphi_sex_{start_index}_{end_index}_withpcs.pt',
        'label': '3-Year Washout'
    }
}

# Process each washout version
all_results = {}

for washout_key, config in washout_configs.items():
    print(f"\n{'='*80}")
    print(f"PROCESSING: {config['label'].upper()}")
    print(f"{'='*80}")
    
    pi_path = config['pi_path']
    
    if not pi_path.exists():
        print(f"⚠️  WARNING: Pi tensor not found: {pi_path}")
        print(f"   Skipping {washout_key}...")
        continue
    
    print(f"Loading pi tensor from: {pi_path}")
    pi_full = torch.load(pi_path, weights_only=False)
    print(f"Pi tensor shape: {pi_full.shape}")
    
    current_results = {}
    
    # 1-year prediction (offset=0)
    print(f"\nProcessing 1-year predictions for {washout_key}...")
    results_1yr = evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year_different_start_end_numeric_sex(
        pi=pi_full,
        Y_100k=Y_subset,
        E_100k=E_subset,
        disease_names=disease_names,
        pce_df=pce_df_subset,
        n_bootstraps=n_bootstraps,
        follow_up_duration_years=1,
        start_offset=0
    )
    current_results['1yr'] = results_1yr
    
    results_df_1yr = pd.DataFrame({
        'Disease': list(results_1yr.keys()),
        'AUC': [r['auc'] for r in results_1yr.values()],
        'CI_lower': [r['ci_lower'] for r in results_1yr.values()],
        'CI_upper': [r['ci_upper'] for r in results_1yr.values()],
        'N_Events': [r['n_events'] for r in results_1yr.values()],
        'Event_Rate': [r['event_rate'] for r in results_1yr.values()]
    })
    results_df_1yr = results_df_1yr.set_index('Disease').sort_values('AUC', ascending=False)
    output_file_1yr = output_dir / f'washout_{washout_key}_1yr_results.csv'
    results_df_1yr.to_csv(output_file_1yr)
    print(f"✓ Saved 1-year results to {output_file_1yr}")
    
    # 10-year prediction (static)
    print(f"\nProcessing 10-year predictions for {washout_key}...")
    results_10yr = evaluate_major_diseases_wsex_with_bootstrap_from_pi(
        pi=pi_full,
        Y_100k=Y_subset,
        E_100k=E_subset,
        disease_names=disease_names,
        pce_df=pce_df_subset,
        n_bootstraps=n_bootstraps,
        follow_up_duration_years=10
    )
    current_results['10yr'] = results_10yr
    
    results_df_10yr = pd.DataFrame({
        'Disease': list(results_10yr.keys()),
        'AUC': [r['auc'] for r in results_10yr.values()],
        'CI_lower': [r['ci_lower'] for r in results_10yr.values()],
        'CI_upper': [r['ci_upper'] for r in results_10yr.values()],
        'N_Events': [r['n_events'] for r in results_10yr.values()],
        'Event_Rate': [r['event_rate'] for r in results_10yr.values()]
    })
    results_df_10yr = results_df_10yr.set_index('Disease').sort_values('AUC', ascending=False)
    output_file_10yr = output_dir / f'washout_{washout_key}_10yr_results.csv'
    results_df_10yr.to_csv(output_file_10yr)
    print(f"✓ Saved 10-year results to {output_file_10yr}")
    
    all_results[washout_key] = current_results

# Create combined comparison file
print(f"\n{'='*80}")
print("CREATING COMBINED WASHOUT COMPARISON FILE")
print(f"{'='*80}")

all_diseases = sorted(list(disease_names.keys()))
comparison_df = pd.DataFrame(index=all_diseases)

for washout_key in washout_configs.keys():
    if washout_key in all_results:
        for horizon_key, results_dict in all_results[washout_key].items():
            comparison_df[f'{washout_key}_{horizon_key}_AUC'] = [results_dict.get(d, {}).get('auc', np.nan) for d in all_diseases]
            comparison_df[f'{washout_key}_{horizon_key}_CI_lower'] = [results_dict.get(d, {}).get('ci_lower', np.nan) for d in all_diseases]
            comparison_df[f'{washout_key}_{horizon_key}_CI_upper'] = [results_dict.get(d, {}).get('ci_upper', np.nan) for d in all_diseases]
    else:
        print(f"Skipping {washout_key} in combined comparison as it failed to process.")

comparison_file = output_dir / 'washout_comparison_1yr_2yr_3yr_vs_baseline.csv'
comparison_df.to_csv(comparison_file)
print(f"✓ Saved combined comparison to {comparison_file}")

# Calculate AUC drops from baseline
print(f"\n{'='*80}")
print("CALCULATING AUC DROPS FROM BASELINE")
print(f"{'='*80}")

if 'no_washout' in all_results:
    baseline_1yr = {d: all_results['no_washout']['1yr'].get(d, {}).get('auc', np.nan) for d in all_diseases}
    baseline_10yr = {d: all_results['no_washout']['10yr'].get(d, {}).get('auc', np.nan) for d in all_diseases}
    
    drops_df = pd.DataFrame(index=all_diseases)
    
    for washout_key in ['1yr', '2yr', '3yr']:
        if washout_key in all_results:
            drops_df[f'{washout_key}_1yr_drop'] = [
                baseline_1yr[d] - all_results[washout_key]['1yr'].get(d, {}).get('auc', np.nan) 
                for d in all_diseases
            ]
            drops_df[f'{washout_key}_10yr_drop'] = [
                baseline_10yr[d] - all_results[washout_key]['10yr'].get(d, {}).get('auc', np.nan) 
                for d in all_diseases
            ]
    
    drops_file = output_dir / 'washout_1yr_2yr_3yr_auc_drops.csv'
    drops_df.to_csv(drops_file)
    print(f"✓ Saved AUC drops to {drops_file}")
    
    # Print summary
    print(f"\nSummary of AUC drops (baseline - washout):")
    print(f"{'Disease':<25} {'1yr_1yr':<10} {'1yr_10yr':<10} {'2yr_1yr':<10} {'2yr_10yr':<10} {'3yr_1yr':<10} {'3yr_10yr':<10}")
    print("-" * 90)
    for disease in all_diseases[:10]:  # Show first 10
        row = [disease[:24]]
        for washout_key in ['1yr', '2yr', '3yr']:
            if washout_key in all_results:
                row.append(f"{drops_df.loc[disease, f'{washout_key}_1yr_drop']:.4f}")
                row.append(f"{drops_df.loc[disease, f'{washout_key}_10yr_drop']:.4f}")
            else:
                row.append("N/A")
                row.append("N/A")
        print(f"{row[0]:<25} {row[1]:<10} {row[2]:<10} {row[3]:<10} {row[4]:<10} {row[5]:<10} {row[6]:<10}")

print(f"\n{'='*80}")
print("EVALUATION COMPLETE!")
print(f"{'='*80}")
print(f"\nResults saved to: {output_dir}")
print(f"  - Individual results: washout_*_1yr_results.csv, washout_*_10yr_results.csv")
print(f"  - Combined comparison: washout_comparison_1yr_2yr_3yr_vs_baseline.csv")
print(f"  - AUC drops: washout_1yr_2yr_3yr_auc_drops.csv")

